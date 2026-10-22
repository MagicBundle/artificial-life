import AlifeCore
import Foundation
import os

// The binary half of the viewer's pipe. `alife stream` writes its frames to stdout, and a frame can be megabytes, so
// the bytes never go through the line-oriented path of `RunController` (which splits on 0x0A and 0x0D, bytes that
// occur inside any frame). A dedicated thread blocks in `read`, so the pipe is drained as fast as the engine writes
// (a full 64 KiB pipe would stall the simulation) whatever the main actor is doing. It feeds a `FrameDecoder` and
// leaves the newest frame in a mailbox; the main actor is woken, with wake-ups coalesced, and takes what is there.

/// How a stream ended, as the reader saw it.
enum StreamEnd: Sendable, Equatable {
    /// End of input between two records: a complete stream.
    case complete
    /// The decoder refused the stream, or it ended inside a record (the process died).
    case failed(StreamError)
    /// The pipe could not be read.
    case unreadable(String)
}

/// What the main actor takes from the mailbox: everything since its last `take()`.
struct StreamDelivery: Sendable, Equatable {
    var header: StreamHeader?
    /// The newest frame; older ones that were not taken are gone.
    var frame: StreamFrame?
    var end: StreamEnd?
    /// Frames that arrived and were replaced by a newer one before they were taken.
    var dropped = 0
    /// Frames that arrived (taken or replaced).
    var received = 0

    var isEmpty: Bool { header == nil && frame == nil && end == nil }
}

/// Holds the latest header and frame between the reader thread and the main actor. All access is under one lock;
/// nothing blocks for long (a frame is stored by reference to its arrays).
final class StreamMailbox: Sendable {
    private let state = OSAllocatedUnfairLock(initialState: StreamDelivery())

    func publish(header: StreamHeader) {
        state.withLock { $0.header = header }
    }

    func publish(frame: StreamFrame) {
        state.withLock {
            if $0.frame != nil { $0.dropped += 1 }
            $0.frame = frame
            $0.received += 1
        }
    }

    func publish(end: StreamEnd) {
        state.withLock { $0.end = end }
    }

    /// Everything published since the last call.
    func take() -> StreamDelivery {
        state.withLock {
            let delivery = $0
            $0 = StreamDelivery()
            return delivery
        }
    }
}

enum StreamReader {
    /// Reads `handle` to end of file on a new thread, decoding what it reads into `mailbox` and yielding to `wake` after
    /// each chunk that published something (a header, a frame, the end; a chunk that completes none of them, the middle
    /// of a frame, wakes nobody) and once more when the input ends (then it finishes `wake`). `wake` should buffer one
    /// element: yields that arrive while the main actor has not yet looked collapse into one.
    static func start(reading handle: FileHandle, into mailbox: StreamMailbox, wake: AsyncStream<Void>.Continuation,
                      chunkSize: Int = 1 << 18) {
        let thread = Thread {
            run(reading: handle, into: mailbox, wake: wake, chunkSize: chunkSize)
        }
        thread.name = "alife-stream-reader"
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    /// The reader's loop; `start` runs it on its own thread. It keeps draining after a decoder error (the process is
    /// being stopped, and a pipe nobody reads would block it), discarding the bytes.
    ///
    /// It calls `read(2)` itself: `FileHandle.read(upToCount:)` may wait for the whole count on a pipe, which for a
    /// 256 KiB buffer is longer than a small frame takes to arrive and, for a paused stream, never. `read(2)` returns
    /// what is there.
    static func run(reading handle: FileHandle, into mailbox: StreamMailbox, wake: AsyncStream<Void>.Continuation,
                    chunkSize: Int) {
        var decoder = FrameDecoder()
        var failed = false
        let descriptor = handle.fileDescriptor
        let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: max(chunkSize, 1), alignment: 16)
        defer {
            buffer.deallocate()
            try? handle.close()
            wake.yield()
            wake.finish()
        }
        while true {
            let count = read(descriptor, buffer.baseAddress, buffer.count)
            if count < 0 {
                if errno == EINTR { continue }
                if !failed { mailbox.publish(end: .unreadable(String(cString: strerror(errno)))) }
                return
            }
            if count == 0 { break }  // end of file: the process closed its stdout, or exited
            if failed { continue }
            var published = false
            do {
                for event in try decoder.feed(Data(bytes: buffer.baseAddress!, count: count)) {
                    switch event {
                    case .header(let header):
                        mailbox.publish(header: header)
                        published = true
                    case .frame(let frame):
                        mailbox.publish(frame: frame)
                        published = true
                    case .record: break  // a play record: the World window's mailbox takes these (§33.1)
                    case .unknownRecord: break  // a record of a later compatible version
                    }
                }
                // `feed` returns the events before an error and throws the error from the next call. This chunk may
                // be the last for a long time (a paused stream, a followed run that has stopped), so publish it now.
                if let error = decoder.failure {
                    mailbox.publish(end: .failed(error))
                    failed = true
                    published = true
                }
            } catch let error as StreamError {
                mailbox.publish(end: .failed(error))
                failed = true
                published = true
            } catch {
                mailbox.publish(end: .unreadable("\(error)"))
                failed = true
                published = true
            }
            if published { wake.yield() }
        }
        if failed { return }
        do {
            try decoder.finish()
            mailbox.publish(end: .complete)
        } catch let error as StreamError {
            mailbox.publish(end: .failed(error))
        } catch {
            mailbox.publish(end: .unreadable("\(error)"))
        }
    }
}
