import AlifeCore
import Foundation
import Testing
import os

@testable import AlifeControl

/// The reader thread and the mailbox, on a real pipe and the committed fixtures. No process is started.
@Suite struct StreamReaderTests {
    /// Starts a reader on a new pipe, writes `bytes` in pieces of `chunk` from another thread, closes the pipe, and
    /// returns the mailbox once the reader has finished, with how many wake-ups the main side saw.
    private func read(_ bytes: [UInt8], chunk: Int = 4096, closeEarly: Bool = false) async -> (StreamDelivery, wakes: Int) {
        let pipe = Pipe()
        let mailbox = StreamMailbox()
        let (wakeups, wake) = AsyncStream.makeStream(of: Void.self, bufferingPolicy: .bufferingNewest(1))
        StreamReader.start(reading: pipe.fileHandleForReading, into: mailbox, wake: wake, chunkSize: 1 << 12)
        let writer = pipe.fileHandleForWriting
        let writing = Task.detached {
            var at = 0
            while at < bytes.count {
                let end = min(at + chunk, bytes.count)
                try? writer.write(contentsOf: Data(bytes[at..<end]))
                at = end
            }
            try? writer.close()
        }
        var wakes = 0
        for await _ in wakeups { wakes += 1 }  // ends when the reader has reached end of file
        await writing.value
        return (mailbox.take(), wakes)
    }

    @Test(arguments: ["2d", "3d"])
    func aFixtureArrivesWhole(kind: String) async throws {
        let fixture = try StreamFixtureData(kind: kind)
        // Chunks of 7 bytes cut every record and every number; the fixture has 0x0A and 0x0D bytes inside frames.
        let (delivery, wakes) = await read(fixture.bytes, chunk: 7)
        #expect(wakes >= 1)
        #expect(delivery.header?.dims == (kind == "3d" ? 3 : 2))
        #expect(delivery.end == .complete)
        #expect(delivery.received == fixture.frameCount)
        // Only the newest frame is kept; the others were replaced before anyone took them.
        #expect(delivery.dropped == fixture.frameCount - 1)
        let frame = try #require(delivery.frame)
        #expect(Int(frame.step) == fixture.lastStep)
        #expect(frame.rows == 64)
    }

    @Test(arguments: ["2d", "3d"])
    func theFixtureHasTheBytesALineSplitterWouldCut(kind: String) throws {
        // The reason the stream cannot use the line path of RunController, which splits on 0x0A (and 0x0D).
        let bytes = try Repo.streamFixture(kind)
        #expect(bytes.contains(0x0A))
        var splitter = LineSplitter()
        let lines = splitter.feed(Data(bytes))
        #expect(lines.count > 0)
    }

    @Test func aStreamThatEndsInsideARecordIsTruncated() async throws {
        let fixture = try StreamFixtureData(kind: "3d")
        let (delivery, _) = await read(Array(fixture.bytes.dropLast(5)))
        #expect(delivery.header != nil)
        guard case .failed(.truncated)? = delivery.end else {
            Issue.record("expected a truncation, got \(String(describing: delivery.end))")
            return
        }
        #expect(delivery.received == fixture.frameCount - 1)
    }

    @Test func aStreamThatIsNotOursIsRefused() async {
        let (delivery, _) = await read(Array("hello, this is not a stream\n".utf8))
        #expect(delivery.header == nil)
        #expect(delivery.end == .failed(.badMagic))
    }

    @Test func theReaderKeepsDrainingAfterAnError() async {
        // A pipe holds 64 KiB; if the reader stopped at the bad magic, the writer thread would block on the second
        // megabyte and this test would never finish.
        let junk = [UInt8](repeating: 0x41, count: 2 << 20)
        let (delivery, _) = await read(junk, chunk: 1 << 16)
        #expect(delivery.end == .failed(.badMagic))
    }

    @Test func anEmptyStreamIsTruncated() async {
        let (delivery, _) = await read([])
        guard case .failed(.truncated)? = delivery.end else {
            Issue.record("expected a truncation, got \(String(describing: delivery.end))")
            return
        }
    }

    // MARK: Wake-ups and early errors

    private func u32(_ bytes: [UInt8], _ offset: Int) -> Int {
        (0..<4).reduce(0) { $0 | Int(bytes[offset + $1]) << (8 * $1) }
    }

    /// A reader on a pipe that stays open until the test closes it, and the wake-ups it has sent.
    private struct OpenPipe: Sendable {
        let writer: FileHandle
        let mailbox = StreamMailbox()
        let wakes = OSAllocatedUnfairLock(initialState: 0)

        init() {
            let pipe = Pipe()
            writer = pipe.fileHandleForWriting
            let (wakeups, wake) = AsyncStream.makeStream(of: Void.self, bufferingPolicy: .bufferingNewest(1))
            StreamReader.start(reading: pipe.fileHandleForReading, into: mailbox, wake: wake, chunkSize: 1 << 12)
            let counter = wakes
            Task.detached {
                for await _ in wakeups { counter.withLock { $0 += 1 } }
            }
        }

        func write(_ bytes: ArraySlice<UInt8>) throws { try writer.write(contentsOf: Data(bytes)) }
        func close() { try? writer.close() }
    }

    @Test func aChunkThatCompletesNothingWakesNobody() async throws {
        let fixture = try StreamFixtureData(kind: "3d")
        let bytes = fixture.bytes
        let headerEnd = 12 + u32(bytes, 8)
        let frameEnd = headerEnd + 8 + u32(bytes, headerEnd + 4)
        let open = OpenPipe()
        defer { open.close() }

        try open.write(bytes[0..<headerEnd])
        #expect(await eventually { open.wakes.withLock { $0 } == 1 })
        #expect(open.mailbox.take().header != nil)

        // The frame in three pieces: the first two are the middle of it, and nothing in them is worth a redraw.
        let third = (frameEnd - headerEnd) / 3
        try open.write(bytes[headerEnd..<(headerEnd + third)])
        try open.write(bytes[(headerEnd + third)..<(headerEnd + 2 * third)])
        try await Task.sleep(for: .milliseconds(300))
        #expect(open.wakes.withLock { $0 } == 1, "a partial frame woke the main actor")
        #expect(open.mailbox.take().isEmpty)

        try open.write(bytes[(headerEnd + 2 * third)..<frameEnd])
        #expect(await eventually { open.wakes.withLock { $0 } == 2 })
        #expect(open.mailbox.take().frame != nil)
    }

    @Test func aDecoderErrorIsPublishedAtOnceNotWhenTheNextBytesArrive() async throws {
        // The header, a good frame and the start of a second whose length is wrong, in one write; nothing follows (a
        // paused stream, a followed run that has stopped). The error must not wait for more bytes or for end of file.
        let fixture = try StreamFixtureData(kind: "3d")
        let bytes = fixture.bytes
        let headerEnd = 12 + u32(bytes, 8)
        let second = headerEnd + 8 + u32(bytes, headerEnd + 4)
        var broken = Array(bytes[0..<(second + 8 + 32)])
        let wrong = u32(bytes, second + 4) + 4
        for k in 0..<4 { broken[second + 4 + k] = UInt8((wrong >> (8 * k)) & 0xFF) }

        let open = OpenPipe()
        defer { open.close() }
        try open.write(broken[...])
        var delivery = StreamDelivery()
        let arrived = await eventually {
            let more = open.mailbox.take()
            delivery.header = delivery.header ?? more.header
            delivery.frame = more.frame ?? delivery.frame
            delivery.end = more.end ?? delivery.end
            return delivery.end != nil
        }
        #expect(arrived, "the pipe is still open and the error has not been seen")
        #expect(delivery.header != nil && delivery.frame != nil, "the events before the error are delivered")
        guard case .failed(.lengthMismatch)? = delivery.end else {
            Issue.record("expected a length mismatch, got \(String(describing: delivery.end))")
            return
        }
    }

    @Test func theMailboxKeepsTheNewestFrameAndEmptiesWhenTaken() throws {
        let fixture = try StreamFixtureData(kind: "2d")
        let mailbox = StreamMailbox()
        for frame in fixture.frames { mailbox.publish(frame: frame) }
        let first = mailbox.take()
        #expect(first.frame?.step == fixture.frames.last?.step)
        #expect(first.dropped == fixture.frameCount - 1)
        #expect(first.received == fixture.frameCount)
        #expect(mailbox.take().isEmpty)
    }
}

/// The frames of a committed fixture, decoded in one piece.
struct StreamFixtureData {
    let bytes: [UInt8]
    let frames: [StreamFrame]

    init(kind: String) throws {
        bytes = try Repo.streamFixture(kind)
        var decoder = FrameDecoder()
        let events = try decoder.feed(Data(bytes))
        try decoder.finish()
        frames = events.compactMap { if case .frame(let frame) = $0 { frame } else { nil } }
        header = events.compactMap { if case .header(let header) = $0 { header } else { nil } }[0]
    }

    let header: StreamHeader
    var frameCount: Int { frames.count }
    var lastStep: Int { Int(frames[frames.count - 1].step) }
}
