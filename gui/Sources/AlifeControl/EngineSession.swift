import AlifeCore
import Foundation

/// One engine process that speaks the binary stream (`alife stream` for the Lab's 3D viewer, `alife play` for the World
/// window) and the handles to it: the process, its stdin, the mailbox its reader thread fills, and the stop ladder.
/// Both controllers hold their sessions through it (DESIGN §33.4.1).
///
/// Main-actor isolated, so it is `Sendable`. The reader thread (`StreamReader`) is started by `launch`; the controller
/// consumes what it leaves through `pump`.
@MainActor
final class EngineSession {
    /// How a process is brought down, once: a `quit` line and EOF on its stdin, then (if it is still there) SIGINT after
    /// `beforeInterrupt`, then SIGTERM after `beforeTerminate` more. A stream has nothing to save, so it skips SIGINT and
    /// waits two seconds; a played world saves a checkpoint on `quit` (686 MB), so it gets 30 s, then the runner's own
    /// SIGINT path (which works while paused), then 10 s.
    struct Ladder: Equatable, Sendable {
        /// How long after the `quit` before SIGINT; nil: no SIGINT rung.
        var beforeInterrupt: Duration?
        /// How long after the last rung before SIGTERM.
        var beforeTerminate: Duration

        static let stream = Ladder(beforeInterrupt: nil, beforeTerminate: .seconds(2))
        static let play = Ladder(beforeInterrupt: .seconds(30), beforeTerminate: .seconds(10))
    }

    /// Everything `launch` creates: the session and the streams its controller pumps.
    struct Launched {
        let session: EngineSession
        let wakeups: AsyncStream<Void>
        let stderr: AsyncStream<Data>
        let exits: AsyncStream<(Int32, Process.TerminationReason)>
    }

    let process: Process
    let mailbox: StreamMailbox
    private(set) var stdin: FileHandle?
    /// The app asked for the end (as opposed to the stream ending, or being cut off for being corrupt).
    var stopRequested = false
    /// The last non-blank line of the process's stderr: what it says went wrong.
    var lastError: String?
    /// How the reader saw the stream end.
    var end: StreamEnd?
    private var ladderTask: Task<Void, Never>?

    init(process: Process, stdin: FileHandle, mailbox: StreamMailbox) {
        self.process = process
        self.stdin = stdin
        self.mailbox = mailbox
    }

    var isRunning: Bool { process.isRunning }

    /// The folder to run in is gone (reported in the repository status line's words).
    struct MissingFolder: LocalizedError {
        let path: String
        var errorDescription: String? { RepoFolder.missingMessage(path) }
    }

    /// Starts `executable` with `arguments` in `directory`, its stdout drained by a reader thread into a mailbox. Throws
    /// if the process cannot be launched.
    static func launch(executable: URL, arguments: [String], directory: URL) throws -> Launched {
        // A missing working directory makes `Process.run` throw "The file <last component> doesn't exist.", which names
        // neither the folder nor the problem. "Open 3D viewer" and the World window read `repoReady`, which can be stale.
        guard RepoFolder.isDirectory(directory.path) else { throw MissingFolder(path: directory.path) }
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.currentDirectoryURL = directory
        process.environment = RunController.engineEnvironment()
        let inPipe = Pipe()
        let outPipe = Pipe()
        let errPipe = Pipe()
        inPipe.fileHandleForWriting.suppressSIGPIPE()
        process.standardInput = inPipe
        process.standardOutput = outPipe
        process.standardError = errPipe
        let (exits, exitSink) = AsyncStream.makeStream(of: (Int32, Process.TerminationReason).self)
        process.terminationHandler = { process in
            exitSink.yield((process.terminationStatus, process.terminationReason))
            exitSink.finish()
        }
        try process.run()
        let mailbox = StreamMailbox()
        let (wakeups, wake) = AsyncStream.makeStream(of: Void.self, bufferingPolicy: .bufferingNewest(1))
        StreamReader.start(reading: outPipe.fileHandleForReading, into: mailbox, wake: wake)
        let errorChunks = RunController.chunks(from: errPipe)
        let session = EngineSession(process: process, stdin: inPipe.fileHandleForWriting, mailbox: mailbox)
        return Launched(session: session, wakeups: wakeups, stderr: errorChunks, exits: exits)
    }

    // MARK: Writing

    /// Writes `data` (a control line) to the process; false if its stdin is closed.
    func write(_ data: Data) throws -> Bool {
        guard let stdin else { return false }
        try stdin.write(contentsOf: data)
        return true
    }

    func closeStdin() {
        try? stdin?.close()
        stdin = nil
    }

    // MARK: Stopping

    /// The `quit` line (when the grammar has one) and EOF on stdin, then the rungs of `ladder` while the process is still
    /// running. Once only.
    func requestStop(quit: Data?, ladder: Ladder) {
        guard !stopRequested else { return }
        stopRequested = true
        if let quit { try? stdin?.write(contentsOf: quit) }
        closeStdin()
        startLadder(ladder)
    }

    /// SIGINT at once and SIGTERM after `terminateAfter`: for a process that reads no stdin yet (a planet being built:
    /// the fork and the spin-up do not look at it). Once only.
    func interruptNow(terminateAfter: Duration) {
        guard !stopRequested else { return }
        stopRequested = true
        closeStdin()
        if process.isRunning { process.interrupt() }
        startLadder(Ladder(beforeInterrupt: nil, beforeTerminate: terminateAfter))
    }

    private func startLadder(_ ladder: Ladder) {
        let process = process
        ladderTask = Task {
            if let before = ladder.beforeInterrupt {
                try? await Task.sleep(for: before)
                guard !Task.isCancelled, process.isRunning else { return }
                process.interrupt()
            }
            try? await Task.sleep(for: ladder.beforeTerminate)
            if !Task.isCancelled, process.isRunning { process.terminate() }
        }
    }

    /// SIGTERM now (the app is quitting, or the stream is corrupt).
    func terminate() {
        closeStdin()
        ladderTask?.cancel()
        if process.isRunning { process.terminate() }
    }

    func release() {
        ladderTask?.cancel()
        closeStdin()
    }

    // MARK: Pumping

    /// Runs until the process has exited and both of its pipes have reached end of file, then reports the exit. The exit
    /// is only looked at after the frames: the last frame and the exit status cannot race. `deliver` takes what the
    /// reader left in the mailbox (after each wake-up, and once more at the end), `lines` the stderr lines, `finished`
    /// the status.
    func pump(_ launched: Launched,
              deliver: @escaping @MainActor (StreamDelivery) -> Void,
              lines: @escaping @MainActor ([String]) -> Void,
              finished: @MainActor (Int32, Process.TerminationReason) -> Void) async {
        async let deliveries: Void = consume(launched.wakeups, deliver)
        async let errors: Void = drain(launched.stderr, lines)
        _ = await (deliveries, errors)
        for await (status, reason) in launched.exits { finished(status, reason) }  // yields exactly once
    }

    private func consume(_ wakeups: AsyncStream<Void>, _ deliver: @escaping @MainActor (StreamDelivery) -> Void) async {
        for await _ in wakeups { deliver(mailbox.take()) }
        deliver(mailbox.take())  // what the last wake-up may have coalesced away
    }

    private func drain(_ chunks: AsyncStream<Data>, _ lines: @escaping @MainActor ([String]) -> Void) async {
        var splitter = LineSplitter()
        for await data in chunks { note(splitter.feed(data), lines) }
        if let rest = splitter.flush() { note([rest], lines) }
    }

    private func note(_ batch: [String], _ lines: @MainActor ([String]) -> Void) {
        guard !batch.isEmpty else { return }
        if let last = batch.last(where: { !$0.trimmingCharacters(in: .whitespaces).isEmpty }) { lastError = last }
        lines(batch)
    }
}

extension FileHandle {
    /// Makes a write to this descriptor fail with EPIPE (which `write(contentsOf:)` throws) when the reader has gone,
    /// instead of delivering SIGPIPE, which ends the process. The app also ignores SIGPIPE as a whole
    /// (`AppDelegate`); this keeps the descriptor safe on its own, wherever the code runs.
    func suppressSIGPIPE() {
        _ = fcntl(fileDescriptor, F_SETNOSIGPIPE, 1)
    }
}
