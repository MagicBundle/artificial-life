import AppKit
import Foundation

enum Job {
    case run, inspect, setup
}

struct LogLine: Identifiable {
    let id: Int
    let text: String
    let isError: Bool
}

struct ExitInfo {
    let text: String
    let success: Bool
}

/// Splits a byte stream into lines (on "\n" or "\r", so progress bars become lines too).
/// Each instance is used from a single pipe handler, so it needs no locking.
private final class LineSplitter {
    private var buffer = Data()

    func feed(_ data: Data) -> [String] {
        buffer.append(data)
        var lines: [String] = []
        while let index = buffer.firstIndex(where: { $0 == 0x0A || $0 == 0x0D }) {
            let lineData = buffer[buffer.startIndex..<index]
            if !lineData.isEmpty { lines.append(String(decoding: lineData, as: UTF8.self)) }
            buffer.removeSubrange(buffer.startIndex..<(index + 1))
        }
        return lines
    }

    func flush() -> String? {
        defer { buffer.removeAll() }
        return buffer.isEmpty ? nil : String(decoding: buffer, as: UTF8.self)
    }
}

/// Owns the engine subprocesses and all UI-visible run state.
/// Everything here is touched on the main thread only (pipe and termination handlers hop back to main).
final class RunController: ObservableObject {
    static let shared = RunController()
    private static let repoKey = "repoPath"
    private static let maxLogLines = 5000

    // Repository
    @Published var repoPath: String {
        didSet {
            UserDefaults.standard.set(repoPath, forKey: Self.repoKey)
            refreshRepo()
        }
    }
    @Published private(set) var repoReady = false
    @Published private(set) var runs: [String] = []

    // Settings and job state
    @Published var settings = RunSettings()
    @Published private(set) var job: Job?
    @Published private(set) var stopRequested = false
    @Published private(set) var viewerRunning = false
    @Published private(set) var lastExit: ExitInfo?

    // Live status
    @Published private(set) var metrics: MetricsSnapshot?
    @Published private(set) var elapsed: TimeInterval = 0
    @Published private(set) var remaining: TimeInterval?
    @Published private(set) var logLines: [LogLine] = []

    private var process: Process?
    private var viewerProcess: Process?
    private var activity: NSObjectProtocol?
    private var startedAt: Date?
    private var deadline: Date?
    private var trackedRun = ""
    private var nextLogID = 0
    private var tickCount = 0
    private var ticker: Timer?
    private let pollQueue = DispatchQueue(label: "alife.metrics", qos: .utility)

    private init() {
        repoPath = UserDefaults.standard.string(forKey: Self.repoKey) ?? "~/artificial-life"
        settings.name = Self.defaultRunName()
        refreshRepo()
        ticker = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            self?.tick()
        }
    }

    // MARK: Derived values

    var repoURL: URL {
        URL(fileURLWithPath: (repoPath as NSString).expandingTildeInPath, isDirectory: true)
    }

    var alifeURL: URL { repoURL.appendingPathComponent(".venv/bin/alife") }

    var validationErrors: [String] { settings.validate(existingRuns: Set(runs)) }

    var canStart: Bool { job == nil && repoReady && validationErrors.isEmpty }

    var targetRunExists: Bool {
        repoReady && !settings.targetRun.isEmpty && runs.contains(settings.targetRun)
    }

    /// The exact command that Start will execute.
    var commandLine: String {
        let args = settings.arguments().map(shellQuote).joined(separator: " ")
        return "cd \(shellQuote(repoPath)) && .venv/bin/alife \(args)"
    }

    // MARK: Repository

    func refreshRepo() {
        repoReady = FileManager.default.isExecutableFile(atPath: alifeURL.path)
        refreshRuns()
    }

    func chooseRepo() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = repoURL
        panel.prompt = "Choose"
        panel.message = "Select the artificial-life repository folder"
        if panel.runModal() == .OK, let url = panel.url {
            repoPath = url.path
        }
    }

    func refreshRuns() {
        let runsDir = repoURL.appendingPathComponent("runs", isDirectory: true)
        let keys: [URLResourceKey] = [.isDirectoryKey, .contentModificationDateKey]
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: runsDir, includingPropertiesForKeys: keys, options: [.skipsHiddenFiles])) ?? []
        let entries: [(name: String, date: Date)] = urls.compactMap { url in
            let values = try? url.resourceValues(forKeys: Set(keys))
            guard values?.isDirectory == true else { return nil }
            return (url.lastPathComponent, values?.contentModificationDate ?? .distantPast)
        }
        let names = entries.sorted { $0.date > $1.date }.map { $0.name }
        if names != runs { runs = names }
        if !runs.contains(settings.sourceRun) {
            let fallback = runs.first ?? ""
            if settings.sourceRun != fallback { settings.sourceRun = fallback }
        }
    }

    private static func defaultRunName() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmm"
        return "run-" + formatter.string(from: Date())
    }

    func suggestNewName() { settings.name = Self.defaultRunName() }

    // MARK: Jobs

    func start() {
        guard canStart else { return }
        let args = settings.arguments()
        beginJob(.run, executable: alifeURL, arguments: args, keepAwake: true)
        guard job == .run else { return }
        trackedRun = settings.targetRun
        if settings.timerEnabled {
            deadline = Date().addingTimeInterval(settings.timerDuration)
            remaining = settings.timerDuration
        }
    }

    func inspect() {
        guard job == nil, repoReady, targetRunExists else { return }
        beginJob(.inspect, executable: alifeURL,
                 arguments: ["inspect", "runs/\(settings.targetRun)", "--check"], keepAwake: false)
    }

    func setupEnvironment() {
        guard job == nil else { return }
        guard let uv = findUV() else {
            appendLines(["Could not find 'uv' in /opt/homebrew/bin, /usr/local/bin, ~/.local/bin or PATH. "
                         + "Install it from https://docs.astral.sh/uv/ and try again."], isError: true)
            return
        }
        beginJob(.setup, executable: uv, arguments: ["sync", "--extra", "viewer", "--extra", "video"],
                 keepAwake: false)
    }

    /// Single SIGINT: the engine finishes the step, writes a checkpoint and exits.
    func stop() {
        guard job == .run, !stopRequested, let p = process, p.isRunning else { return }
        stopRequested = true
        appendLines(["Stop requested (one SIGINT); waiting for the engine to checkpoint and exit..."], isError: false)
        p.interrupt()
    }

    /// SIGTERM: ends the process immediately, without a final checkpoint.
    func forceStop() {
        guard let p = process, p.isRunning else { return }
        appendLines(["Force stop (SIGTERM)."], isError: true)
        p.terminate()
    }

    private func beginJob(_ kind: Job, executable: URL, arguments: [String], keepAwake: Bool) {
        lastExit = nil
        stopRequested = false
        elapsed = 0
        remaining = nil
        deadline = nil
        logLines.removeAll()
        appendLines(["$ " + ([executable.path] + arguments).map(shellQuote).joined(separator: " ")], isError: false)
        let launched = launch(executable, arguments, prefix: "") { [weak self] status, reason in
            self?.finish(kind, status: status, reason: reason)
        }
        guard let p = launched else { return }
        process = p
        job = kind
        startedAt = Date()
        if keepAwake {
            activity = ProcessInfo.processInfo.beginActivity(
                options: [.idleSystemSleepDisabled, .userInitiated], reason: "alife run in progress")
        }
    }

    private func finish(_ kind: Job, status: Int32, reason: Process.TerminationReason) {
        if let activity {
            ProcessInfo.processInfo.endActivity(activity)
            self.activity = nil
        }
        process = nil
        job = nil
        deadline = nil
        remaining = nil
        let info = Self.describe(status: status, reason: reason, kind: kind)
        lastExit = info
        appendLines([info.text], isError: !info.success)
        stopRequested = false
        refreshRepo()
        pollMetrics()
    }

    private static func describe(status: Int32, reason: Process.TerminationReason, kind: Job) -> ExitInfo {
        if reason == .uncaughtSignal {
            return ExitInfo(text: "Terminated by signal \(status).", success: false)
        }
        switch status {
        case 0: return ExitInfo(text: "Finished: exit code 0.", success: true)
        case 130: return ExitInfo(text: "Stopped by interrupt: exit code 130 (checkpoint written).", success: true)
        case 1: return ExitInfo(text: "Failed: exit code 1 (see log).", success: false)
        case 2: return ExitInfo(text: "Configuration or usage error: exit code 2.", success: false)
        case 3:
            let what = kind == .inspect ? "Invariant check found problems" : "Invariant violation"
            return ExitInfo(text: "\(what): exit code 3.", success: false)
        case 4: return ExitInfo(text: "Checkpoint error: exit code 4.", success: false)
        default: return ExitInfo(text: "Exited with code \(status).", success: false)
        }
    }

    // MARK: Viewer, Finder

    /// The viewer is a separate process and competes with a running simulation for the GPU.
    func openViewer() {
        guard repoReady, viewerProcess == nil, targetRunExists else { return }
        let args = ["view", "runs/\(settings.targetRun)"]
        appendLines(["$ " + ([alifeURL.path] + args).map(shellQuote).joined(separator: " ")], isError: false)
        let launched = launch(alifeURL, args, prefix: "[viewer] ") { [weak self] status, _ in
            self?.viewerProcess = nil
            self?.viewerRunning = false
            self?.appendLines(["[viewer] exited with code \(status)."], isError: status != 0)
        }
        guard let p = launched else { return }
        viewerProcess = p
        viewerRunning = true
    }

    func closeViewer() { viewerProcess?.terminate() }

    func revealInFinder() {
        let url = repoURL.appendingPathComponent("runs/\(settings.targetRun)", isDirectory: true)
        if !settings.targetRun.isEmpty, FileManager.default.fileExists(atPath: url.path) {
            NSWorkspace.shared.activateFileViewerSelecting([url])
        } else {
            NSWorkspace.shared.activateFileViewerSelecting([repoURL])
        }
    }

    /// Called when the app quits: ask a running simulation to stop cleanly (it outlives the app long enough
    /// to write its checkpoint).
    func shutdown() {
        stop()
        viewerProcess?.terminate()
    }

    // MARK: Log

    func clearLog() { logLines.removeAll() }

    var logText: String { logLines.map(\.text).joined(separator: "\n") }

    private func appendLines(_ lines: [String], isError: Bool) {
        guard !lines.isEmpty else { return }
        for text in lines {
            logLines.append(LogLine(id: nextLogID, text: text, isError: isError))
            nextLogID += 1
        }
        if logLines.count > Self.maxLogLines {
            logLines.removeFirst(logLines.count - Self.maxLogLines)
        }
    }

    // MARK: Process plumbing

    /// Starts `executable` in the repo directory, streaming its output into the log. `onExit` runs on the main
    /// thread after the process has exited and both pipes have been drained.
    private func launch(_ executable: URL, _ arguments: [String], prefix: String,
                        onExit: @escaping (Int32, Process.TerminationReason) -> Void) -> Process? {
        let p = Process()
        p.executableURL = executable
        p.arguments = arguments
        p.currentDirectoryURL = repoURL
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        let extra = ["/opt/homebrew/bin", "/usr/local/bin", NSHomeDirectory() + "/.local/bin"]
        let existing = (env["PATH"] ?? "/usr/bin:/bin").split(separator: ":").map(String.init)
        env["PATH"] = (existing + extra.filter { !existing.contains($0) }).joined(separator: ":")
        p.environment = env
        p.standardInput = FileHandle.nullDevice
        let outPipe = Pipe()
        let errPipe = Pipe()
        p.standardOutput = outPipe
        p.standardError = errPipe

        let group = DispatchGroup()
        group.enter()  // termination
        p.terminationHandler = { _ in group.leave() }
        do {
            try p.run()
        } catch {
            group.leave()
            appendLines(["Failed to launch \(executable.path): \(error.localizedDescription)"], isError: true)
            return nil
        }
        attach(outPipe, isError: false, prefix: prefix, group: group)
        attach(errPipe, isError: true, prefix: prefix, group: group)
        group.notify(queue: .main) {
            onExit(p.terminationStatus, p.terminationReason)
        }
        return p
    }

    private func attach(_ pipe: Pipe, isError: Bool, prefix: String, group: DispatchGroup) {
        let splitter = LineSplitter()
        group.enter()
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty {  // EOF
                handle.readabilityHandler = nil
                let rest = splitter.flush()
                DispatchQueue.main.async {
                    if let rest { self?.appendLines([prefix + rest], isError: isError) }
                    group.leave()
                }
                return
            }
            let lines = splitter.feed(data).map { prefix + $0 }
            if !lines.isEmpty {
                DispatchQueue.main.async { self?.appendLines(lines, isError: isError) }
            }
        }
    }

    private func findUV() -> URL? {
        let fm = FileManager.default
        var dirs = ["/opt/homebrew/bin", "/usr/local/bin", NSHomeDirectory() + "/.local/bin"]
        dirs += (ProcessInfo.processInfo.environment["PATH"] ?? "").split(separator: ":").map(String.init)
        for dir in dirs {
            let path = (dir as NSString).appendingPathComponent("uv")
            if fm.isExecutableFile(atPath: path) { return URL(fileURLWithPath: path) }
        }
        return nil
    }

    // MARK: Ticker (1 s): timer, elapsed time; every 2 s: metrics and run list

    private func tick() {
        tickCount += 1
        if job == .run, let started = startedAt {
            let now = Date()
            elapsed = now.timeIntervalSince(started)
            if let deadline {
                remaining = max(0, deadline.timeIntervalSince(now))
                if now >= deadline, !stopRequested {
                    appendLines(["Timer elapsed."], isError: false)
                    stop()
                }
            }
        }
        if tickCount % 2 == 0 {
            refreshRuns()
            pollMetrics()
        }
    }

    private func pollMetrics() {
        let run = job == .run ? trackedRun : settings.targetRun
        guard !run.isEmpty else {
            if metrics != nil { metrics = nil }
            return
        }
        let url = repoURL.appendingPathComponent("runs/\(run)/metrics.jsonl")
        pollQueue.async { [weak self] in
            let snapshot = MetricsTailer.readLast(at: url)
            DispatchQueue.main.async {
                guard let self, self.metrics != snapshot else { return }
                self.metrics = snapshot
            }
        }
    }
}
