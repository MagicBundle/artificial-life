import AlifeCore
import AppKit
import Foundation

/// The health check of an engine: `alife --help` run in the folder, which must answer with exit code 0. Off the main thread, with a deadline.
enum EngineHealth {
    /// A minute: the first start after an install is the slow one (the packages are compiled and cached on first import).
    static let deadline: TimeInterval = 60

    /// Runs `executable --help` in `directory`. `deadline` ends it (SIGTERM, then SIGKILL two seconds later).
    static func check(executable: URL, directory: URL, deadline: TimeInterval = EngineHealth.deadline) async -> WelcomeGuide.Health {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(returning: run(executable: executable, directory: directory, deadline: deadline))
            }
        }
    }

    private final class Box: @unchecked Sendable {
        private let lock = NSLock()
        private var data = Data()
        private var timedOut = false
        func append(_ more: Data) { lock.lock(); data.append(more); lock.unlock() }
        func markTimedOut() { lock.lock(); timedOut = true; lock.unlock() }
        var snapshot: (Data, Bool) { lock.lock(); defer { lock.unlock() }; return (data, timedOut) }
    }

    private static func run(executable: URL, directory: URL, deadline: TimeInterval) -> WelcomeGuide.Health {
        let process = Process()
        process.executableURL = executable
        process.arguments = ["--help"]
        process.currentDirectoryURL = directory
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        process.standardInput = FileHandle.nullDevice
        let box = Box()
        let done = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in done.signal() }
        do {
            try process.run()
        } catch {
            return .failed(WelcomeGuide.healthFailure(status: nil, timedOut: false, lines: [error.localizedDescription]))
        }
        // (the pipe is drained while the process runs, so a chatty one cannot block on a full pipe)
        let reader = DispatchQueue(label: "alife.health.read")
        let drained = DispatchSemaphore(value: 0)
        reader.async {
            while true {
                let chunk = pipe.fileHandleForReading.availableData
                if chunk.isEmpty { break }
                box.append(chunk)
            }
            drained.signal()
        }
        if done.wait(timeout: .now() + deadline) == .timedOut {
            box.markTimedOut()
            process.terminate()
            if done.wait(timeout: .now() + 2) == .timedOut {
                kill(process.processIdentifier, SIGKILL)
                _ = done.wait(timeout: .now() + 2)
            }
        }
        _ = drained.wait(timeout: .now() + 2)
        let (data, timedOut) = box.snapshot
        let lines = String(decoding: data, as: UTF8.self).split(whereSeparator: \.isNewline).map(String.init)
        if timedOut { return .failed(WelcomeGuide.healthFailure(status: nil, timedOut: true, lines: lines)) }
        if process.terminationReason == .exit, process.terminationStatus == 0 { return .passed }
        return .failed(WelcomeGuide.healthFailure(
            status: process.terminationReason == .exit ? process.terminationStatus : nil, timedOut: false, lines: lines))
    }
}

/// What Welcome knows that the controller does not: whether `uv` is there, whether the engine in the chosen folder answers `alife --help`
/// (once per folder and launch, off the main thread), and whether Welcome has been shown in this session. The page reads `step`; the actions
/// are the card's buttons. Everything with a side effect (the check, the clipboard, the browser) is a closure the tests replace.
@MainActor
final class WelcomeModel: ObservableObject {
    @Published private(set) var health = WelcomeGuide.Health.unchecked
    @Published private(set) var uvFound = true
    @Published private(set) var announced = false
    /// `alife --help` for the engine at `executable` in `directory`.
    var check: (URL, URL) async -> WelcomeGuide.Health = { executable, directory in
        await EngineHealth.check(executable: executable, directory: directory)
    }
    var copy: (String) -> Void = { text in
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
    var open: (URL) -> Void = { NSWorkspace.shared.open($0) }
    private var checkedPath: String?
    private var running: Task<Void, Never>?
    private var setupWasRunning = false

    /// A set-up runs in the chosen folder (one that runs in another folder is not this folder's).
    private func setupRuns(_ controller: RunController) -> Bool { controller.job == .setup && controller.setupFolder == controller.repoPath }

    func facts(_ controller: RunController) -> WelcomeGuide.Facts {
        WelcomeGuide.Facts(
            folderPath: controller.repoPath.trimmingCharacters(in: .whitespacesAndNewlines),
            folderExists: controller.repoFolderExists,
            isCheckout: controller.repoFolderExists && RepoFolder.isCheckout(controller.repoPath),
            engineReady: controller.repoReady,
            uvFound: uvFound,
            setupRunning: setupRuns(controller),
            otherJobRunning: controller.job != nil && !setupRuns(controller),
            setupFailure: controller.setupTrouble,
            health: health,
            announced: announced)
    }

    func step(_ controller: RunController) -> WelcomeGuide.Step? { WelcomeGuide.step(facts(controller)) }

    /// Looks again at everything Welcome depends on: `uv`, and the engine of the chosen folder (checked once per folder; a folder whose engine
    /// went away is forgotten). Call when the page appears, the folder or the engine changes, and when the app is brought back.
    func refresh(_ controller: RunController) {
        let found = controller.uvLocator() != nil
        if found != uvFound { uvFound = found }
        let path = controller.repoURL.path
        // A set-up that ended (it may have mended an engine that did not start): the engine is asked again.
        let setupRunning = controller.job == .setup
        if setupWasRunning, !setupRunning { checkedPath = nil }
        setupWasRunning = setupRunning
        if !controller.repoReady {
            running?.cancel()
            running = nil
            checkedPath = nil
            if health != .unchecked { health = .unchecked }
        } else if checkedPath != path, !setupRunning {
            startCheck(controller, path: path)
        }
        if !announced, step(controller) != nil { announced = true }
    }

    /// The card's Check again: the engine is asked again, and `uv` and the folder are looked at again.
    func checkAgain(_ controller: RunController) {
        controller.refreshRepo()
        checkedPath = nil
        refresh(controller)
    }

    private func startCheck(_ controller: RunController, path: String) {
        running?.cancel()
        checkedPath = path
        health = .checking
        let executable = controller.alifeURL
        let directory = controller.repoURL
        running = Task { [weak self] in
            guard let self else { return }
            let result = await self.check(executable, directory)
            guard !Task.isCancelled, self.checkedPath == path else { return }
            self.health = result
            self.running = nil
        }
    }

    func copyFolderSteps() { copy(WelcomeGuide.fetchSteps()) }
    func copyUVCommand() { copy(WelcomeGuide.uvInstallCommand) }
    func copyReport(_ trouble: WelcomeGuide.Trouble) { copy(WelcomeGuide.report(for: trouble)) }
    func openUVPage() { if let url = URL(string: WelcomeGuide.uvPage) { open(url) } }

    /// The lines the card shows while uv works: the end of the job's log.
    static func progressLines(_ controller: RunController, count: Int = 6) -> [String] {
        WelcomeGuide.tail(controller.logLines.map(\.text), count: count)
    }
}
