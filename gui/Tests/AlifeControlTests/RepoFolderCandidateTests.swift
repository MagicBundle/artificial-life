import Foundation
import Testing

@testable import AlifeControl

/// Finding the repository folder with no stored choice (F3) and saying why Start is grey (F2), docs/APP.md §1.3.
@MainActor
@Suite struct RepoFolderCandidateTests {
    private func folder(_ name: String, in scratch: Scratch, engine: Bool, body: String = "exit 0") throws -> String {
        let dir = scratch.url.appendingPathComponent(name)
        let bin = dir.appendingPathComponent(".venv/bin")
        try FileManager.default.createDirectory(at: engine ? bin : dir, withIntermediateDirectories: true)
        if engine {
            let alife = bin.appendingPathComponent("alife")
            try "#!/bin/bash\n\(body)\n".write(to: alife, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: alife.path)
        }
        return dir.path
    }

    private func make(stored: String? = nil, candidates: [String]) -> RunController {
        RunController(storedPath: stored, savePath: { _ in }, candidates: candidates, isApp: false)
    }

    @Test func theUsualPlacesAreTriedAndNoneIsThePathOfAPerson() {
        #expect(RepoFolder.candidates == ["~/artificial-life", "~/alife"])
        #expect(!RepoFolder.fieldPrompt.contains("artificial-life"))
        #expect(RepoFolder.initialPath(stored: nil, candidates: []).isEmpty)
    }

    @Test func withNothingStoredTheFirstCandidateWithTheEngineIsUsedWithoutANote() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let noEngine = try folder("first", in: scratch, engine: false)
        let second = try folder("second", in: scratch, engine: true)
        let third = try folder("third", in: scratch, engine: true)
        let missing = scratch.url.appendingPathComponent("missing").path
        let controller = make(candidates: [missing, noEngine, second, third])
        #expect(controller.repoPath == second)
        #expect(controller.repoReady && controller.logLines.isEmpty)
    }

    @Test func aStoredFolderThatIsGoneStaysChosenEvenWhenACandidateHasTheEngine() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let usable = try folder("alife", in: scratch, engine: true)
        let gone = scratch.url.appendingPathComponent("gone").path
        let controller = make(stored: gone, candidates: [scratch.url.appendingPathComponent("none").path, usable])
        #expect(controller.repoPath == gone, "detection only fills an empty choice")
        #expect(controller.detectedRepo == usable && controller.repoProblem == RepoFolder.missingMessage(gone))
        #expect(controller.startBlockedReason == RepoFolder.missingMessage(gone))
        #expect(controller.logLines.isEmpty)
    }

    @Test func noButtonIsOfferedWhenNothingWasDetectedOrTheFolderExists() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let none = scratch.url.appendingPathComponent("none").path
        #expect(make(stored: scratch.url.appendingPathComponent("gone").path, candidates: [none]).detectedRepo == nil)
        let bare = try folder("bare", in: scratch, engine: false)
        let usable = try folder("alife", in: scratch, engine: true)
        #expect(make(stored: bare, candidates: [usable]).detectedRepo == nil, "an existing folder is not second-guessed")
        let empty = make(candidates: [none])
        #expect(empty.detectedRepo == nil)
        empty.useDetectedFolder()
        #expect(empty.repoPath.isEmpty)
    }

    @Test func withNothingFoundThePathIsEmptyAndTheMessageIsPlain() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let controller = make(candidates: [scratch.url.appendingPathComponent("none").path])
        #expect(controller.repoPath.isEmpty && !controller.repoReady && !controller.repoFolderExists)
        let message = try #require(controller.repoProblem)
        #expect(message.hasPrefix("No alife folder is chosen yet."))
        #expect(message.contains("~/artificial-life") && message.contains("~/alife"))
        #expect(controller.startBlockedReason == message)
        #expect(controller.commandLine.hasPrefix("cd <repository folder> &&"))
        // The empty path is not the current directory: nothing is listed from it and nothing starts.
        #expect(controller.repoURL.path != FileManager.default.currentDirectoryPath)
        controller.start()
        #expect(controller.job == nil)
        #expect(controller.logLines.map(\.text) == [message])
    }

    @Test func startSaysWhyItIsGreyInPlainWords() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let none = scratch.url.appendingPathComponent("none").path
        // 1. The folder is gone.
        let gone = scratch.url.appendingPathComponent("gone").path
        let missing = make(stored: gone, candidates: [none])
        #expect(missing.startBlockedReason == "Folder not found: \(gone). Choose the alife folder.")
        // 2. The folder exists, the engine does not.
        let bare = try folder("bare", in: scratch, engine: false)
        let noEngine = make(stored: bare, candidates: [none])
        let reason = try #require(noEngine.startBlockedReason)
        #expect(reason.contains(".venv/bin/alife is missing") && reason.contains("Set up engine") && !reason.contains("epository"))
        // 3. Ready, but the settings are wrong: the first thing wrong.
        let ready = try folder("ready", in: scratch, engine: true)
        let controller = make(stored: ready, candidates: [none])
        #expect(controller.canStart && controller.startBlockedReason == nil)
        controller.settings.name = ""
        #expect(!controller.canStart)
        #expect(controller.startBlockedReason == "Enter a run name.")
        controller.settings.name = "ok-name"
        #expect(controller.startBlockedReason == nil)
    }

    @Test func startSaysNothingWhileAJobRuns() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let ready = try folder("ready", in: scratch, engine: true, body: "exec sleep 30")
        let controller = make(stored: ready, candidates: [])
        controller.settings.name = "reason-job"
        controller.start()
        defer { controller.forceStop() }
        #expect(controller.job == .run)
        #expect(controller.startBlockedReason == nil, "Start is not shown while a job runs")
        controller.forceStop()
        #expect(await eventually(timeout: .seconds(10)) { controller.job == nil })
    }
}
