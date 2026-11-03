@testable import AlifeCore
import Foundation
import Testing

@testable import AlifeControl

/// Welcome (docs/APP.md §3.1, PR-G): the folder, the set-up job and its failures, the health check and the model that ties them to the card.
/// Real processes, with shell scripts for `uv` and `alife`; nothing is installed and no folder but a scratch one is touched.
@MainActor
@Suite("Welcome", .serialized)
struct WelcomeTests {
    /// A scratch alife folder (a checkout with or without an engine), a stand-in `uv` and a controller over them.
    @MainActor struct Rig {
        let scratch: Scratch
        let folder: URL
        let controller: RunController
        let model = WelcomeModel()
        var uv: URL?
        /// What the model asked the clipboard and the browser to do.
        let clipboard = Recorder<String>()
        let opened = Recorder<URL>()

        init(engine: Bool = false, checkout: Bool = true, uvBody: String? = "exit 0", alife: String = "exit 0", folderExists: Bool = true) throws {
            scratch = try Scratch()
            folder = scratch.url.appendingPathComponent("alife-folder")
            if folderExists { try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true) }
            if folderExists, checkout {
                try "[project]\nname = \"alife\"\n".write(to: folder.appendingPathComponent("pyproject.toml"), atomically: true, encoding: .utf8)
                try FileManager.default.createDirectory(at: folder.appendingPathComponent("src/alife"), withIntermediateDirectories: true)
            }
            if engine { try Self.installEngine(in: folder, scratch: scratch, body: alife) }
            controller = RunController(storedPath: folder.path, savePath: { _ in }, candidates: [], isApp: false)
            if let uvBody { uv = try scratch.script("uv", uvBody) }
            let found = uv
            controller.uvLocator = { found }
            model.copy = { [clipboard] in clipboard.items.append($0) }
            model.open = { [opened] in opened.items.append($0) }
        }

        static func installEngine(in folder: URL, scratch: Scratch, body: String) throws {
            let bin = folder.appendingPathComponent(".venv/bin")
            try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
            let alife = try scratch.script("alife-\(UUID().uuidString)", body)
            try FileManager.default.moveItem(at: alife, to: bin.appendingPathComponent("alife"))
        }

        func cleanup() { scratch.cleanup() }

        /// A `uv` that installs an engine answering `alife` with `alife`'s body, printing a few lines first.
        static func installingUV(alife: String = "exit 0") -> String {
            """
            echo "Resolved 41 packages in 12ms"
            echo "Installed 41 packages in 1.2s"
            mkdir -p .venv/bin
            printf '#!/bin/bash\\n\(alife)\\n' > .venv/bin/alife
            chmod +x .venv/bin/alife
            """
        }
    }

    final class Recorder<T> {
        var items: [T] = []
    }

    // MARK: The folder

    @Test func aCheckoutWithoutAnEngineIsFoundInAUsualPlaceAndIsOfferedForSetUp() throws {
        let rig = try Rig(engine: false)
        defer { rig.cleanup() }
        #expect(RepoFolder.isCheckout(rig.folder.path))
        #expect(RepoFolder.firstCheckout(["/nonexistent-x", rig.folder.path]) == rig.folder.path)
        // with no choice stored: an engine first, else a checkout, else nothing
        #expect(RepoFolder.initialPath(stored: nil, candidates: [rig.folder.path]) == rig.folder.path)
        #expect(RepoFolder.initialPath(stored: nil, candidates: ["/nonexistent-x"]).isEmpty)
        let plain = rig.scratch.url.appendingPathComponent("plain")
        try FileManager.default.createDirectory(at: plain, withIntermediateDirectories: true)
        #expect(!RepoFolder.isCheckout(plain.path), "a folder that is not alife is never chosen for the person")
        let withEngine = rig.scratch.url.appendingPathComponent("with-engine")
        try Rig.installEngine(in: withEngine, scratch: rig.scratch, body: "exit 0")
        #expect(RepoFolder.initialPath(stored: nil, candidates: [rig.folder.path, withEngine.path]) == withEngine.path)
        #expect(RepoFolder.initialPath(stored: "/stored", candidates: [rig.folder.path]) == "/stored")
    }

    @Test func aCheckoutIsAFolderThatNamesAlifeAndHasItsSource() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        func make(_ name: String, pyproject: String?, source: Bool) throws -> String {
            let dir = scratch.url.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            if let pyproject { try pyproject.write(to: dir.appendingPathComponent("pyproject.toml"), atomically: true, encoding: .utf8) }
            if source { try FileManager.default.createDirectory(at: dir.appendingPathComponent("src/alife"), withIntermediateDirectories: true) }
            return dir.path
        }
        let mine = "[project]\nname = \"alife\"\nversion = \"0.1\"\n"
        #expect(RepoFolder.isCheckout(try make("ok", pyproject: mine, source: true)))
        #expect(!RepoFolder.isCheckout(try make("no-source", pyproject: mine, source: false)), "a pyproject alone is not alife")
        #expect(!RepoFolder.isCheckout(try make("no-pyproject", pyproject: nil, source: true)))
        #expect(!RepoFolder.isCheckout(try make("other-name", pyproject: "[project]\nname = \"alife-tools\"\n", source: true)))
        #expect(!RepoFolder.isCheckout(try make("other-project", pyproject: "[project]\nname = \"shop\"\n[tool.x]\nnote = \"name = alife\"\n", source: true)))
    }

    @Test func aFolderThatIsNotAlifeIsNeverOfferedASetUp() throws {
        let rig = try Rig(checkout: false, uvBody: Rig.installingUV())
        defer { rig.cleanup() }
        rig.model.refresh(rig.controller)
        #expect(rig.model.step(rig.controller) == .notAlife(chosen: rig.folder.path))
        // the Settings button reads the same rule
        #expect(!RepoFolder.isCheckout(rig.controller.repoPath))
    }

    @Test func theAlifeWindowSaysAlifeFolderAndTheOldAppsKeepRepositoryFolder() throws {
        let rig = try Rig(folderExists: false)
        defer { rig.cleanup() }
        #expect(rig.controller.repoProblem == RepoFolder.missingMessage(rig.folder.path), "the old wording is the default")
        #expect(rig.controller.repoProblem?.contains("repository folder") == true)
        rig.controller.alifeWording = true
        #expect(rig.controller.repoProblem == "Folder not found: \(rig.folder.path). Choose the alife folder.")
        #expect(RepoFolder.missingMessage("", alifeWording: true).hasPrefix("No alife folder is chosen yet."))
        #expect(RepoFolder.missingMessage("").hasPrefix("No repository folder is chosen yet."))
    }

    @Test func theCommandTheCardShowsIsTheOneTheButtonRuns() {
        #expect(WelcomeGuide.setupCommand == (["uv"] + RunController.setupArguments).joined(separator: " "))
        #expect(RepoFolder.candidates == RepoFolderPlaces.usual)
    }

    // MARK: The set-up

    @Test func aSetUpThatWorksLeavesAnEngineAndNoTrouble() async throws {
        let rig = try Rig(uvBody: Rig.installingUV())
        defer { rig.cleanup() }
        #expect(!rig.controller.repoReady && rig.controller.setupTrouble == nil)
        rig.controller.setupEnvironment()
        #expect(rig.controller.job == .setup)
        #expect(await eventually { rig.controller.job == nil })
        #expect(rig.controller.repoReady && rig.controller.setupTrouble == nil)
        #expect(WelcomeModel.progressLines(rig.controller).contains { $0.contains("Installed 41 packages") })
    }

    @Test func aSetUpThatFailsKeepsTheLastLinesAndTryingAgainClearsThem() async throws {
        let rig = try Rig(uvBody: "echo 'Resolved 41 packages'; echo 'error: failed to download mlx' >&2; echo 'Caused by: network is down' >&2; exit 2")
        defer { rig.cleanup() }
        rig.controller.setupEnvironment()
        #expect(await eventually { rig.controller.job == nil && rig.controller.setupTrouble != nil })
        let trouble = try #require(rig.controller.setupTrouble)
        #expect(!rig.controller.repoReady)
        #expect(trouble.summary.contains("exit code 2"))
        #expect(trouble.lines.contains("error: failed to download mlx") && trouble.lines.contains("Caused by: network is down"))
        #expect(WelcomeGuide.step(rig.model.facts(rig.controller)) == .setupFailed(trouble))
        // Set up again starts the job and the old failure is not shown beside it
        rig.controller.setupEnvironment()
        #expect(rig.controller.setupTrouble == nil && rig.model.step(rig.controller) == .settingUp)
        #expect(await eventually { rig.controller.job == nil && rig.controller.setupTrouble != nil })
    }

    @Test func aChangeOfFolderForgetsWhatWentWrongInTheOldOne() async throws {
        let rig = try Rig(uvBody: "echo 'error: no' >&2; exit 1")
        defer { rig.cleanup() }
        rig.controller.setupEnvironment()
        #expect(await eventually { rig.controller.setupTrouble != nil })
        rig.controller.repoPath = rig.scratch.url.appendingPathComponent("other").path
        #expect(rig.controller.setupTrouble == nil)
        #expect(rig.model.step(rig.controller) == .chooseFolder(chosen: rig.controller.repoPath))
    }

    @Test func cancellingASetUpStopsUvAtACleanPointAndSaysNothingWasRemoved() async throws {
        let rig = try Rig(uvBody: "trap 'exit 130' INT; echo 'Downloading mlx'; while true; do sleep 0.1; done")
        defer { rig.cleanup() }
        rig.controller.cancelSetup()  // (nothing runs: nothing happens)
        #expect(rig.controller.job == nil && rig.controller.logLines.isEmpty)
        rig.controller.setupEnvironment()
        #expect(await eventually { rig.model.step(rig.controller) == .settingUp && WelcomeModel.progressLines(rig.controller).contains("Downloading mlx") })
        rig.controller.cancelSetup()
        #expect(await eventually { rig.controller.job == nil })
        let trouble = try #require(rig.controller.setupTrouble)
        #expect(trouble.summary.contains("cancelled") && trouble.summary.contains("Nothing was removed"))
    }

    @Test func aSecondCancelSendsNothingMore() async throws {
        let rig = try Rig(uvBody: "trap '' INT; echo started; while true; do sleep 0.1; done")
        defer { rig.cleanup() }
        rig.controller.setupEnvironment()
        #expect(await eventually { WelcomeModel.progressLines(rig.controller).contains("started") })
        rig.controller.cancelSetup()
        rig.controller.cancelSetup()
        #expect(rig.controller.logLines.filter { $0.text.hasPrefix("Set-up cancelled") }.count == 1)
        rig.controller.forceStop()
        #expect(await eventually { rig.controller.job == nil })
    }

    @Test func aSetUpThatCannotStartIsToldOnTheCard() throws {
        let rig = try Rig(uvBody: "exit 0")
        defer { rig.cleanup() }
        let notExecutable = rig.scratch.url.appendingPathComponent("uv-no-exec")
        try "#!/bin/bash\nexit 0\n".write(to: notExecutable, atomically: true, encoding: .utf8)
        rig.controller.uvLocator = { notExecutable }
        rig.controller.setupEnvironment()
        #expect(rig.controller.job == nil)
        let trouble = try #require(rig.controller.setupTrouble)
        #expect(trouble.summary.contains("uv could not be started") && trouble.lines.contains { $0.contains("Failed to launch") })
        #expect(rig.model.step(rig.controller) == .setupFailed(trouble))
    }

    @Test func aSetUpBelongsToItsFolderWhateverIsChosenWhileItRuns() async throws {
        let rig = try Rig(uvBody: "echo working; sleep 1; echo 'error: no' >&2; exit 1")
        defer { rig.cleanup() }
        let other = rig.scratch.url.appendingPathComponent("other")
        try FileManager.default.createDirectory(at: other, withIntermediateDirectories: true)
        try "[project]\nname = \"alife\"\n".write(to: other.appendingPathComponent("pyproject.toml"), atomically: true, encoding: .utf8)
        try FileManager.default.createDirectory(at: other.appendingPathComponent("src/alife"), withIntermediateDirectories: true)
        rig.controller.setupEnvironment()
        #expect(rig.controller.setupFolder == rig.folder.path)
        rig.controller.repoPath = other.path
        // the new folder is not "setting up", and no other job may start in it meanwhile
        rig.model.refresh(rig.controller)
        #expect(rig.model.step(rig.controller) == .setUpEngine(uvFound: true))
        #expect(rig.model.facts(rig.controller).otherJobRunning)
        #expect(await eventually { rig.controller.job == nil })
        #expect(rig.controller.setupTrouble == nil, "the old folder's failure is not the new folder's")
        #expect(rig.model.step(rig.controller) == .setUpEngine(uvFound: true))
        #expect(rig.controller.setupFolder == nil)
        // back in the first folder nothing is remembered of it either (it was dropped, not kept for later)
        rig.controller.repoPath = rig.folder.path
        #expect(rig.controller.setupTrouble == nil)
    }

    @Test func aRepairOfAnEngineThatDidNotStartShowsItsProgressAndIsCheckedAgainWhenItEnds() async throws {
        let rig = try Rig(engine: true, uvBody: "echo 'Installed 2 packages'; sleep 0.5; printf '#!/bin/bash\\nexit 0\\n' > .venv/bin/alife", alife: "echo broken >&2; exit 1")
        defer { rig.cleanup() }
        rig.model.refresh(rig.controller)
        #expect(await eventually { if case .failed = rig.model.health { true } else { false } })
        guard case .checkFailed = rig.model.step(rig.controller) else { Issue.record("no failure card"); return }
        rig.controller.setupEnvironment()
        rig.model.refresh(rig.controller)
        #expect(rig.model.step(rig.controller) == .settingUp, "the repair is told, not the old failure")
        #expect(await eventually { rig.controller.job == nil })
        rig.model.refresh(rig.controller)  // (the window does this when the job ends)
        #expect(await eventually { rig.model.health == .passed }, "the engine is asked again when the set-up ends")
        #expect(rig.model.step(rig.controller) == nil)
    }

    @Test func theLibraryStaysUnderAFailureCard() throws {
        // the Worlds page shows the list whenever the engine is there; a failed check does not hide it (WorldsPage.body)
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/AlifeControl/WorldsPage.swift"), encoding: .utf8)
        #expect(source.contains("if controller.repoReady, showsLibrary { librarySection }"))
    }

    @Test func withoutUvTheCardSaysSoBeforeAnythingIsClicked() throws {
        let rig = try Rig(uvBody: nil)
        defer { rig.cleanup() }
        rig.model.refresh(rig.controller)
        #expect(!rig.model.uvFound)
        #expect(rig.model.step(rig.controller) == .setUpEngine(uvFound: false))
        // the button's own path says it too, in the log
        rig.controller.setupEnvironment()
        #expect(rig.controller.job == nil && rig.controller.logLines.last?.text.contains("Could not find 'uv'") == true)
        // uv is installed meanwhile: Check again sees it
        let later = try rig.scratch.script("uv-later", "exit 0")
        rig.controller.uvLocator = { later }
        rig.model.checkAgain(rig.controller)
        #expect(rig.model.uvFound && rig.model.step(rig.controller) == .setUpEngine(uvFound: true))
    }

    // MARK: The check

    @Test func theEngineIsAskedOnceAndAFolderThatWorksShowsNothing() async throws {
        let rig = try Rig(engine: true)
        defer { rig.cleanup() }
        var asked = 0
        rig.model.check = { _, _ in asked += 1; return .passed }
        rig.model.refresh(rig.controller)
        #expect(rig.model.health == .checking && rig.model.step(rig.controller) == nil, "a launch with a working folder shows no card")
        #expect(await eventually { rig.model.health == .passed })
        rig.model.refresh(rig.controller)
        rig.model.refresh(rig.controller)
        #expect(asked == 1 && rig.model.step(rig.controller) == nil && !rig.model.announced)
    }

    @Test func aPersonShownWelcomeIsToldTheCheckAndThenWelcomeGoesAway() async throws {
        let rig = try Rig(uvBody: Rig.installingUV())
        defer { rig.cleanup() }
        var release: CheckedContinuation<Void, Never>?
        rig.model.check = { _, _ in
            await withCheckedContinuation { release = $0 }
            return .passed
        }
        rig.model.refresh(rig.controller)
        #expect(rig.model.announced && rig.model.step(rig.controller) == .setUpEngine(uvFound: true))
        rig.controller.setupEnvironment()
        #expect(await eventually { rig.controller.job == nil && rig.controller.repoReady })
        rig.model.refresh(rig.controller)  // (the window does this when the engine appears)
        #expect(rig.model.step(rig.controller) == .checking)
        #expect(await eventually { release != nil })
        release?.resume()
        #expect(await eventually { rig.model.health == .passed })
        #expect(rig.model.step(rig.controller) == nil)
    }

    @Test func anEngineThatDoesNotStartIsToldWithWhatItPrintedAndCheckAgainAsksAgain() async throws {
        let rig = try Rig(engine: true, alife: "echo 'ModuleNotFoundError: No module named mlx' >&2; exit 1")
        defer { rig.cleanup() }
        rig.model.refresh(rig.controller)
        #expect(await eventually { if case .failed = rig.model.health { true } else { false } })
        guard case .checkFailed(let trouble) = rig.model.step(rig.controller) else { Issue.record("no failure card"); return }
        #expect(trouble.summary.contains("exit code 1") && trouble.lines == ["ModuleNotFoundError: No module named mlx"])
        // mended outside the app; Check again looks again
        try FileManager.default.removeItem(at: rig.folder.appendingPathComponent(".venv/bin/alife"))
        try Rig.installEngine(in: rig.folder, scratch: rig.scratch, body: "exit 0")
        rig.model.checkAgain(rig.controller)
        #expect(await eventually { rig.model.health == .passed })
        #expect(rig.model.step(rig.controller) == nil)
    }

    @Test func aFolderThatLosesItsEngineOrChangesIsCheckedAgain() async throws {
        let rig = try Rig(engine: true)
        defer { rig.cleanup() }
        var paths: [String] = []
        rig.model.check = { executable, _ in paths.append(executable.path); return .passed }
        rig.model.refresh(rig.controller)
        #expect(await eventually { rig.model.health == .passed })
        try FileManager.default.removeItem(at: rig.folder.appendingPathComponent(".venv"))
        rig.controller.refreshRepo()
        rig.model.refresh(rig.controller)
        #expect(rig.model.health == .unchecked && rig.model.step(rig.controller) == .setUpEngine(uvFound: true))
        let other = rig.scratch.url.appendingPathComponent("other")
        try Rig.installEngine(in: other, scratch: rig.scratch, body: "exit 0")
        rig.controller.repoPath = other.path
        rig.model.refresh(rig.controller)
        #expect(await eventually { rig.model.health == .passed })
        #expect(paths == [rig.folder.appendingPathComponent(".venv/bin/alife").path, other.appendingPathComponent(".venv/bin/alife").path])
    }

    @Test func theRealCheckRunsAlifeHelpInTheFolderWithADeadline() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let ok = try scratch.script("ok", "[ \"$1\" = \"--help\" ] && [ \"$(pwd -P)\" = \"$(cd \"$ALIFE_T\" 2>/dev/null && pwd -P)\" ] || exit 7; exit 0")
        setenv("ALIFE_T", scratch.url.path, 1)
        defer { unsetenv("ALIFE_T") }
        #expect(await EngineHealth.check(executable: ok, directory: scratch.url) == .passed)
        let bad = try scratch.script("bad", "for i in 1 2 3 4 5 6 7 8 9 10; do echo \"line $i\"; done; echo 'Traceback' >&2; exit 3")
        guard case .failed(let trouble) = await EngineHealth.check(executable: bad, directory: scratch.url) else { Issue.record("passed"); return }
        #expect(trouble.summary.contains("exit code 3") && trouble.lines.last == "Traceback" && trouble.lines.count == 6)
        let slow = try scratch.script("slow", "exec sleep 30")
        let started = Date()
        guard case .failed(let late) = await EngineHealth.check(executable: slow, directory: scratch.url, deadline: 0.4) else { Issue.record("passed"); return }
        let took = Date().timeIntervalSince(started)
        #expect(late.summary.contains("did not answer") && took < 10, "took \(took) s")
        let missing = scratch.url.appendingPathComponent("nothing-here")
        guard case .failed(let none) = await EngineHealth.check(executable: missing, directory: scratch.url) else { Issue.record("passed"); return }
        #expect(none.summary.contains("could not be started"))
    }

    // MARK: The card's buttons

    @Test func theButtonsCopyAndOpenWhatTheyShow() throws {
        let rig = try Rig(uvBody: nil)
        defer { rig.cleanup() }
        rig.model.copyFolderSteps()
        rig.model.copyUVCommand()
        rig.model.copyReport(.init(summary: "It stopped.", lines: ["a", "b"]))
        rig.model.openUVPage()
        #expect(rig.clipboard.items == [WelcomeGuide.fetchSteps(), "brew install uv", "It stopped.\na\nb"])
        #expect(rig.opened.items == [URL(string: WelcomeGuide.uvPage)])
    }
}
