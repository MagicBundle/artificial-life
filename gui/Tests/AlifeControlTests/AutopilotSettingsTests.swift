import AlifeCore
import Foundation
import Testing

@testable import AlifeControl

/// Settings > Autopilot (docs/APP.md §3.8 row 14): the model's address and name, kept in the app's own preferences, the start of every new
/// world's form, and the test that asks the engine's `llm-check`. Nothing is contacted: the check is a stand-in.
@MainActor
@Suite("Settings > Autopilot", .serialized)
struct AutopilotSettingsTests {
    @MainActor struct Rig {
        let scratch: Scratch
        let suite = "alife-autopilot-\(UUID().uuidString)"
        let defaults: UserDefaults
        let controller: RunController
        let settings: AutopilotSettings
        var asked: Asked

        final class Asked { var calls: [(URL, [String], URL)] = []; var result = LLMCheckResult.parse(status: 0, reason: .exit, output: "OK the model answered"); var delay: Duration = .zero }

        init(engine: Bool = true) throws {
            scratch = try Scratch()
            defaults = try #require(UserDefaults(suiteName: suite))
            let root = scratch.url.appendingPathComponent("folder")
            if engine {
                try FileManager.default.createDirectory(at: root.appendingPathComponent(".venv/bin"), withIntermediateDirectories: true)
                let alife = try scratch.script("alife", "exit 0")
                try FileManager.default.moveItem(at: alife, to: root.appendingPathComponent(".venv/bin/alife"))
            }
            controller = RunController(storedPath: root.path, savePath: { _ in }, candidates: [], isApp: false)
            settings = AutopilotSettings(controller: controller, defaults: defaults)
            asked = Asked()
            let asked = asked
            settings.runCheck = { exe, args, dir in
                asked.calls.append((exe, args, dir))
                try? await Task.sleep(for: asked.delay)
                return asked.result
            }
        }

        func finish() {
            defaults.removePersistentDomain(forName: suite)
            scratch.cleanup()
        }
    }

    @Test func itStartsFromTheEnginesDefaultsAndKeepsWhatIsTyped() throws {
        let rig = try Rig()
        defer { rig.finish() }
        #expect(rig.settings.url == DirectorDefaults.url && rig.settings.model == DirectorDefaults.model && rig.settings.isDefault)
        rig.settings.url = "http://127.0.0.1:9/v1/chat/completions"
        rig.settings.model = "some-model"
        #expect(!rig.settings.isDefault)
        // (a new settings object over the same preferences reads them back)
        let again = AutopilotSettings(controller: rig.controller, defaults: rig.defaults)
        #expect(again.url == "http://127.0.0.1:9/v1/chat/completions" && again.model == "some-model")
        again.useDefaults()
        #expect(again.isDefault)
        #expect(AutopilotSettings(controller: rig.controller, defaults: rig.defaults).url == DirectorDefaults.url)
    }

    @Test func aNewWorldsFormStartsFromTheSavedModelOnlyWhenItIsUsable() throws {
        let rig = try Rig()
        defer { rig.finish() }
        rig.settings.url = "  http://127.0.0.1:9/x  "
        rig.settings.model = " m "
        var director = DirectorSettings()
        rig.settings.apply(to: &director)
        #expect(director.llmURL == "http://127.0.0.1:9/x" && director.llmModel == "m", "trimmed")
        rig.settings.url = "not an address"
        var other = DirectorSettings()
        rig.settings.apply(to: &other)
        #expect(other.llmURL == DirectorDefaults.url && other.llmModel == DirectorDefaults.model, "a half-typed address never reaches a command")
        rig.settings.url = "ftp://x/y"
        rig.settings.apply(to: &other)
        #expect(other.llmURL == DirectorDefaults.url)
        rig.settings.url = "http://127.0.0.1:9/x"
        rig.settings.model = "  "
        rig.settings.apply(to: &other)
        #expect(other.llmModel == DirectorDefaults.model)
    }

    @Test func theControllerAppliesItToANewWorldButNotToABranch() throws {
        let rig = try Rig()
        defer { rig.finish() }
        rig.settings.url = "http://127.0.0.1:9/x"
        rig.settings.model = "m"
        let settings = rig.settings
        rig.controller.autopilotDefaults = { director in settings.apply(to: &director) }
        rig.controller.settings.mode = .new
        rig.controller.loadDirector()
        #expect(rig.controller.settings.director.llmURL == "http://127.0.0.1:9/x" && rig.controller.settings.director.llmModel == "m")
        #expect(rig.controller.settings.directorContext.baseline.llmURL == DirectorDefaults.url, "the baseline stays the engine's: the command carries the difference")
        rig.controller.settings.director.llmOn = true
        rig.controller.settings.director.mode = .live
        let command = rig.controller.settings.directorContext.baseline
        #expect(rig.controller.settings.director.arguments(context: rig.controller.settings.directorContext).contains("--llm-url=http://127.0.0.1:9/x"))
        _ = command
        rig.controller.settings.mode = .fork
        rig.controller.settings.sourceRun = ""
        rig.controller.loadDirector()
        #expect(rig.controller.settings.director.llmURL == DirectorDefaults.url, "a branch keeps its source's, never the Settings'")
    }

    @Test func theOldAppsHaveNoSuchDefault() throws {
        let rig = try Rig()
        defer { rig.finish() }
        rig.controller.loadDirector()
        #expect(rig.controller.autopilotDefaults == nil && rig.controller.settings.director.llmURL == DirectorDefaults.url)
    }

    @Test func theTestAsksTheEnginesCheckWithTheAddressAndTheModelAndShowsTheAnswer() async throws {
        let rig = try Rig()
        defer { rig.finish() }
        rig.settings.url = "http://127.0.0.1:9/x"
        rig.settings.model = "m"
        #expect(rig.settings.testReason == nil)
        rig.settings.test()
        #expect(rig.settings.check == .running && rig.settings.testReason == "Testing…")
        #expect(await eventually { rig.settings.check != .running })
        #expect(rig.settings.check == .result(LLMCheckResult.parse(status: 0, reason: .exit, output: "OK the model answered")))
        let call = try #require(rig.asked.calls.first)
        #expect(call.1 == ["llm-check", "--llm-url=http://127.0.0.1:9/x", "--llm-model=m"] && call.0.lastPathComponent == "alife")
        #expect(call.2.path == rig.controller.repoURL.path)
    }

    @Test func aFailureIsShownAsAFailure() async throws {
        let rig = try Rig()
        defer { rig.finish() }
        rig.asked.result = LLMCheckResult.parse(status: 1, reason: .exit, output: "FAIL connection refused")
        rig.settings.test()
        #expect(await eventually { rig.settings.check != .running })
        #expect(rig.settings.check == .result(LLMCheckResult.parse(status: 1, reason: .exit, output: "FAIL connection refused")))
        if case .result(let result) = rig.settings.check { #expect(!result.ok && result.text.contains("connection refused")) }
    }

    @Test func editingWhileItRunsDropsTheAnswerForOtherValues() async throws {
        let rig = try Rig()
        defer { rig.finish() }
        rig.asked.delay = .milliseconds(200)
        rig.settings.test()
        rig.settings.model = "another"
        #expect(await eventually { rig.asked.calls.count == 1 })
        #expect(await eventually(timeout: .seconds(5)) { rig.settings.check == .idle || rig.settings.check == .running })
        try await Task.sleep(for: .milliseconds(400))
        #expect(rig.settings.check == .idle, "an answer for the old values would mislead")
    }

    @Test func editingForgetsAnOldAnswer() async throws {
        let rig = try Rig()
        defer { rig.finish() }
        rig.settings.test()
        #expect(await eventually { rig.settings.check != .running })
        rig.settings.url = "http://127.0.0.1:9/y"
        #expect(rig.settings.check == .idle)
    }

    @Test func theTestHasAReasonWhenItCannotRun() throws {
        let noEngine = try Rig(engine: false)
        defer { noEngine.finish() }
        #expect(noEngine.settings.testReason == "Set up the engine first.")
        noEngine.settings.test()
        #expect(noEngine.asked.calls.isEmpty)
        let rig = try Rig()
        defer { rig.finish() }
        rig.settings.url = "nothing"
        #expect(rig.settings.testReason == "Type an http or https address and a model name.")
        rig.settings.test()
        #expect(rig.asked.calls.isEmpty)
        rig.settings.url = "http://x/y"
        rig.settings.model = ""
        #expect(rig.settings.testReason != nil)
    }

    @Test func theDomainsOfTheOtherAppsAreNeverNamed() {
        #expect(AppShell.autopilotURLKey.hasPrefix("shell.") && AppShell.autopilotModelKey.hasPrefix("shell."))
    }
}
