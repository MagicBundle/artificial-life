import Foundation
import Testing

@testable import AlifeCore

/// Welcome (docs/APP.md §3.1, PR-G): which card a person without a working engine sees, the checklist, and the words.
@Suite struct WelcomeGuideTests {
    private func facts(folder: Bool = true, engine: Bool = true, uv: Bool = true, running: Bool = false, failure: WelcomeGuide.Trouble? = nil,
                       health: WelcomeGuide.Health = .passed, announced: Bool = false, path: String = "/Users/x/alife") -> WelcomeGuide.Facts {
        WelcomeGuide.Facts(folderPath: path, folderExists: folder, engineReady: engine, uvFound: uv, setupRunning: running, setupFailure: failure,
                           health: health, announced: announced)
    }

    @Test func aWorkingEngineNeedsNoCardAndEveryOtherStateHasOne() {
        #expect(WelcomeGuide.step(facts()) == nil)
        #expect(WelcomeGuide.step(facts(folder: false, engine: false, path: "")) == .chooseFolder(chosen: nil))
        #expect(WelcomeGuide.step(facts(folder: false, engine: false)) == .chooseFolder(chosen: "/Users/x/alife"))
        #expect(WelcomeGuide.step(facts(engine: false)) == .setUpEngine(uvFound: true))
        #expect(WelcomeGuide.step(facts(engine: false, uv: false)) == .setUpEngine(uvFound: false))
        #expect(WelcomeGuide.step(facts(engine: false, running: true)) == .settingUp)
        let trouble = WelcomeGuide.Trouble(summary: "s", lines: ["a"])
        #expect(WelcomeGuide.step(facts(engine: false, failure: trouble)) == .setupFailed(trouble))
        // a set-up that runs again hides the old failure
        #expect(WelcomeGuide.step(facts(engine: false, running: true, failure: trouble)) == .settingUp)
        #expect(WelcomeGuide.step(facts(health: .failed(trouble))) == .checkFailed(trouble))
    }

    @Test func theCheckIsToldOnlyToSomeoneWhoWasAlreadyShownWelcome() {
        for health in [WelcomeGuide.Health.unchecked, .checking] {
            #expect(WelcomeGuide.step(facts(health: health)) == nil, "a launch with a working folder shows nothing")
            #expect(WelcomeGuide.step(facts(health: health, announced: true)) == .checking)
        }
        #expect(WelcomeGuide.step(facts(health: .passed, announced: true)) == nil)
    }

    @Test func theFolderComesBeforeTheEngineAndTheEngineBeforeTheCheck() {
        func marks(_ f: WelcomeGuide.Facts) -> [WelcomeGuide.Mark] { WelcomeGuide.checklist(f).map(\.mark) }
        #expect(marks(facts(folder: false, engine: false)) == [.current, .waiting, .waiting])
        #expect(marks(facts(engine: false)) == [.done, .current, .waiting])
        #expect(marks(facts(health: .checking)) == [.done, .done, .current])
        #expect(marks(facts(health: .failed(.init(summary: "x"))))  == [.done, .done, .current])
        #expect(marks(facts()) == [.done, .done, .done])
        #expect(WelcomeGuide.checklist(facts(engine: false, running: true))[1].detail == "Installing now")
        #expect(WelcomeGuide.checklist(facts())[0].detail == "/Users/x/alife")
    }

    @Test func everyCardHasATitleAndASentenceAndNoneUsesARetiredWord() {
        let trouble = WelcomeGuide.Trouble(summary: "It stopped.", lines: [])
        let steps: [WelcomeGuide.Step] = [.chooseFolder(chosen: nil), .chooseFolder(chosen: "/x"), .setUpEngine(uvFound: true), .setUpEngine(uvFound: false),
                                          .settingUp, .setupFailed(trouble), .checking, .checkFailed(trouble)]
        var texts = [WelcomeGuide.fetchExplanation, WelcomeGuide.fetchSteps()]
        for step in steps {
            #expect(!WelcomeGuide.title(step).isEmpty && !WelcomeGuide.explanation(step).isEmpty)
            texts += [WelcomeGuide.title(step), WelcomeGuide.explanation(step)]
        }
        for text in texts {
            for word in ["Repository", "repository", "Checkpoint", "Fork", "Resume", "the Lab", "Storm Worlds"] { #expect(!text.contains(word), "'\(word)' in \"\(text)\"") }
        }
        #expect(WelcomeGuide.explanation(.chooseFolder(chosen: nil)).contains("~/artificial-life"))
    }

    @Test func theInstructionsAreCopyableAndNameTheCommandTheButtonRuns() {
        let steps = WelcomeGuide.fetchSteps().split(separator: "\n").map(String.init)
        #expect(steps == ["brew install git uv", "git clone <the address you were given> ~/artificial-life", "cd ~/artificial-life",
                          "uv sync --inexact --extra viewer --extra video"])
        #expect(WelcomeGuide.fetchSteps(into: "~/alife").contains("git clone <the address you were given> ~/alife"))
        #expect(WelcomeGuide.setupCommand == "uv " + ["sync", "--inexact", "--extra", "viewer", "--extra", "video"].joined(separator: " "))
        #expect(URL(string: WelcomeGuide.uvPage)?.host == "docs.astral.sh")
    }

    @Test func aFailureKeepsTheLastLinesAndSaysWhatToDo() {
        let many = (1...40).map { "line \($0)" } + ["", "  ", "last"]
        let failure = WelcomeGuide.setupFailure(status: 1, signalled: false, lines: many)
        #expect(failure.lines.count == 8 && failure.lines.last == "last" && failure.lines.first == "line 34")
        #expect(failure.summary.contains("exit code 1"))
        #expect(WelcomeGuide.tail(["$ /long/path/uv sync", "a"]) == ["a"], "the command line of the log is not progress")
        #expect(WelcomeGuide.setupFailure(status: 130, signalled: false, lines: []).summary.contains("cancelled"))
        #expect(WelcomeGuide.setupFailure(status: 15, signalled: true, lines: []).summary.contains("Set up again"))
        #expect(WelcomeGuide.report(for: failure).split(separator: "\n").count == 9)
        let health = WelcomeGuide.healthFailure(status: 3, timedOut: false, lines: ["Traceback", "ImportError: no module"])
        #expect(health.summary.contains("exit code 3") && health.lines == ["Traceback", "ImportError: no module"])
        #expect(WelcomeGuide.healthFailure(status: nil, timedOut: true, lines: []).summary.contains("did not answer"))
        #expect(WelcomeGuide.healthFailure(status: nil, timedOut: false, lines: ["boom"]).summary.contains("could not be started"))
    }

    @Test func theUsualPlacesAreTheOnesTheAppLooksAt() {
        #expect(RepoFolderPlaces.usual == ["~/artificial-life", "~/alife"])
    }
}
