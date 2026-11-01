import Foundation
import Testing

@testable import AlifeCore

/// The words and numbers of the quit confirmation and panel (docs/APP.md §5.4), and the `LaunchPlan` flags that switch them on.
@Suite struct QuitWordingTests {
    @Test func bothModesConfirmARunAndWaitForIt() {
        for mode in LaunchMode.allCases {
            #expect(mode.plan.confirmsRunOnQuit, "\(mode)")
            #expect(mode.plan.quitWaitsForRun, "\(mode)")
        }
    }

    @Test func theQuestionNamesTheRunAndStatesTheLoss() {
        #expect(QuitWording.confirmTitle(run: "run-1") == "Stop 'run-1' and quit?")
        #expect(QuitWording.confirmDetail.contains("save point"))
        #expect(QuitWording.confirmDetail.contains("lost"))
    }

    @Test func theLossIsTheMinutesBetweenTwoSavesAtOneTimes() {
        #expect(QuitWording.lossMinutes(baseRate: 14.4) == 7)  // 6000 / 14.4 = 417 s
        #expect(QuitWording.lossMinutes(baseRate: 10) == 10)
        #expect(QuitWording.lossMinutes(baseRate: 1000) == 1)  // never "0 minutes"
        for bad in [0.0, -3, .nan, .infinity] { #expect(QuitWording.lossMinutes(baseRate: bad) == 7, "\(bad)") }
    }

    @Test func theBaseRateComesFromPlayJSONAndFallsBack() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let file = scratch.url.appendingPathComponent("play.json")
        #expect(QuitWording.baseRate(playFile: file) == 14.4)  // no file
        try """
            {"play_version": 1, "base_rate": 20.0, "cadence": {"metrics_every": 250, "heavy_metrics_every": 250,
             "checkpoint_every": 6000, "keep_checkpoints": 3}, "tick": 25, "weather": null, "planet": null}
            """.write(to: file, atomically: true, encoding: .utf8)
        #expect(QuitWording.baseRate(playFile: file) == 20)
        #expect(QuitWording.lossMinutes(baseRate: QuitWording.baseRate(playFile: file)) == 5)
        try "not json".write(to: file, atomically: true, encoding: .utf8)
        #expect(QuitWording.baseRate(playFile: file) == 14.4)
    }

    @Test func thePanelNamesWhatIsBeingSaved() {
        func title(_ worlds: [String], _ run: String? = nil) -> String {
            QuitPanelModel(worlds: worlds, run: run, elapsedSeconds: 0, lossMinutes: nil).title
        }
        #expect(title([]) == "Saving...")
        #expect(title(["storm"]) == "Saving 'storm'...")
        #expect(title([], "r1") == "Saving run 'r1'...")
        #expect(title(["a", "b"]) == "Saving 'a' and 'b'...")
        #expect(title(["a", "b"], "r1") == "Saving 'a', 'b' and run 'r1'...")
    }

    @Test func quitNowIsOfferedAfterTenSecondsWithTheCostInWords() {
        func model(_ seconds: Int, worlds: [String] = ["storm"], run: String? = nil, minutes: Int? = 7) -> QuitPanelModel {
            QuitPanelModel(worlds: worlds, run: run, elapsedSeconds: seconds, lossMinutes: minutes)
        }
        #expect(!model(9).offersQuitNow && model(9).warning == nil)
        #expect(model(10).offersQuitNow)
        let warning = model(10).warning ?? ""
        #expect(warning.contains("about 7 minutes of play at 1x"))
        #expect(warning.contains("powers"))
        #expect((model(10, minutes: 1).warning ?? "").contains("1 minute of play"))
        #expect(!(model(10, minutes: 1).warning ?? "").contains("1 minutes"))
        let run = model(12, worlds: [], run: "r1", minutes: nil).warning ?? ""
        #expect(run.contains("steps since its last save point") && !run.contains("minute"))
        #expect(model(10, worlds: [], run: nil, minutes: nil).warning == nil)
        #expect(model(-5).elapsedSeconds == 0)
    }
}
