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
        #expect(QuitWording.confirmDetail.contains("after 120 s"))
        #expect(QuitWording.confirmDetail.contains("lost"))
    }

    @Test func aRunWaitsLongerThanAnythingElse() {
        #expect(QuitWording.waitSeconds(runGoing: true) == 120)
        #expect(QuitWording.waitSeconds(runGoing: false) == 40)
        #expect(QuitWording.runWaitSeconds > QuitWording.otherWaitSeconds)
    }

    // MARK: Why the app is asked to quit

    @Test func theEndOfTheSessionIsToldFromAQuitByTheUser() {
        let fourCC = QuitReason.fourCC
        #expect(fourCC("quit") == 0x71756974)
        for code in ["rlgo", "logo", "rrst", "rsdn", "rest", "shut"] {
            #expect(QuitReason(code: fourCC(code)) == .sessionEnd, "\(code)")
            #expect(!QuitReason(code: fourCC(code)).asksQuestion, "\(code): nobody is there to answer")
        }
        #expect(QuitReason(code: fourCC("quia")) == .user)  // kAEQuitAll
        #expect(QuitReason(code: nil) == .user)  // a menu Quit sends no reason
        #expect(QuitReason(code: 0) == .user)
        #expect(QuitReason(code: fourCC("what")) == .user)  // unknown: ask, as for a user
        #expect(QuitReason.user.asksQuestion)
    }

    // MARK: The minutes a played world can lose

    @Test func theLossIsTheMinutesBetweenTwoSavesAtOneTimes() {
        #expect(QuitWording.lossMinutes(baseRate: 14.4) == 7)  // 6000 / 14.4 = 417 s
        #expect(QuitWording.lossMinutes(baseRate: 10) == 10)
        #expect(QuitWording.lossMinutes(baseRate: 1000) == 1)  // never "0 minutes"
        for bad in [0.0, -3, .nan, .infinity] { #expect(QuitWording.lossMinutes(baseRate: bad) == 7, "\(bad)") }
    }

    private static let playJSON = """
        {"play_version": 1, "base_rate": 20.0, "cadence": {"metrics_every": 250, "heavy_metrics_every": 250,
         "checkpoint_every": 6000, "keep_checkpoints": 3}, "tick": 25, "weather": null, "planet": null}
        """

    @Test func theBaseRateComesFromPlayJSONAndFallsBack() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let file = scratch.url.appendingPathComponent("play.json")
        #expect(QuitWording.baseRate(playFile: file) == 14.4)  // no file
        try Self.playJSON.write(to: file, atomically: true, encoding: .utf8)
        #expect(QuitWording.baseRate(playFile: file) == 20)
        #expect(QuitWording.lossMinutes(baseRate: QuitWording.baseRate(playFile: file)) == 5)
        try "not json".write(to: file, atomically: true, encoding: .utf8)
        #expect(QuitWording.baseRate(playFile: file) == 14.4)
    }

    @Test func theWorldsPlayFilesAreReadOnceAndTheLargestLossIsTheOneShown() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        for (name, rate) in [("fast", "20.0"), ("slow", "10.0")] {
            let dir = scratch.url.appendingPathComponent("runs/\(name)")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try Self.playJSON.replacingOccurrences(of: "20.0", with: rate)
                .write(to: dir.appendingPathComponent("play.json"), atomically: true, encoding: .utf8)
        }
        var cache: [String: Int] = [:]
        #expect(QuitWording.lossMinutes(worlds: [], repo: scratch.url, cache: &cache) == nil)
        #expect(QuitWording.lossMinutes(worlds: ["fast", "slow"], repo: scratch.url, cache: &cache) == 10)
        #expect(cache == ["fast": 5, "slow": 10])
        // Read once: the files may go and the answer stays.
        try FileManager.default.removeItem(at: scratch.url.appendingPathComponent("runs/slow/play.json"))
        #expect(QuitWording.lossMinutes(worlds: ["slow"], repo: scratch.url, cache: &cache) == 10)
        // A world with no file, or no folder to look in, counts at the default rate.
        var fresh: [String: Int] = [:]
        #expect(QuitWording.lossMinutes(worlds: ["slow"], repo: scratch.url, cache: &fresh) == 7)
        #expect(QuitWording.lossMinutes(worlds: ["x"], repo: nil, cache: &fresh) == 7)
    }

    // MARK: The panel

    private func model(worlds: [String] = [], run: Bool = false, name: String? = nil, other: String? = nil, seconds: Int = 0,
                       limit: Int = 40, minutes: Int? = nil) -> QuitPanelModel {
        QuitPanelModel(worlds: worlds, hasRun: run, runName: name, other: other, elapsedSeconds: seconds, limitSeconds: limit,
                       lossMinutes: minutes)
    }

    @Test func thePanelNamesWhatIsBeingSaved() {
        #expect(model().title == "Saving...")
        #expect(model(worlds: ["storm"]).title == "Saving 'storm'...")
        #expect(model(run: true, name: "r1").title == "Saving run 'r1'...")
        #expect(model(worlds: ["a", "b"]).title == "Saving 'a' and 'b'...")
        #expect(model(worlds: ["a", "b"], run: true, name: "r1").title == "Saving 'a', 'b' and run 'r1'...")
        #expect(model(other: "the environment setup").title == "Saving the environment setup...")
        // No empty name in the words.
        #expect(model(worlds: [""]).title == "Saving a world...")
        #expect(model(run: true).title == "Saving the run...")
    }

    @Test func thePanelStatesTheLimitAndNeverLeavesTheCostOut() {
        #expect(model(run: true, name: "r", seconds: 3, limit: 120).detail.contains("Elapsed 3 s of at most 120 s"))
        let run = model(run: true, name: "r", limit: 120).cost
        #expect(run.contains("stopped anyway after 120 s") && run.contains("loses the steps since its last save point"))
        // A run with no known name still says what it costs.
        #expect(model(run: true, limit: 120).cost == run)
        let world = model(worlds: ["w"], limit: 120, minutes: 7).cost
        #expect(world.contains("after 40 s"), "a world keeps its own 40 s ladder whatever the run's limit is")
        #expect(world.contains("6000 steps") && world.contains("about 7 minutes of play at 1x") && world.contains("powers"))
        #expect(model(worlds: ["w"], minutes: 1).cost.contains("about 1 minute of play"))
        #expect(!model(worlds: ["w"], minutes: 1).cost.contains("1 minutes"))
        #expect(model(worlds: ["w"]).cost.contains("about 7 minutes"), "no read of play.json: the default rate")
        let setup = model(other: "the environment setup").cost
        #expect(setup.contains("environment setup is stopped anyway after 40 s") && setup.contains("again"))
        #expect(model(other: "the check").cost.contains("check is stopped anyway"))
        for everything in [model(worlds: ["w"]), model(run: true), model(other: "the check"), model(worlds: ["w"], run: true)] {
            #expect(!everything.cost.isEmpty)
        }
        #expect(model().cost.isEmpty)
    }

    @Test func quitNowIsOfferedAfterTenSeconds() {
        #expect(!model(worlds: ["w"], seconds: 9).offersQuitNow)
        #expect(model(worlds: ["w"], seconds: 10).offersQuitNow)
        #expect(model(seconds: -5).elapsedSeconds == 0)
        #expect(model(limit: 0).limitSeconds == 1)
    }
}
