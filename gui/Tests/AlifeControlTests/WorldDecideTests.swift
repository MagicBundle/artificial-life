@testable import AlifeCore
import AppKit
import CoreGraphics
import Foundation
import SwiftUI
import Testing
import simd

@testable import AlifeControl

/// A hand-cranked clock for the model.
fileprivate final class Clock {
    var t: Double = 1000
}

/// What the dock was asked for.
fileprivate final class Dock {
    var attention = 0
    var badge: String?
}

fileprivate struct DecideRig {
    let model: WorldModel
    let play: PlayController
    let commands: URL
    let clock: Clock
    let dock: Dock
    let defaults: UserDefaults
    var sent: [String] { ((try? String(contentsOf: commands, encoding: .utf8)) ?? "").split(separator: "\n").map(String.init) }
}

/// A live world playing the G3b fixture (L = 48, the powers Food and Strike, two alerts on the stack, a chapter report, the
/// player's Food running to 18,000) that records the control lines it receives. The model is made *before* the world starts, so
/// it hears the whole stream.
@MainActor
fileprivate func makeDecideRig(_ scratch: Scratch, prefs: AutopausePrefs? = nil, windowKey: Bool = true, origin: Int? = nil,
                               strikeReady: Bool = false) async throws -> DecideRig {
    let commands = scratch.url.appendingPathComponent("commands.txt")
    let script = try scratch.script("alife-live", "cat \"$1\"\nwhile IFS= read -r line; do echo \"$line\" >> \"$2\"; done")
    let play = PlayController()
    if let origin { play.readOrigin = { _, _ in origin } }
    let suite = "alife-decide-\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suite))
    if let prefs { defaults.set(prefs.json, forKey: WorldModel.autopauseKey) }
    let model = WorldModel(play: play, defaults: defaults)
    let clock = Clock()
    let dock = Dock()
    model.now = { clock.t }
    model.windowIsKey = { windowKey }
    model.requestAttention = { dock.attention += 1 }
    model.setBadge = { dock.badge = $0 }
    play.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url,
               arguments: [Repo.playFixtureURL("3d_g3b").path, commands.path])
    #expect(await eventually { play.store.header != nil && play.store.plans.records.count == 7 && play.store.clock?.reason == .alert })
    if strikeReady {
        // (the fixture's Strike is cooling down until step 15,000)
        var record = try #require(play.store.environment)
        record.environment.powers?["strike"] = .init(ready: nil, busy: nil)
        play.store.apply(.envs(record))
    }
    model.setFreeRect(CGRect(x: 0, y: 0, width: 900, height: 700))
    model.goHome()
    model.edit {
        $0.distance = 30
        $0.target = SIMD3(24, 24, 0)
    }
    return DecideRig(model: model, play: play, commands: commands, clock: clock, dock: dock, defaults: defaults)
}

fileprivate func chapterClock(chapter: UInt32, step: UInt64, rate: Float = 0, saving: Bool = false, flags: UInt16 = 1,
                              reason: PlayClock.Reason = .chapter) -> PlayClock {
    PlayClock(head: RecordHead(step: step, seq: 0, version: 1, flags: 0), rate: rate, achieved: rate == 0 ? 0 : 14.4, stepMs: 15, frameMs: 3,
              lateSteps: 0, saveStallSteps: 0, reason: reason, saving: saving, flags: flags, fe: 1, chapter: chapter, inChapter: 6000,
              chapterLen: 6000)
}

/// The decide half of the World window against a played fixture: the lines that reach the world, and what the window shows of
/// what it said (DESIGN §34.7). No renderer, no real window.
@MainActor
@Suite struct WorldDecideTests {
    private func sent(_ rig: DecideRig, _ line: String) async -> Bool {
        await eventually { rig.sent.contains(line) }
    }

    private func plan(_ id: Int, _ status: PlanBody.Status, cost: Int = 1) -> PlayPlan {
        let body = PlanBody(
            id: id, kind: .power, status: status, power: "food", action: "regen_pulse", raw: [:], params: [:], clamped: [], duration: 6000,
            stepApplied: nil, ends: nil, cost: cost, influence: nil, ready: nil, holds: nil, text: "Food x1.3 for 6,000 steps.", caution: [],
            precedent: nil, since: nil, refused: nil)
        return PlayPlan(head: RecordHead(step: 12_000, seq: 0, version: 1, flags: status == .preview ? 1 : 0), plan: body)
    }

    /// The alerts of the fixture's live batch, as a later delivery would announce them (the stand-in sends the whole stream in
    /// one piece, and the first delivery of a world resets what the window holds before it: a live batch comes after).
    private func liveAlerts(_ rig: DecideRig, ids: [UInt64] = [12_000_600, 12_000_601, 12_000_602, 11_500_603, 11_750_605]) throws -> [AlertChange] {
        try ids.map { AlertChange(kind: .raised, alert: try #require(rig.play.store.alerts.alert(id: $0))) }
    }

    /// Feeds a `PLAN` the way the controller does.
    private func hear(_ rig: DecideRig, _ plan: PlayPlan) {
        rig.play.store.apply(plan: plan)
        rig.model.handle(plans: [plan])
    }

    // MARK: What the stream's alerts do to the window

    @Test func aReportLineIsTickerNewsOnlyIfItIsDatedWithinASnapshotOfTheChaptersEnd() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.resetInteraction()
        func report(_ ticks: [(Int, String)]) -> PlayReport {
            let body = ReportBody(
                chapter: 7, nAbs: 7, a: 24_000, b: 30_000, mode: 0, final: true, withheld: [], headline: .init(id: "H", text: "h", refs: []),
                lead: nil, body: [], interventions: [], ticker: ticks.map { .init(step: $0.0, text: $0.1) },
                since: .init(influence: nil, alerts: [:], ending: []), entities: [:],
                versions: .init(report: 2, template: 2, checker: 1, lexicon: 3), factsDigest: "d", latencyS: nil, fallback: nil)
            return PlayReport(head: RecordHead(step: 30_000, seq: 0, version: 1, flags: 2), report: body)
        }
        // the last line of the report is 750 steps old: it was told when it happened, and the tab keeps it
        rig.model.handle(reports: [report([(29_000, "Step 29,000: older"), (29_250, "Step 29,250: Merrow formed")])])
        #expect(rig.model.tickerLine == nil, "a line of 750 steps ago is not news")
        rig.model.handle(reports: [report([(29_250, "Step 29,250: older"), (29_900, "Step 29,900: Merrow formed")])])
        #expect(rig.model.tickerLine?.text == "Step 29,900: Merrow formed")
    }

    @Test func theLiveAlertsToastOnceAndTheLogRowIsTheTickerAndTheBacklogIsHistory() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.handle(alerts: try liveAlerts(rig), backlog: true)
        #expect(rig.model.toasts.isEmpty && rig.model.tickerLine == nil, "a re-opened world's history is not news")
        // The storm naming (stack surface, importance 2) toasts; the chapter's own alert (the card is its notice), the surge
        // (importance 1), the log rows and the backlog's fall do not.
        rig.model.handle(alerts: try liveAlerts(rig), backlog: false)
        #expect(rig.model.toasts.map(\.text) == ["Storm Merrow: Storm Merrow reached named strength."], "\(rig.model.toasts.map(\.text))")
        #expect(rig.model.tickerLine?.text == "Storm Merrow reached 0.7 times the naming threshold.", "the last log row of the batch")
        let layout = rig.model.alertStack(maxRows: 3)
        #expect(layout.rows.map(\.id) == [12_000_600, 12_000_601, 11_500_603] && layout.hidden == 0)
        #expect(rig.model.alertStack(maxRows: 2).hidden == 1)
    }

    @Test func aToastGoesToTheStormAndExpiresWithTheClock() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.handle(alerts: try liveAlerts(rig), backlog: false)
        let toast = try #require(rig.model.toasts.first)
        rig.model.activate(toast.action)
        #expect(rig.model.selection == .storm(id: 7))
        rig.clock.t += 2.9
        rig.model.expireTimers()
        #expect(rig.model.toasts.count == 1)
        rig.clock.t += 0.2
        rig.model.expireTimers()
        #expect(rig.model.toasts.isEmpty)
        rig.clock.t += 17
        rig.model.expireTimers()
        #expect(rig.model.tickerLine == nil, "the ticker line goes after seventeen seconds")
    }

    @Test func aStopWhileTheWindowIsAwayCallsTheDockOnceAndBadgesTheUnreadAlerts() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch, windowKey: false)
        defer { rig.play.shutdown() }
        // The fixture's last clock is an alert stop (reason 4, rate 0): the dock is asked, once.
        #expect(rig.dock.attention == 1, "the dock was asked once")
        // Alerts raised live and on the stack, not the chapter's own: the storm naming and the surge.
        rig.model.handle(alerts: try liveAlerts(rig), backlog: false)
        #expect(rig.dock.badge == "2" && rig.model.attention.unread == 2, "\(rig.dock.badge ?? "-")")
        rig.model.windowBecameKey()
        #expect(rig.dock.badge == nil && rig.model.attention.unread == 0)
        rig.model.handle(alerts: [], backlog: false)
        #expect(rig.dock.attention == 1)
    }

    @Test func withTheWindowInFrontTheDockIsLeftAlone() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch, windowKey: true)
        defer { rig.play.shutdown() }
        #expect(rig.dock.attention == 0 && rig.dock.badge == nil)
    }

    // MARK: Acknowledging, flying, watching

    @Test func kAcknowledgesTheNewestAlertOnTheStackOnePerPressAndTellsTheWorld() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.perform(.acknowledgeAlert)
        rig.model.perform(.acknowledgeAlert)
        #expect(await eventually { rig.sent == ["ack 12000601", "ack 12000600"] }, "\(rig.sent)")
        #expect(rig.model.alertStack(maxRows: 3).rows.map(\.id) == [11_500_603])
        rig.model.perform(.acknowledgeAlert)
        rig.model.perform(.acknowledgeAlert)  // nothing left
        #expect(await eventually { rig.sent.count == 3 })
        try await Task.sleep(for: .milliseconds(100))
        #expect(rig.sent == ["ack 12000601", "ack 12000600", "ack 11500603"])
    }

    @Test func theCheckButtonAcknowledgesThatRowAndNothingElseDoes() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.acknowledge(alert: 11_500_603)
        #expect(await sent(rig, "ack 11500603"))
        // An unknown id is not sent.
        rig.model.acknowledge(alert: 42)
        try await Task.sleep(for: .milliseconds(100))
        #expect(rig.sent == ["ack 11500603"])
        // Esc, Continue and a click on the row's arrow acknowledge nothing.
        rig.model.perform(.escape)
        rig.model.perform(.continueRun)
        rig.model.activate(AlertStack.row(for: try #require(rig.play.store.alerts.alert(id: 12_000_601))).action)
        #expect(rig.model.alertStack(maxRows: 3).rows.map(\.id) == [12_000_600, 12_000_601])
    }

    @Test func aFliesToTheNewestAlertAndSelectsItsStorm() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.perform(.flyToAlert)
        #expect(rig.model.selection == .storm(id: 7) && rig.model.isFlying)
    }

    @Test func theStarWatchesAtMostThreeGroups() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        for id in [17, 18, 19] { rig.model.toggleWatch(kin: id) }
        #expect(await eventually { rig.sent == ["watch 17", "watch 18", "watch 19"] })
        rig.model.toggleWatch(kin: 20)
        #expect(rig.model.note?.contains("Three groups") == true && rig.model.watch.ids == [17, 18, 19])
        rig.model.toggleWatch(kin: 18)
        #expect(await sent(rig, "unwatch 18"))
        try await Task.sleep(for: .milliseconds(100))
        #expect(rig.sent.count == 4, "the fourth star sent nothing")
    }

    // MARK: Esc

    @Test func escGoesOneLevelBackAndNeverResumesOrAcknowledges() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch, strikeReady: true)
        defer { rig.play.shutdown() }
        rig.play.store.apply(.clck(chapterClock(chapter: 2, step: 12_000)))
        rig.model.handleClock()
        rig.model.setSelection(.storm(id: 7))
        #expect(rig.model.chapterCard != nil && !rig.model.cardCollapsed)
        // Level 1: aiming a power.
        rig.model.choosePower(try #require(rig.model.offers.first { $0.name == .strike }))
        #expect(rig.model.powers.isAiming && rig.model.cardFolded, "aiming folds the card to a bar by itself")
        rig.model.perform(.escape)
        #expect(!rig.model.powers.isAiming && rig.model.powers.selected == nil && rig.model.selection == .storm(id: 7))
        // Level 2: the card folds to the map.
        rig.model.perform(.escape)
        #expect(rig.model.cardCollapsed && rig.model.selection == .storm(id: 7))
        // Level 3: the selection.
        rig.model.perform(.escape)
        #expect(rig.model.selection.isNone)
        // Nothing was sent that resumes the world or acknowledges an alert.
        try await Task.sleep(for: .milliseconds(100))
        #expect(!rig.sent.contains("run") && !rig.sent.contains { $0.hasPrefix("ack") }, "\(rig.sent)")
    }

    @Test func aNewChapterCardStartsUnfolded() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        rig.play.store.apply(.clck(chapterClock(chapter: 2, step: 12_000)))
        rig.model.handleClock()
        rig.model.perform(.escape)
        #expect(rig.model.cardCollapsed)
        rig.play.store.apply(.clck(chapterClock(chapter: 2, step: 12_000)))
        rig.model.handleClock()
        #expect(rig.model.cardCollapsed, "the same chapter's next clock does not unfold it")
        rig.play.store.apply(.clck(chapterClock(chapter: 3, step: 18_000)))
        rig.model.handleClock()
        #expect(!rig.model.cardCollapsed, "a new chapter, a new card")
    }

    // MARK: Continue

    @Test func gContinuesAStoppedWorldAndNeverPausesARunningOne() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        #expect(rig.model.clockStrip.paused)
        rig.model.perform(.continueRun)
        #expect(await sent(rig, "run"))
        rig.play.setRate(.multiplier(1))
        rig.model.perform(.continueRun)
        try await Task.sleep(for: .milliseconds(100))
        #expect(!rig.sent.contains("pause"), "continue does not pause: \(rig.sent)")
    }

    @Test func theStripNamesTheChapterThatEndedAndTheAlertThatStoppedTheWorld() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        // The fixture's last clock: reason 4 with the chapter alert (importance 3, pausing) on the stack.
        #expect(rig.model.clockStrip.readout == "Chapter 2 complete" || rig.model.clockStrip.readout == "alert"
                || !rig.model.clockStrip.readout.isEmpty)
        rig.play.store.apply(.clck(chapterClock(chapter: 2, step: 12_000, reason: .chapter)))
        #expect(rig.model.clockStrip.readout == "Chapter 2 complete")
        rig.play.store.apply(.clck(chapterClock(chapter: 2, step: 12_000, reason: .alert)))
        #expect(rig.model.clockStrip.readout == "Chapter 2 complete", "the pausing alert on the stack is the chapter's: its title")
        rig.model.acknowledge(alert: 12_000_600)
        #expect(rig.model.clockStrip.readout == "alert", "nothing pausing is left on the stack")
    }

    @Test func theChapterAlertIsOffTheStackWhileTheCardIsUp() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        // Stopped by the alert: no card, so the chapter's alert is a row like another.
        rig.play.store.apply(.clck(chapterClock(chapter: 2, step: 12_000, reason: .alert)))
        #expect(rig.model.chapterCard == nil)
        let without = rig.model.alertStack(maxRows: 5)
        #expect(without.rows.map(\.id).contains(12_000_600))
        // The card is up: it, and the strip's "Chapter 2 complete", announce the chapter; a row would say it a third time in a
        // dock that has no room (third review, stage 8). The other alerts keep their rows, and the count stays right.
        rig.play.store.apply(.clck(chapterClock(chapter: 2, step: 12_000, reason: .chapter)))
        #expect(rig.model.chapterCard != nil)
        let with = rig.model.alertStack(maxRows: 5)
        #expect(!with.rows.map(\.id).contains(12_000_600))
        #expect(with.rows.map(\.id) == without.rows.map(\.id).filter { $0 != 12_000_600 })
        #expect(rig.model.clockStrip.readout == "Chapter 2 complete")
    }

    // MARK: The chapter notice (the pause is off)

    @Test func aWorldThatRunsOnShowsANoticeAndItGoesByItself() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        rig.play.store.apply(.clck(chapterClock(chapter: 3, step: 18_000, rate: 1, flags: 2)))
        rig.model.handleClock()
        #expect(rig.model.chapterNotice == 3 && rig.model.chapterCard?.form == .notice)
        rig.clock.t += WorldModel.noticeSeconds - 1
        rig.model.expireTimers()
        #expect(rig.model.chapterNotice == 3)
        rig.clock.t += 2
        rig.model.expireTimers()
        #expect(rig.model.chapterNotice == nil && rig.model.chapterCard == nil)
        // The same chapter's later clocks do not bring it back; the player can close it sooner.
        rig.model.handleClock()
        rig.play.store.apply(.clck(chapterClock(chapter: 4, step: 24_000, rate: 1, flags: 2)))
        rig.model.handleClock()
        #expect(rig.model.chapterNotice == 4)
        rig.model.dismissNotice()
        #expect(rig.model.chapterCard == nil)
    }

    @Test func disarmingThePauseWhileStoppedAtTheCardDoesNotMakeItANotice() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        // Stopped at the chapter's end, the player turns the lock off: the card stays (the world is still stopped).
        rig.play.store.apply(.clck(chapterClock(chapter: 3, step: 18_000, rate: 0, flags: 0)))
        rig.model.handleClock()
        #expect(rig.model.chapterNotice == nil && rig.model.chapterCard?.form == .ready)
    }

    @Test func anEventDoesNotToastInAWorldThatHasAnAlertEngine() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.resetInteraction()
        let event = WorldEvent(id: 5000, step: 5, kind: EventKind.stormDissolved, fam: "storm", a: 7, b: 0, x: 4.5, y: 20.25, z: nil, imp: 2, text: "Storm Merrow dissolved")
        rig.model.handle(events: [event], backlog: false)
        #expect(rig.model.toasts.isEmpty, "its alert is the toast; the event is the chronicle's and the ticker's")
        #expect(rig.model.tickerLine?.text == "Storm Merrow dissolved")
        rig.model.handle(events: [event], backlog: true)
        #expect(rig.model.toasts.isEmpty)
    }

    // MARK: The pauses the player armed

    @Test func theChoiceIsSentRightAfterTheHeaderAndKept() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        // The fixture's header: chapter on, `pop_fall` on, `storm_named` off.
        let rig = try await makeDecideRig(scratch, prefs: AutopausePrefs(chapter: false, alert: ["pop_fall": false, "storm_named": true, "ghost": true]))
        defer { rig.play.shutdown() }
        #expect(await eventually { rig.sent.count >= 3 })
        #expect(rig.sent == ["autopause chapter off", "autopause alert pop_fall off", "autopause alert storm_named on"], "\(rig.sent)")
        rig.model.setAlertPause("pop_surge", true)
        rig.model.setChapterPause(true)
        #expect(await eventually { rig.sent.suffix(2) == ["autopause alert pop_surge on", "autopause chapter on"] })
        let saved = AutopausePrefs(json: rig.defaults.string(forKey: WorldModel.autopauseKey))
        #expect(saved.chapter == true && saved.alert["pop_surge"] == true && saved.alert["pop_fall"] == false)
        #expect(rig.model.alertPauseOn("pop_surge") && !rig.model.alertPauseOn("pop_fall") && rig.model.alertPauseOn("storm_named"))
    }

    @Test func aChoiceThatIsTheWorldsDefaultSendsNothing() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        try await Task.sleep(for: .milliseconds(150))
        #expect(rig.sent.isEmpty, "\(rig.sent)")
    }

    @Test func theLockTogglesTheChapterPauseFromTheClocksFlag() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        #expect(rig.play.store.clock?.chapterPauseArmed == true)
        rig.model.toggleChapterPause()
        #expect(await sent(rig, "autopause chapter off"))
        rig.play.store.apply(.clck(chapterClock(chapter: 2, step: 12_000, flags: 0)))
        rig.model.toggleChapterPause()
        #expect(await sent(rig, "autopause chapter on"))
        #expect(rig.model.pausableClasses.map(\.name) == ["pop_fall", "pop_surge", "storm_named"].sorted { a, b in
            (rig.play.store.play?.alerts?.classes[a]?.label ?? "") < (rig.play.store.play?.alerts?.classes[b]?.label ?? "") },
                "the stack's classes by label; the log's and the chapter's are not choices")
    }

    // MARK: Powers

    @Test func theTwoClickFlowSendsAPreviewThenThePowerAndNothingElse() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        let food = try #require(rig.model.offers.first { $0.name == .food })
        rig.model.tab = .powers
        rig.model.choosePower(food)
        // (the fixture's requests are 3, 4 and 9: the window's own start past them)
        #expect(await sent(rig, "power preview food id=10 mult=1.3 steps=6000"))
        hear(rig, plan(10, .preview))
        #expect(rig.model.powers.phase == .previewed(id: 10))
        rig.model.perform(.confirmPower)
        #expect(await eventually { rig.sent == ["power preview food id=10 mult=1.3 steps=6000", "power food id=10 mult=1.3 steps=6000"] })
        hear(rig, plan(10, .queued))
        hear(rig, plan(10, .applied))
        #expect(rig.model.powers.phase == .idle)
        try await Task.sleep(for: .milliseconds(100))
        #expect(rig.sent.count == 2)
    }

    @Test func returnSendsAPowerOnlyWhileThePowersTabShowsItsPreview() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.tab = .powers
        rig.model.choosePower(try #require(rig.model.offers.first { $0.name == .food }))
        #expect(await sent(rig, "power preview food id=10 mult=1.3 steps=6000"))
        hear(rig, plan(10, .preview))
        #expect(rig.model.returnConfirmsPower)
        // the player looks at the Cast, folds the card, lets the world run: a stray Return does not apply the old preview
        rig.model.tab = .cast
        rig.model.cardCollapsed = true
        #expect(!rig.model.returnConfirmsPower)
        rig.model.perform(.confirmPower)
        try await Task.sleep(for: .milliseconds(100))
        #expect(rig.sent == ["power preview food id=10 mult=1.3 steps=6000"], "nothing was applied")
        // back on the Powers tab the preview is there again and Return sends it
        rig.model.tab = .powers
        #expect(rig.model.returnConfirmsPower)
        rig.model.perform(.confirmPower)
        #expect(await eventually { rig.sent.last == "power food id=10 mult=1.3 steps=6000" })
    }

    @Test func aNewChapterDropsTheUnsentPreviewOfTheChapterBefore() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.tab = .powers
        rig.model.choosePower(try #require(rig.model.offers.first { $0.name == .food }))
        #expect(await sent(rig, "power preview food id=10 mult=1.3 steps=6000"))
        hear(rig, plan(10, .preview))
        #expect(rig.model.powers.phase == .previewed(id: 10))
        rig.play.store.apply(.clck(chapterClock(chapter: 4, step: 24_000, rate: 0, flags: 1)))
        rig.model.handleClock()
        #expect(rig.model.powers.phase == .idle && !rig.model.returnConfirmsPower)
    }

    @Test func returnWithNothingPreviewedSendsNothing() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.perform(.confirmPower)
        try await Task.sleep(for: .milliseconds(100))
        #expect(rig.sent.isEmpty)
    }

    @Test func aStrikeIsAimedWithTheMapAndAClickDoesNotSelectAnything() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch, strikeReady: true)
        defer { rig.play.shutdown() }
        let strike = try #require(rig.model.offers.first { $0.name == .strike })
        rig.model.choosePower(strike)
        #expect(rig.model.powers.isAiming)
        let projector = rig.model.projector(size: CGSize(width: 900, height: 700))
        let at = try #require(projector.screen(SIMD3(30, 20, 0)))
        rig.model.click(at: at, shift: false, clicks: 1)
        #expect(await eventually { rig.sent.count == 1 })
        let words = try #require(rig.sent.first).split(separator: " ").map(String.init)
        #expect(Array(words.prefix(4)) == ["power", "preview", "strike", "id=10"], "\(words)")
        let x = try #require(words.first { $0.hasPrefix("x=") }.flatMap { Double($0.dropFirst(2)) })
        let y = try #require(words.first { $0.hasPrefix("y=") }.flatMap { Double($0.dropFirst(2)) })
        #expect(abs(x - 30) < 0.5 && abs(y - 20) < 0.5, "the click named the floor under it: \(x), \(y)")
        #expect(words.contains("radius=15.625") && words.contains("kill=0.45"))
        #expect(rig.model.selection.isNone && rig.model.followTarget == .none, "no focus or follow line: the map was a target")
        // A double click and a hover are not a second aim.
        rig.model.click(at: at, shift: false, clicks: 2)
        rig.model.hover(at: at)
        try await Task.sleep(for: .milliseconds(100))
        #expect(rig.sent.count == 1)
        #expect(rig.model.overlay.aimWorld != nil, "the pointer's place on the floor is what the overlay draws")
    }

    @Test func aBusyPowerIsNotChosenAndEndingAnEffectEarlyIsOneCancelLine() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        let strike = try #require(rig.model.offers.first { $0.name == .strike })
        #expect(!strike.usable && strike.busy == "cooldown until step 15000")
        rig.model.choosePower(strike)
        #expect(rig.model.powers.phase == .idle)
        let running = ActiveEffect.running(rig.play.store)
        #expect(running.map(\.name) == [.food])
        rig.model.cancelPower(try #require(running.first).name)
        #expect(await sent(rig, "power cancel food id=10"))
    }

    @Test func theCostOneSettingSkipsTheSecondClick() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch)
        defer { rig.play.shutdown() }
        rig.defaults.set(true, forKey: WorldModel.skipConfirmKey)
        rig.model.choosePower(try #require(rig.model.offers.first { $0.name == .food }))
        hear(rig, plan(10, .preview, cost: 1))
        #expect(await eventually { rig.sent == ["power preview food id=10 mult=1.3 steps=6000", "power food id=10 mult=1.3 steps=6000"] })
    }

    // MARK: A new world

    @Test func aNewWorldForgetsTheDecideState() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeDecideRig(scratch, windowKey: false, strikeReady: true)
        defer { rig.play.shutdown() }
        rig.model.handle(alerts: try liveAlerts(rig), backlog: false)
        rig.model.toggleWatch(kin: 17)
        rig.model.choosePower(try #require(rig.model.offers.first { $0.name == .strike }))
        rig.model.cardCollapsed = true
        #expect(rig.dock.badge != nil && rig.model.tickerLine != nil)
        rig.model.resetInteraction()
        #expect(rig.model.watch.ids.isEmpty && rig.model.powers.phase == .idle && !rig.model.cardCollapsed)
        #expect(rig.model.tickerLine == nil && rig.model.toasts.isEmpty && rig.dock.badge == nil && rig.model.chapterNotice == nil)
    }
}
