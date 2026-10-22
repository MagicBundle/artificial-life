import Foundation
import Testing

@testable import AlifeCore

/// The chapter card, its since strip and the Report tab, on the committed `play_tiny3d_g3b` stream (chapter 2 ended at step 12,000
/// with a report, the player's Food running to 18,000, four units of six free, two alerts on the stack) and on records built by hand.
@Suite struct ChapterCardTests {
    private func clock(chapter: UInt32, step: UInt64 = 12_000, rate: Float = 0, reason: PlayClock.Reason = .chapter,
                       saving: Bool = false, flags: UInt16 = 1, inChapter: UInt32 = 6000) -> PlayClock {
        PlayClock(head: RecordHead(step: step, seq: 0, version: 1, flags: 0), rate: rate, achieved: rate == 0 ? 0 : 14.4, stepMs: 15,
                  frameMs: 3, lateSteps: 0, saveStallSteps: 0, reason: reason, saving: saving, flags: flags, fe: 1, chapter: chapter,
                  inChapter: inChapter, chapterLen: 6000)
    }

    private func store(_ clock: PlayClock) throws -> WorldStore {
        var store = try WorldFixtures.store("3d_g3b")
        store.apply(.clck(clock))
        return store
    }

    // MARK: The card

    @Test func aStoppedChapterEndIsTheWholeCardWithItsReport() throws {
        let card = try #require(ChapterCard.make(store: store(clock(chapter: 2)), originChapter: 0))
        #expect(card.form == .ready && !card.saving && card.title == "Chapter 2 complete" && card.nAbs == 2 && card.number == 2)
        let report = try #require(card.report)
        #expect(report.headline?.text == "Storm Merrow reached named strength" && report.headline?.id == "HS")
        #expect(report.headline?.action == ChronicleAction(at: SIMD2(4.5, 20.25), select: .storm(id: 7)), "the headline flies to the storm")
        #expect(report.lead?.action == nil, "a sentence with no entity goes nowhere")
        #expect(report.body.map(\.id) == ["TO", "K1"] && report.body[1].action?.select == .kin(id: 17))
        #expect(report.interventions.map(\.id) == ["I1"] && report.hiddenBody == 0 && !report.quiet && report.withheld == nil)
        #expect(card.previousHeadline == nil, "this chapter has its report: the old headline is not shown")
    }

    @Test func theCardIsUpWhileTheCheckpointIsWrittenWithTheReportOrTheLastHeadline() throws {
        // Chapter 3 ended: its report is not in yet. The card shows the previous headline, greyed, and no buttons.
        let building = try #require(ChapterCard.make(store: store(clock(chapter: 3, step: 18_000, rate: 1, saving: true)), originChapter: 0))
        #expect(building.form == .saving && building.saving && building.report == nil)
        #expect(building.previousHeadline == "Storm Merrow reached named strength")
        // The same chapter the moment its report has landed: the card fills in.
        let filled = try #require(ChapterCard.make(store: store(clock(chapter: 2, rate: 1, saving: true)), originChapter: 0))
        #expect(filled.saving && filled.report?.headline?.text == "Storm Merrow reached named strength")
    }

    @Test func aWorldWhosePauseIsOffGetsTheNoticeEvenWhileItSaves() throws {
        let saving = try #require(ChapterCard.make(store: store(clock(chapter: 2, rate: 1, saving: true, flags: 2)), originChapter: 0))
        #expect(saving.form == .notice && saving.report != nil, "it keeps running through the save: no card with no buttons")
        // The world that is about to stop is a card from the start.
        let stopping = try #require(ChapterCard.make(store: store(clock(chapter: 2, rate: 1, saving: true, flags: 1)), originChapter: 0))
        #expect(stopping.form == .saving)
    }

    @Test func aWorldThatRunsOnShowsANoticeUntilItIsClosed() throws {
        // Pause off: reason 3 at a running rate with the pause flag clear.
        let run = clock(chapter: 2, rate: 1, flags: 2)
        let notice = try #require(ChapterCard.make(store: store(run), originChapter: 0, noticeVisible: true))
        #expect(notice.form == .notice)
        #expect(ChapterCard.make(store: try store(run), originChapter: 0, noticeVisible: false) == nil, "closed or timed out")
        // With the pause armed the same clock (the instant before the world stops) is the card, not a notice.
        let armed = try #require(ChapterCard.make(store: store(clock(chapter: 2, rate: 1, flags: 1)), originChapter: 0, noticeVisible: false))
        #expect(armed.form == .ready)
    }

    @Test func thereIsNoCardUnlessTheClockIsAtAChapterEnd() throws {
        for reason in [PlayClock.Reason.running, .userPause, .endOfSteps, .alert, .error] {
            #expect(ChapterCard.make(store: try store(clock(chapter: 2, reason: reason)), originChapter: 0) == nil, "\(reason)")
        }
        #expect(ChapterCard.make(store: WorldStore(), originChapter: 0) == nil, "no clock yet")
    }

    @Test func theNumberCountsFromTheForksOrigin() throws {
        let card = try #require(ChapterCard.make(store: store(clock(chapter: 2)), originChapter: 1))
        #expect(card.number == 1 && card.title == "Chapter 1 complete" && card.nAbs == 2)
        #expect(ChapterCard.make(store: try store(clock(chapter: 2)), originChapter: 2)?.number == 1, "never below 1")
    }

    @Test func aReportThatCouldNotBeCutIsSaidSoAndNothingIsGuessed() throws {
        // The fixture's mode-2 record of chapter 2 is withheld. A withheld template record for chapter 5 shows its reason only.
        var s = try store(clock(chapter: 5, step: 30_000))
        let body = ReportBody(
            chapter: 5, nAbs: 5, a: 24_000, b: 30_000, mode: 0, final: true, withheld: ["the cut 24000 is not the step of a metrics row"],
            headline: nil, lead: nil, body: [], interventions: [], ticker: [],
            since: .init(influence: nil, alerts: [:], ending: []), entities: [:],
            versions: .init(report: 2, template: 2, checker: 1, lexicon: 3), factsDigest: "d", latencyS: nil, fallback: nil)
        s.apply(.rprt(PlayReport(head: RecordHead(step: 30_000, seq: 9, version: 1, flags: 2), report: body)))
        let report = try #require(ChapterCard.make(store: s, originChapter: 0)?.report)
        #expect(report.withheld == "the cut 24000 is not the step of a metrics row" && report.headline == nil && report.body.isEmpty)
    }

    // MARK: The report card

    private func record(body lines: Int, headline: String = "HQ") -> PlayReport {
        let body = ReportBody(
            chapter: 7, nAbs: 7, a: 0, b: 6000, mode: 0, final: true, withheld: [], headline: .init(id: headline, text: "A quiet chapter", refs: []),
            lead: .init(id: "L", text: "lead", refs: []), body: (0..<lines).map { .init(id: "B\($0)", text: "line \($0)", origin: "sheet", refs: []) },
            interventions: [], ticker: [.init(step: 5, text: "tick")], since: .init(influence: nil, alerts: [:], ending: []), entities: [:],
            versions: .init(report: 2, template: 2, checker: 1, lexicon: 3), factsDigest: "d", latencyS: nil, fallback: nil)
        return PlayReport(head: RecordHead(step: 6000, seq: 0, version: 1, flags: 2), report: body)
    }

    @Test func theCardKeepsSixBodyLinesAndSaysHowManyAreInTheTab() {
        let card = ReportCard.make(record(body: 9), bodyLimit: ReportCard.cardBodyLines)
        #expect(card.body.count == 6 && card.hiddenBody == 3 && ReportCard.cardBodyLines == 6)
        let all = ReportCard.make(record(body: 9))
        #expect(all.body.count == 9 && all.hiddenBody == 0, "the tab shows every sentence")
    }

    @Test func aQuietChapterIsMarkedByItsHeadlineAndAPendingReportBySelectMode() {
        #expect(ReportCard.make(record(body: 0, headline: "HQ")).quiet)
        #expect(!ReportCard.make(record(body: 0, headline: "HS")).quiet)
        var pending = record(body: 1)
        pending.head.flags = 0  // not final: a select report on its way
        #expect(ReportCard.make(pending).ordering && !ReportCard.make(record(body: 1)).ordering)
    }

    @Test func aSentenceFliesToTheFirstEntityItNamesThatTheReportCanPlace() {
        let entities: [String: ReportBody.Entity] = [
            "K1": .init(kind: "kin", id: 1, name: "A", x: nil, y: nil), "S2": .init(kind: "storm", id: 2, name: "B", x: 3, y: 4),
            "H3": .init(kind: "house", id: 3, name: "C", x: 7, y: 8),
        ]
        #expect(ChronicleAction.of(refs: ["K1"], entities: entities) == ChronicleAction(at: nil, select: .kin(id: 1)))
        #expect(ChronicleAction.of(refs: ["nope", "S2", "K1"], entities: entities) == ChronicleAction(at: SIMD2(3, 4), select: .storm(id: 2)))
        #expect(ChronicleAction.of(refs: ["H3"], entities: entities) == ChronicleAction(at: SIMD2(7, 8), select: .none), "a house has a place, not a cast row")
        #expect(ChronicleAction.of(refs: [], entities: entities) == nil && ChronicleAction.of(refs: ["nope"], entities: entities) == nil)
        #expect(ChronicleAction.of(refs: ["K1"], entities: [:]) == nil)
    }

    // MARK: The since strip

    @Test func theStripSaysWhatHasChangedForThePlayer() throws {
        let card = try #require(ChapterCard.make(store: store(clock(chapter: 2)), originChapter: 0))
        let strip = card.since
        #expect(strip.haveUnits == 4 && strip.capUnits == 6 && strip.nextUnit == 13_000)
        #expect(strip.ready == ["Food"], "Strike is cooling down; Bloom and Blight are not in this world's header")
        #expect(strip.ending == [.init(label: "Food", end: 18_000)], "the player's Food ends within a chapter; the rule's Bloom has no end")
        #expect(strip.alertCounts == [.init(label: "Population surge", count: 1)], "from the report, with the class's label")
        #expect(strip.top.map(\.id) == [12_000_601, 11_500_603], "the chapter's own alert and the cleared one are not in it")
        #expect(!strip.isEmpty)
    }

    @Test func aStripWithNothingToSayIsEmptyAndFullUnitsHaveNoNextOne() throws {
        #expect(SinceStrip.make(store: WorldStore(), range: 0..<1, reported: nil, horizon: 6000).isEmpty)
        var s = try store(clock(chapter: 2))
        var env = try #require(s.environment)
        env.environment.influence = .init(have: 6, cap: 6, window: 18_000, freeAt: [])
        s.apply(.envs(env))
        #expect(SinceStrip.make(store: s, range: 6001..<12_001, reported: nil, horizon: 6000).nextUnit == nil)
    }

    @Test func withoutAReportTheAlertsAreCountedOnTheBoard() throws {
        let s = try store(clock(chapter: 2))
        let strip = SinceStrip.make(store: s, range: 6001..<12_001, reported: nil, horizon: 6000)
        // the storm naming and the surge (the chapter alert and the cleared fall are not counted)
        #expect(strip.alertCounts.map(\.count).reduce(0, +) == 2 && Set(strip.alertCounts.map(\.label)) == ["Storm named", "Population surge"])
    }

    @Test func anEffectThatEndsLaterThanTheHorizonIsNotListed() throws {
        let s = try store(clock(chapter: 2))
        #expect(SinceStrip.make(store: s, range: 6001..<12_001, reported: nil, horizon: 5999).ending.isEmpty)
        #expect(SinceStrip.make(store: s, range: 6001..<12_001, reported: nil, horizon: 6000).ending.count == 1)
    }

    // MARK: The Report tab

    @Test func theTabHoldsTheNewestInFullAndTheOthersAsHeadlines() throws {
        let s = try WorldFixtures.store("3d_g3b")
        let tab = ReportTabModel(shelf: s.reports)
        #expect(tab.newest?.chapter == 2 && tab.earlier.map(\.chapter) == [1])
        #expect(tab.newest?.body.count == 2 && tab.newest?.hiddenBody == 0, "the tab is not capped")
        #expect(ReportTabModel(shelf: WorldStore().reports).newest == nil)
    }

    @Test func aReportIsFoundByItsAbsoluteChapterAndTheTemplateBeatsAPendingSelect() throws {
        let s = try WorldFixtures.store("3d_g3b")
        #expect(s.reports.report(nAbs: 2)?.mode == 0 && s.reports.report(nAbs: 1)?.report.chapter == 1)
        #expect(s.reports.report(nAbs: 9) == nil)
    }
}
