import Foundation
import Testing

@testable import AlifeCore

/// The reducers G3b adds to the World window's store: the alert board, the report shelf and the plan log
/// (DESIGN §34.3, §34.4, §34.5), on the committed `play_tiny3d_g3b` stream and on records built by hand.
@Suite struct WorldStoreG3bTests {
    private func alert(_ id: UInt64, version: Int = 1, state: PlayAlert.State = .raised, surface: PlayAlert.Surface = .alert,
                       imp: Int = 2, expires: Int? = nil, text: String = "") -> PlayAlert {
        PlayAlert(id: id, version: version, step: Int(id / 1000), alertClass: "pop_fall", key: "world", state: state,
                  surface: surface, imp: imp, title: "alert \(id)", text: text.isEmpty ? "text \(id)" : text, x: nil, y: nil,
                  z: nil, ref: nil, pause: false, expires: expires, watched: false, calibrated: false)
    }

    private func batch(_ alerts: [PlayAlert], backlog: Bool = false) -> PlayAlertBatch {
        PlayAlertBatch(head: RecordHead(step: 0, seq: 0, version: 1, flags: backlog ? 1 : 0), alerts: alerts)
    }

    private func report(chapter: Int, mode: Int = 0, final: Bool = true, step: UInt64 = 0, text: String = "headline",
                        backlog: Bool = false) -> PlayReport {
        let body = ReportBody(
            chapter: chapter, nAbs: chapter + 74, a: 0, b: 6000, mode: mode, final: final, withheld: [],
            headline: .init(id: "HS", text: text, refs: []), lead: nil, body: [], interventions: [], ticker: [],
            since: .init(influence: nil, alerts: [:], ending: []), entities: [:],
            versions: .init(report: 2, template: 2, checker: 1, lexicon: 3), factsDigest: "d", latencyS: nil, fallback: nil)
        let flags = (backlog ? 1 : 0) | (final ? 2 : 0)
        return PlayReport(head: RecordHead(step: step, seq: 0, version: 1, flags: UInt16(flags)), report: body)
    }

    private func plan(_ id: Int, _ status: PlanBody.Status, step: UInt64 = 0) -> PlayPlan {
        let body = PlanBody(
            id: id, kind: .power, status: status, power: "food", action: "regen_pulse", raw: [:], params: [:], clamped: [],
            duration: 6000, stepApplied: status == .applied ? 12000 : nil, ends: nil, cost: 1, influence: nil, ready: nil,
            holds: nil, text: "Food x1.3", caution: [], precedent: nil, since: nil,
            refused: status == .refused ? "busy" : nil)
        return PlayPlan(head: RecordHead(step: step, seq: 0, version: 1, flags: status == .preview ? 1 : 0), plan: body)
    }

    // MARK: The stream's records

    @Test func everyNewRecordOfTheFixtureLandsInItsStore() throws {
        let events = try PlayFixture(kind: "3d_g3b").events()
        guard case .header(let header)? = events.first else { Issue.record("no header"); return }
        var store = WorldStore()
        store.reset(header: header)
        var changes: WorldStore.Changes = []
        for case .record(let record) in events { changes.formUnion(store.apply(record)) }
        #expect(changes.isSuperset(of: [.alerts, .reports, .plans, .environment, .clock]))
        // Alerts: the backlog pop_fall (raised at 11250), the six of the live batch, one id cleared by the live batch.
        #expect(store.alerts.alerts.count == 6, "ids: 11250600 (cleared), 11500603, 11750605, 12000600 to 12000602")
        #expect(store.alerts.alert(id: 11_250_600)?.state == .cleared && store.alerts.alert(id: 11_250_600)?.version == 3)
        #expect(store.alerts.alert(id: 11_500_603)?.version == 2)
        #expect(store.alerts.stack(atStep: 12000).map(\.id) == [12_000_601, 12_000_600, 11_500_603],
                "the alert surface, newest first, no cleared one, 11750605 is a log row")
        #expect(store.alerts.logRows.map(\.id) == [12_000_602, 11_750_605])
        // Reports: chapter 1, and chapter 2 in three modes.
        #expect(store.reports.chapters == [2, 1] && store.reports.entries.count == 4)
        #expect(store.reports.current(chapter: 2)?.mode == 0, "the pending select report does not replace the template")
        #expect(store.reports.report(chapter: 2, mode: 2)?.report.isWithheld == true)
        // Plans: seven records, three requests (3, 4 and the intent 9).
        #expect(store.plans.records.count == 7 && store.plans.cards.map(\.plan.id) == [9, 3, 4])
        #expect(store.plans.card(id: 3)?.plan.status == .cancelled, "the newest record of request 3")
        #expect(store.plans.card(id: 9)?.plan.kind == .intent)
    }

    @Test func aNewStreamForgetsTheOldAlertsReportsAndPlans() throws {
        let events = try PlayFixture(kind: "3d_g3b").events()
        guard case .header(let header)? = events.first else { Issue.record("no header"); return }
        var store = WorldStore()
        store.reset(header: header)
        for case .record(let record) in events { store.apply(record) }
        store.reset(header: header)
        #expect(store.alerts.alerts.isEmpty && store.reports.entries.isEmpty && store.plans.records.isEmpty)
        #expect(store.alerts.acknowledged.isEmpty)
    }

    @Test func aRecordReportsItsChangeAndTheRevision() {
        var store = WorldStore()
        let before = store.revision
        #expect(store.apply(.alrt(batch([alert(1_000_600)]))) == .alerts && store.revision == before + 1)
        #expect(store.apply(.alrt(batch([alert(1_000_600)]))).isEmpty, "the same alert again is nothing")
        #expect(store.revision == before + 1)
        #expect(store.apply(.rprt(report(chapter: 1))) == .reports && store.revision == before + 2)
        #expect(store.apply(.rprt(report(chapter: 1))).isEmpty, "the same report again is nothing")
        #expect(store.apply(.plan(plan(1, .preview))) == .plans && store.revision == before + 3)
    }

    // MARK: The alert board

    @Test func alertsAreKeptInOrderOfTheirIdsAndTheNewestVersionWins() {
        var board = AlertBoard()
        let first = board.insert([alert(5_000_600), alert(3_000_600), alert(4_000_600)])
        #expect(first.map(\.kind) == [.raised, .raised, .raised] && first.map(\.alert.id) == [5_000_600, 3_000_600, 4_000_600])
        #expect(board.alerts.map(\.id) == [3_000_600, 4_000_600, 5_000_600])
        // An older or equal version is ignored; a higher one replaces the row and says it was updated.
        #expect(board.insert([alert(4_000_600, version: 1, text: "stale")]).isEmpty)
        let updated = board.insert([alert(4_000_600, version: 2, state: .updated, text: "worse")])
        #expect(updated.map(\.kind) == [.updated] && board.alert(id: 4_000_600)?.text == "worse")
        #expect(board.insert([alert(4_000_600, version: 2, state: .updated, text: "again")]).isEmpty, "version 2 again")
        #expect(board.alerts.count == 3)
        // The end of the condition: cleared, kept in the ring, off the stack, and not brought back by a late old version.
        let cleared = board.insert([alert(4_000_600, version: 3, state: .cleared)])
        #expect(cleared.map(\.kind) == [.cleared])
        #expect(board.stack(atStep: 6000).map(\.id) == [5_000_600, 3_000_600])
        #expect(board.insert([alert(4_000_600, version: 2, state: .updated)]).isEmpty)
        #expect(board.alert(id: 4_000_600)?.state == .cleared)
    }

    @Test func anAlertSeenFirstAsUpdatedOrClearedIsStillOnTheBoard() {
        var board = AlertBoard()
        let changes = board.insert([alert(2_000_600, version: 3, state: .updated), alert(1_000_600, version: 4, state: .cleared)])
        #expect(changes.map(\.kind) == [.raised, .cleared], "an unknown id is new, an unknown cleared id is a clear")
        #expect(board.stack(atStep: 3000).map(\.id) == [2_000_600])
    }

    @Test func theBoardKeepsTheNewestTwoHundred() {
        var board = AlertBoard()
        board.insert((1...220).map { alert(UInt64($0) * 1000 + 600) })
        #expect(board.alerts.count == AlertBoard.capacity)
        #expect(board.alerts.first?.id == 21_600 && board.alerts.last?.id == 220_600)
        // An alert older than the ring is not new, however often it is sent; a newer one is.
        #expect(board.insert([alert(5_600)]).isEmpty)
        #expect(board.insert([alert(221_600)]).map(\.alert.id) == [221_600])
        #expect(board.alerts.count == AlertBoard.capacity)
    }

    @Test func theStackShowsTheAlertSurfaceMinusClearedAcknowledgedAndExpired() {
        var board = AlertBoard()
        board.insert([
            alert(1_000_600, imp: 3),  // no expiry: stays until cleared or acknowledged
            alert(2_000_600, expires: 4_500), alert(3_000_600, surface: .log), alert(4_000_600, expires: 9_000),
        ])
        #expect(board.stack(atStep: 4_000).map(\.id) == [4_000_600, 2_000_600, 1_000_600], "newest first; the log row is not here")
        #expect(board.stack(atStep: 4_500).map(\.id) == [4_000_600, 1_000_600], "expires is exclusive")
        #expect(board.stack(atStep: 20_000).map(\.id) == [1_000_600], "the importance-3 alert stays")
        #expect(board.logRows.map(\.id) == [3_000_600])
        #expect(board.newestUnacknowledged(atStep: 4_000)?.id == 4_000_600)
        // Acknowledging is explicit and by id; an unknown id is refused; the next one is then the newest.
        let known = board.acknowledge(id: 4_000_600), unknown = board.acknowledge(id: 99)
        #expect(known && !unknown)
        #expect(board.newestUnacknowledged(atStep: 4_000)?.id == 2_000_600)
        #expect(board.stack(atStep: 4_000).map(\.id) == [2_000_600, 1_000_600])
        #expect(board.alert(id: 4_000_600) != nil, "acknowledging removes it from the stack, not from the board")
    }

    @Test func anAcknowledgementThroughTheStoreIsARevision() {
        var store = WorldStore()
        store.apply(.alrt(batch([alert(1_000_600)])))
        let before = store.revision
        let known = store.acknowledge(alert: 1_000_600)
        #expect(known && store.revision == before + 1)
        let unknown = store.acknowledge(alert: 7)
        #expect(!unknown && store.revision == before + 1)
    }

    @Test func onlyANewAlertOfTheStackSurfaceAtImportanceTwoToastsAndNeverTheBacklog() {
        var board = AlertBoard()
        let live = board.insert([
            alert(1_000_600, imp: 1), alert(2_000_600, imp: 2), alert(3_000_600, imp: 3),
            alert(4_000_600, surface: .log, imp: 3), alert(5_000_600, imp: 0),
        ])
        #expect(AlertBoard.toasts(from: live, backlog: false).map(\.id) == [2_000_600, 3_000_600])
        #expect(AlertBoard.toasts(from: live, backlog: true).isEmpty)
        // An update (a worse number, a higher importance) is not a second toast, and a clear is not one.
        let later = board.insert([alert(1_000_600, version: 2, state: .updated, imp: 3), alert(2_000_600, version: 2, state: .cleared)])
        #expect(AlertBoard.toasts(from: later, backlog: false).isEmpty)
    }

    // MARK: The report shelf

    @Test func theShelfKeepsTheNewestReportOfEachChapterAndMode() {
        var shelf = ReportShelf()
        let kept1 = shelf.insert(report(chapter: 1, step: 6000, text: "old"))
        #expect(kept1)
        let kept2 = shelf.insert(report(chapter: 1, step: 12000, text: "new"))
        #expect(kept2, "a later record of the same key replaces it")
        #expect(shelf.report(chapter: 1, mode: 0)?.report.headline?.text == "new")
        let kept3 = shelf.insert(report(chapter: 1, step: 6000, text: "stale"))
        #expect(!kept3, "an earlier one does not")
        let kept4 = shelf.insert(report(chapter: 1, step: 12000, text: "new"))
        #expect(!kept4, "the same one is nothing")
        #expect(shelf.entries.count == 1)
    }

    @Test func aPendingSelectReportNeverReplacesAFinalOneAndAFinalOneReplacesTheTemplateInView() {
        var shelf = ReportShelf()
        shelf.insert(report(chapter: 3, mode: 0, step: 18000, text: "template"))
        shelf.insert(report(chapter: 3, mode: 1, final: false, step: 18000, text: "pending"))
        #expect(shelf.current(chapter: 3)?.report.headline?.text == "template", "a pending select report is not the report")
        shelf.insert(report(chapter: 3, mode: 1, final: true, step: 18000, text: "select"))
        #expect(shelf.current(chapter: 3)?.report.headline?.text == "select")
        let kept5 = shelf.insert(report(chapter: 3, mode: 1, final: false, step: 18001, text: "regressed"))
        #expect(!kept5)
        #expect(shelf.current(chapter: 3)?.report.headline?.text == "select")
        #expect(shelf.report(chapter: 3, mode: 0)?.report.headline?.text == "template", "the template stays on the shelf")
        // With only a pending select report on the shelf it is what there is.
        var only = ReportShelf()
        only.insert(report(chapter: 4, mode: 1, final: false))
        #expect(only.current(chapter: 4)?.mode == 1)
        // The away report is its own thing.
        var away = ReportShelf()
        away.insert(report(chapter: 5, mode: 2))
        #expect(away.current(chapter: 5) == nil && away.report(chapter: 5, mode: 2) != nil)
    }

    @Test func theShelfKeepsTwentyFourChaptersNewestFirst() {
        var shelf = ReportShelf()
        for chapter in 1...30 {
            shelf.insert(report(chapter: chapter, mode: 0))
            shelf.insert(report(chapter: chapter, mode: 1))
        }
        #expect(shelf.chapters == Array((7...30).reversed()) && shelf.chapters.count == ReportShelf.capacity)
        #expect(shelf.entries.count == 2 * ReportShelf.capacity, "a chapter leaves with all its modes")
        #expect(shelf.newestFirst.map(\.chapter) == shelf.chapters)
        let kept6 = shelf.insert(report(chapter: 2))
        #expect(!kept6, "a chapter older than the shelf is not kept")
        #expect(shelf.chapters.count == ReportShelf.capacity)
    }

    // MARK: The plan log

    @Test func aRequestHasOneCardThatTheNewestRecordUpdatesInPlace() {
        var log = PlanLog()
        log.insert(plan(3, .preview))
        log.insert(plan(4, .preview))
        log.insert(plan(3, .queued))
        log.insert(plan(3, .applied))
        #expect(log.records.count == 4 && log.newest?.plan.status == .applied)
        #expect(log.cards.map(\.plan.id) == [3, 4], "newest request first")
        #expect(log.card(id: 3)?.plan.status == .applied && log.card(id: 4)?.plan.status == .preview)
        log.insert(plan(4, .refused))
        #expect(log.cards.map(\.plan.id) == [4, 3] && log.card(id: 4)?.plan.refused == "busy")
        #expect(log.card(id: 8) == nil)
    }

    @Test func theLogKeepsTheLastFiftyRecords() {
        var log = PlanLog()
        for i in 0..<60 { log.insert(plan(i, .queued, step: UInt64(i))) }
        #expect(log.records.count == PlanLog.capacity && log.records.first?.plan.id == 10 && log.records.last?.plan.id == 59)
        #expect(log.card(id: 3) == nil && log.card(id: 10)?.plan.id == 10)
        #expect(log.cards.count == PlanLog.capacity)
    }
}
