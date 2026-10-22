import Foundation
import Testing

@testable import AlifeCore

@Suite struct AlertStackTests {
    private func alert(_ id: UInt64, imp: Int = 2, cls: String = "storm_named", surface: PlayAlert.Surface = .alert,
                       state: PlayAlert.State = .raised, ref: PlayAlert.Ref? = nil, x: Double? = nil, y: Double? = nil,
                       expires: Int? = nil, pause: Bool = false, watched: Bool = false, step: Int? = nil) -> PlayAlert {
        PlayAlert(id: id, version: 1, step: step ?? Int(id / 1000), alertClass: cls, key: "k\(id)", state: state, surface: surface,
                  imp: imp, title: "title \(id)", text: "text \(id)", x: x, y: y, z: nil, ref: ref, pause: pause, expires: expires,
                  watched: watched, calibrated: true)
    }

    private func board(_ alerts: [PlayAlert]) -> AlertBoard {
        var board = AlertBoard()
        board.insert(alerts)
        return board
    }

    // MARK: The layout

    @Test func theStackShowsTheMostImportantFirstAndTheNewestAmongEquals() {
        let b = board([alert(1_000_600, imp: 1), alert(2_000_600, imp: 3, pause: true), alert(3_000_600, imp: 2), alert(4_000_600, imp: 2)])
        let layout = AlertStack.layout(b, step: 5000)
        #expect(layout.rows.map(\.id) == [2_000_600, 4_000_600, 3_000_600] && layout.hidden == 1)
        #expect(layout.rows.map(\.tone) == [.urgent, .important, .important])
    }

    @Test func aPausingAlertIsNeverTheOneHiddenUnderTheCount() {
        // Ten later, lesser alerts: the importance-3 one stays in view (DESIGN §34.7).
        let b = board([alert(1_000_600, imp: 3, pause: true)] + (2...11).map { alert(UInt64($0) * 1000 + 600, imp: 1) })
        for rows in [1, 2, 3] {
            let layout = AlertStack.layout(b, step: 20_000, maxRows: rows)
            #expect(layout.rows.first?.id == 1_000_600 && layout.rows.count == rows && layout.hidden == 11 - rows)
        }
        #expect(AlertStack.layout(b, step: 20_000, maxRows: 0).rows.isEmpty)
    }

    @Test func onlyTheStackSurfaceIsOnTheStackAndAnAcknowledgedOrClearedAlertLeaves() {
        var b = board([alert(1_000_600), alert(2_000_600, surface: .log), alert(3_000_600), alert(4_000_600, state: .cleared)])
        #expect(AlertStack.layout(b, step: 5000).rows.map(\.id) == [3_000_600, 1_000_600])
        b.acknowledge(id: 3_000_600)
        #expect(AlertStack.layout(b, step: 5000).rows.map(\.id) == [1_000_600])
    }

    @Test func anExpiredAlertLeavesAndOneWithNoExpiryStays() {
        let b = board([alert(1_000_600, expires: 3000), alert(2_000_600, imp: 3, expires: nil)])
        #expect(AlertStack.layout(b, step: 2999).rows.count == 2)
        #expect(AlertStack.layout(b, step: 3000).rows.map(\.id) == [2_000_600], "expires is the first step it is gone")
        #expect(AlertStack.layout(b, step: 1_000_000).rows.map(\.id) == [2_000_600])
    }

    @Test func aRowKnowsWhereItGoesAndWhatItWatches() {
        let storm = alert(1_000_600, ref: .init(kind: .storm, id: 7), x: 4.5, y: 20.25)
        let kin = alert(2_000_600, ref: .init(kind: .kin, id: 17))
        let power = alert(3_000_600, ref: .init(kind: .power, id: 3))
        let world = alert(4_000_600, ref: .init(kind: .world, id: 0))
        let rows = [storm, kin, power, world].map { AlertStack.row(for: $0) }
        #expect(rows[0].action == ChronicleAction(at: SIMD2(4.5, 20.25), select: .storm(id: 7)) && rows[0].kin == nil)
        #expect(rows[1].action == ChronicleAction(at: nil, select: .kin(id: 17)) && rows[1].kin == 17)
        #expect(rows[2].action == nil && rows[3].action == nil, "a power or the world has no place to go to")
        // A place alone (no entity) still flies.
        #expect(AlertStack.row(for: alert(5_000_600, x: 1, y: 2)).action == ChronicleAction(at: SIMD2(1, 2), select: .none))
    }

    @Test func theWatchStarIsLitByTheWorldOrByTheWatchSet() {
        let kin = alert(2_000_600, ref: .init(kind: .kin, id: 17))
        #expect(!AlertStack.row(for: kin).watched)
        var watch = WatchSet()
        _ = watch.toggle(17)
        #expect(AlertStack.row(for: kin, watch: watch).watched)
        #expect(AlertStack.row(for: alert(3_000_600, watched: true)).watched, "the world says its alert is about a watched thread")
        #expect(!AlertStack.row(for: alert(3_000_600, ref: .init(kind: .storm, id: 17)), watch: watch).watched)
    }

    @Test func aHugeRefIdDoesNotTrap() {
        let row = AlertStack.row(for: alert(1_000_600, ref: .init(kind: .kin, id: Int64.max)))
        #expect(row.kin == Int(Int64.max))  // (Int is 64 bits here; the guard is for 32-bit hosts and negative ids)
        #expect(AlertStack.row(for: alert(2_000_600, ref: .init(kind: .kin, id: -1))).kin == nil)
        #expect(AlertStack.row(for: alert(2_000_600, ref: .init(kind: .storm, id: -1))).action == nil)
    }

    // MARK: The watch set

    @Test func theWatchSetHoldsThreeAndTogglesByEveryLine() {
        var watch = WatchSet()
        let added = [watch.toggle(1), watch.toggle(2), watch.toggle(3)]
        #expect(added == [.watch(1), .watch(2), .watch(3)])
        #expect(watch.isFull && watch.ids == [1, 2, 3])
        let refused = watch.toggle(4)
        #expect(refused == nil, "full: nothing is sent")
        let dropped = watch.toggle(2)
        #expect(dropped == .unwatch(2) && watch.ids == [1, 3] && !watch.isFull)
        let again = watch.toggle(4)
        #expect(again == .watch(4))
        let cleared = watch.clear()
        #expect(cleared == .watch(nil) && watch.ids.isEmpty)
        let nothing = watch.clear()
        #expect(nothing == nil, "nothing was watched")
        let negative = watch.toggle(-1)
        #expect(negative == nil, "a kin id is not negative")
    }

    // MARK: The board's views of itself

    @Test func thePausingAlertIsTheNewestOneOnTheStack() {
        var b = board([alert(1_000_600, imp: 3, pause: true), alert(2_000_600, pause: false), alert(3_000_600, imp: 3, pause: true)])
        let newest = b.pausing(atStep: 4000)?.id
        #expect(newest == 3_000_600)
        b.acknowledge(id: 3_000_600)
        let older = b.pausing(atStep: 4000)?.id
        #expect(older == 1_000_600)
        b.acknowledge(id: 1_000_600)
        let none = b.pausing(atStep: 4000)
        #expect(none == nil)
    }

    @Test func theAlertsOfAChapterAreTheStackSurfaceOnesInItsStepsExceptItsOwn() {
        let b = board([
            alert(5_000_600, imp: 1, step: 5000), alert(6_001_600, imp: 1, step: 6001), alert(7_000_600, imp: 3, step: 7000),
            alert(8_000_600, imp: 2, cls: "chapter", step: 8000), alert(9_000_600, surface: .log, step: 9000),
            alert(10_000_600, state: .cleared, step: 10_000), alert(12_000_600, imp: 2, step: 12_000), alert(12_001_600, step: 12_001),
        ])
        #expect(b.raised(in: 6001..<12_001).map(\.id) == [7_000_600, 12_000_600, 6_001_600])
        #expect(b.raised(in: 6001..<12_001, excluding: []).map(\.id).contains(8_000_600))
    }

    @Test func theFixtureStreamStacksAsTheRecordsSay() throws {
        let store = try WorldFixtures.store("3d_g3b")
        let layout = AlertStack.layout(store.alerts, step: 12_000)
        // the chapter alert (3), the storm (2), the surge (1); the cleared fall and the two log rows are not on it
        #expect(layout.rows.map(\.id) == [12_000_600, 12_000_601, 11_500_603] && layout.hidden == 0)
        #expect(layout.rows[0].pauses && layout.rows[1].action?.select == .storm(id: 7))
    }
}
