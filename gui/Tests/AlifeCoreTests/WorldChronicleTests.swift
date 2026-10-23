import Foundation
import Testing

@testable import AlifeCore

/// Grouping, filters and the toast rule of the chronicle tab (DESIGN §33.4.8).
@Suite struct ChronicleReducerTests {
    private func event(_ id: UInt64, step: Int, fam: String = "kin", imp: Int = 1) -> WorldEvent {
        WorldEvent(id: id, step: step, kind: 1, fam: fam, a: 0, b: 0, x: nil, y: nil, z: nil, imp: imp, text: "e\(id)")
    }

    private func chronicle(_ events: [WorldEvent]) -> Chronicle {
        var c = Chronicle()
        _ = c.insert(events)
        return c
    }

    @Test func aMinuteOfGameTimeIs864StepsAtTheBaseRate() {
        #expect(ChronicleReducer.stepsPerMinute(baseRate: 14.4) == 864)
        #expect(ChronicleReducer.stepsPerMinute(baseRate: 30) == 1800)
        #expect(ChronicleReducer.stepsPerMinute(baseRate: 0) == 1, "never a division by zero")
        #expect(ChronicleReducer.minute(ofStep: 863, baseRate: 14.4) == 0)
        #expect(ChronicleReducer.minute(ofStep: 864, baseRate: 14.4) == 1)
        #expect(ChronicleReducer.minute(ofStep: 10_000, baseRate: 14.4) == 11)
    }

    @Test func eventsAreGroupedByMinuteNewestFirst() {
        let events = [
            event(1_000, step: 1), event(500_000, step: 500), event(863_000, step: 863),
            event(864_000, step: 864), event(1_800_000, step: 1_800),
        ]
        let groups = ChronicleReducer.groups(chronicle(events), baseRate: 14.4)
        #expect(groups.map(\.minute) == [2, 1, 0])
        #expect(groups.map { $0.events.map(\.step) } == [[1_800], [864], [863, 500, 1]])
        #expect(groups[0].steps == 1728..<2592 && groups[2].steps == 0..<864)
        #expect(groups[0].title == "Minute 2" && groups[2].subtitle == "steps 0 to 863")
    }

    @Test func aGroupHeaderWritesItsStepsGroupedLikeEveryOtherNumber() {
        // minute 13 of 864 steps: "steps 11232 to 12095" read as a different number than the rows below it (third review, stage 8)
        let groups = ChronicleReducer.groups(chronicle([event(11_300_000, step: 11_300)]), baseRate: 14.4)
        #expect(groups.map(\.title) == ["Minute 13"] && groups[0].subtitle == "steps 11,232 to 12,095")
        #expect(Numbers.stepLabel(468_000) == "step 468,000" && Numbers.stepLabel(0) == "step 0")
    }

    @Test func theGroupsStayWholeWhenTheRateIsChanged() {
        // Game time is steps over 60·B: the commanded rate is not in it, so 0.25x and 2x group the same events alike.
        let events = (0..<20).map { event(UInt64($0 + 1) * 1000, step: ($0 + 1) * 100) }
        let a = ChronicleReducer.groups(chronicle(events), baseRate: 14.4)
        #expect(a.flatMap(\.events).count == 20 && a.map(\.minute) == a.map(\.minute).sorted(by: >))
    }

    @Test func aFilterKeepsItsFamiliesAndEmptyMinutesDisappear() {
        let events = [
            event(1_000, step: 10, fam: "storm"), event(2_000, step: 20, fam: "kin"), event(900_000, step: 900, fam: "house"),
            event(1_800_000, step: 1_800, fam: "director"), event(1_801_000, step: 1_801, fam: "system"),
            event(1_802_000, step: 1_802, fam: "focus"),
        ]
        let c = chronicle(events)
        #expect(ChronicleReducer.groups(c, filter: .storms, baseRate: 14.4).flatMap(\.events).map(\.fam) == ["storm"])
        let lineages = ChronicleReducer.groups(c, filter: [.lineages, .houses], baseRate: 14.4)
        #expect(lineages.map(\.minute) == [1, 0] && lineages.flatMap(\.events).map(\.fam) == ["house", "kin"])
        // The focus's own events belong to "System", so they are not lost to a filter of their own.
        let system = ChronicleReducer.groups(c, filter: .system, baseRate: 14.4).flatMap(\.events)
        #expect(system.map(\.fam) == ["focus", "system"])
        #expect(ChronicleReducer.groups(c, filter: [], baseRate: 14.4).isEmpty)
        #expect(ChronicleReducer.groups(c, filter: .all, baseRate: 14.4).flatMap(\.events).count == 6)
    }

    @Test func everyFamilyHasAFilterAndAnUnknownOneIsNeverHidden() {
        for (fam, filter) in [("storm", ChronicleFilter.storms), ("kin", .lineages), ("house", .houses), ("director", .director),
                              ("system", .system), ("focus", .system), ("something new", .system)] {
            #expect(ChronicleFilter.of(family: fam) == filter, "\(fam)")
        }
        #expect(ChronicleFilter.choices.reduce(into: ChronicleFilter()) { $0.formUnion($1.filter) } == ChronicleFilter.all)
    }

    @Test func onlyImportanceTwoRaisesAToastAndNotTheBacklog() {
        let fresh = [event(1000, step: 1, imp: 0), event(2000, step: 2, imp: 1), event(3000, step: 3, imp: 2), event(4000, step: 4, imp: 2)]
        #expect(ChronicleReducer.toasts(from: fresh, backlog: false).map(\.id) == [3000, 4000])
        #expect(ChronicleReducer.toasts(from: fresh, backlog: true).isEmpty, "a re-opened world's history is not news")
        #expect(ChronicleReducer.toasts(from: [], backlog: false).isEmpty)
    }

    @Test func theFixturesChronicleGroupsAndHasItsToast() throws {
        let store = try WorldFixtures.store()
        let groups = ChronicleReducer.groups(store.chronicle, baseRate: store.play?.baseRate ?? 14.4)
        #expect(groups.count == 1 && groups[0].events.map(\.id) == [100_000, 99_000, 96_000], "steps 96 to 100 are in minute 0")
        #expect(ChronicleReducer.toasts(from: store.chronicle.events, backlog: false).map(\.id) == [99_000])
    }
}

@Suite struct ToastQueueTests {
    private func toast(_ id: UInt64) -> WorldEvent {
        WorldEvent(id: id, step: 1, kind: 11, fam: "storm", a: 1, b: 0, x: nil, y: nil, z: nil, imp: 2, text: "t\(id)")
    }

    @Test func aToastLastsThreeSeconds() {
        var queue = ToastQueue()
        queue.push([toast(1)], at: 10)
        #expect(queue.toasts.map(\.id) == [1] && queue.nextExpiry == 13)
        let early = queue.expire(at: 12.99)
        #expect(!early && queue.toasts.count == 1)
        let dropped = queue.expire(at: 13)
        #expect(dropped && queue.toasts.isEmpty && queue.nextExpiry == nil)
        let again = queue.expire(at: 14)
        #expect(!again, "nothing left to drop")
    }

    @Test func eachToastHasItsOwnClock() {
        var queue = ToastQueue()
        queue.push([toast(1)], at: 0)
        queue.push([toast(2)], at: 2)
        _ = queue.expire(at: 3.5)
        #expect(queue.toasts.map(\.id) == [2] && queue.nextExpiry == 5)
    }

    @Test func aBurstKeepsTheNewestThree() {
        var queue = ToastQueue()
        queue.push((1...10).map(toast), at: 0)
        #expect(queue.toasts.map(\.id) == [8, 9, 10] && ToastQueue.capacity == 3)
    }

    @Test func anEventIsShownOnceHoweverOftenItIsPushed() {
        var queue = ToastQueue()
        queue.push([toast(1)], at: 0)
        _ = queue.expire(at: 5)
        queue.push([toast(1)], at: 6)
        #expect(queue.toasts.isEmpty, "it was shown")
        queue.push([toast(2), toast(2)], at: 6)
        #expect(queue.toasts.map(\.id) == [2])
    }

    @Test func aToastCanBeDismissedOrCleared() {
        var queue = ToastQueue()
        queue.push([toast(1), toast(2)], at: 0)
        queue.dismiss(id: 1)
        #expect(queue.toasts.map(\.id) == [2])
        queue.clear()
        #expect(queue.toasts.isEmpty)
    }

    @Test func theMemoryOfShownToastsIsBounded() {
        var queue = ToastQueue()
        for i in 1...3000 {
            queue.push([toast(UInt64(i))], at: Double(i))
            _ = queue.expire(at: Double(i) + 10)
        }
        queue.push([toast(3000)], at: 5000)
        #expect(queue.toasts.count <= 1)
    }
}

/// The toasts of alerts and the ticker line (DESIGN §34.7).
@Suite struct AlertToastAndTickerTests {
    private func alert(_ id: UInt64, title: String = "Storm Merrow", text: String = "Storm Merrow reached named strength.",
                       surface: PlayAlert.Surface = .alert, state: PlayAlert.State = .raised) -> PlayAlert {
        PlayAlert(id: id, version: 1, step: Int(id / 1000), alertClass: "storm_named", key: "S7", state: state, surface: surface, imp: 2,
                  title: title, text: text, x: 4.5, y: 20.25, z: nil, ref: .init(kind: .storm, id: 7), pause: false, expires: nil,
                  watched: false, calibrated: true)
    }

    private func event(_ id: UInt64, imp: Int = 1, fam: String = "storm", kind: Int = 11) -> WorldEvent {
        WorldEvent(id: id, step: Int(id / 1000), kind: kind, fam: fam, a: 7, b: 0, x: 4.5, y: 20.25, z: nil, imp: imp, text: "event \(id)")
    }

    @Test func anAlertToastsWithItsTitleAndTextAndAClickGoesToTheStorm() throws {
        var queue = ToastQueue()
        queue.push(alerts: [alert(12_000_601)], at: 10)
        let toast = try #require(queue.toasts.first)
        #expect(toast.text == "Storm Merrow: Storm Merrow reached named strength." && toast.id == 12_000_601 && toast.expires == 13)
        #expect(toast.action == ChronicleAction(at: SIMD2(4.5, 20.25), select: .storm(id: 7)))
        queue.push(alerts: [alert(12_000_602, title: "", text: "plain")], at: 10)
        #expect(queue.toasts.last?.text == "plain", "no title: the text alone")
    }

    @Test func anAlertAndAnEventShareOneMemoryAndOneCapacity() {
        var queue = ToastQueue()
        queue.push(alerts: [alert(5_000_601)], at: 0)
        queue.push([event(6_000)], at: 0)
        queue.push(alerts: [alert(5_000_601), alert(7_000_601), alert(8_000_601)], at: 0)
        #expect(queue.toasts.map(\.id) == [6_000, 7_000_601, 8_000_601], "the oldest of four went; the repeated alert did not come back")
    }

    @Test func theTickerShowsTheNewestForSeventeenSeconds() {
        var ticker = Ticker()
        #expect(ticker.line == nil && ticker.nextExpiry == nil)
        ticker.show(events: [event(1_000), event(2_000)], at: 100)
        #expect(ticker.line?.text == "event 2000" && ticker.line?.expires == 117 && ticker.nextExpiry == 117 && Ticker.duration == 17)
        ticker.show(events: [event(3_000)], at: 105)
        #expect(ticker.line?.text == "event 3000" && ticker.line?.expires == 122, "the newest replaces it and restarts the time")
        let early = ticker.expire(at: 121.9)
        #expect(!early && ticker.line != nil)
        let gone = ticker.expire(at: 122)
        #expect(gone && ticker.line == nil)
        let again = ticker.expire(at: 130)
        #expect(!again)
    }

    @Test func theTickerTellsWhatIsToldAndNotTheFillerOrTheLog() {
        var ticker = Ticker()
        ticker.show(events: [event(1_000, imp: 0), event(2_000, fam: "system"), event(3_000, fam: "focus", kind: 14)], at: 0)
        #expect(ticker.line == nil, "imp 0, system and focus are the log's")
        ticker.show(events: [event(4_000, imp: 1), event(5_000, imp: 0)], at: 0)
        #expect(ticker.line?.text == "event 4000", "the last event worth a line")
        ticker.show(events: [], at: 1)
        #expect(ticker.line?.text == "event 4000")
    }

    @Test func aRepeatedEventDoesNotRestartTheLineButAReportsLineAlwaysDoes() {
        var ticker = Ticker()
        ticker.show(events: [event(4_000)], at: 0)
        ticker.show(events: [event(4_000)], at: 10)
        #expect(ticker.line?.expires == 17, "the same event again (a backlog) changes nothing")
        ticker.show(id: 0, text: "Storm Merrow reached named strength.", at: 12)
        #expect(ticker.line?.text == "Storm Merrow reached named strength." && ticker.line?.expires == 29)
        ticker.show(id: 0, text: "Another.", at: 13)
        #expect(ticker.line?.text == "Another.")
        ticker.show(id: 0, text: "", at: 14)
        #expect(ticker.line?.text == "Another.", "nothing to say is not a line")
    }

    @Test func aLogSurfaceAlertIsATickerLineWithItsFlyTo() {
        var ticker = Ticker()
        ticker.show(alerts: [alert(1_000_601, surface: .alert)], at: 0)
        #expect(ticker.line == nil, "the stack surface is the stack's")
        ticker.show(alerts: [alert(2_000_601, text: "Aldric has held together.", surface: .log)], at: 0)
        #expect(ticker.line?.text == "Aldric has held together." && ticker.line?.action?.select == .storm(id: 7))
        ticker.show(alerts: [alert(3_000_601, surface: .log, state: .cleared)], at: 1)
        #expect(ticker.line?.id == 2_000_601, "a clearing is not news")
        ticker.clear()
        #expect(ticker.line == nil)
        ticker.show(events: [event(2_000_601)], at: 2)
        #expect(ticker.line != nil, "after a clear the same id may show again")
    }

    @Test func theRawKinEventsAreHiddenInAWorldThatTellsThreads() {
        func kin(_ id: UInt64, kind: Int) -> WorldEvent {
            WorldEvent(id: id, step: Int(id / 1000), kind: kind, fam: "kin", a: 1, b: 0, x: nil, y: nil, z: nil, imp: 1, text: "kin \(kind)")
        }
        var chronicle = Chronicle()
        _ = chronicle.insert([kin(1_000, kind: EventKind.kinSplit), kin(2_000, kind: EventKind.kinFormed), kin(3_000, kind: EventKind.kinMerged),
                              kin(4_000, kind: EventKind.kinEnded)])
        func kinds(hiding: Bool) -> [Int] {
            ChronicleReducer.groups(chronicle, baseRate: 14.4, hidingRawKin: hiding).flatMap(\.events).map(\.kind).sorted()
        }
        #expect(kinds(hiding: false) == [2, 3, 18, 19], "a G3a world has nothing else of its lineages")
        #expect(kinds(hiding: true) == [18, 19])
        #expect(ChronicleReducer.isRawKin(kin(1, kind: 4)) && !ChronicleReducer.isRawKin(kin(1, kind: 18)))
    }
}
