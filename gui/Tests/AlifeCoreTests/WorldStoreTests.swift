import Foundation
import Testing

@testable import AlifeCore

/// The reducers of the World window's store, on the committed play stream (`play_tiny3d`) and on events built by hand.
@Suite struct WorldStoreTests {
    private func decoded(_ kind: String = "3d") throws -> (header: StreamHeader, records: [StreamRecord]) {
        let events = try PlayFixture(kind: kind).events()
        guard case .header(let header)? = events.first else { throw StreamError.badMagic }
        return (header, events.compactMap { if case .record(let r) = $0 { r } else { nil } })
    }

    private func event(_ id: UInt64, step: Int = 0, imp: Int = 1, text: String = "") -> WorldEvent {
        WorldEvent(id: id, step: step, kind: 1, fam: "kin", a: 0, b: 0, x: nil, y: nil, z: nil, imp: imp,
                   text: text.isEmpty ? "event \(id)" : text)
    }

    private func batch(_ events: [WorldEvent], backlog: Bool = false, step: UInt64 = 0) -> PlayEventBatch {
        PlayEventBatch(head: RecordHead(step: step, seq: 0, version: 1, flags: backlog ? 1 : 0), events: events)
    }

    // MARK: The stream's records

    @Test func everyRecordOfTheFixtureLandsInItsSlot() throws {
        let (header, records) = try decoded()
        var store = WorldStore()
        store.reset(header: header)
        for record in records { store.apply(record) }
        #expect(store.header == header && store.play == header.play)
        #expect(store.frame?.step == 104, "the newest FRMA wins")
        #expect(store.identities != nil && store.labels != nil && store.territory != nil)
        #expect(store.storms != nil && store.map != nil && store.environment != nil)
        #expect(store.clock?.head.step == 104, "the newest CLCK wins")
        #expect(store.focus != nil)
        #expect(!store.chronicle.events.isEmpty)
        #expect(store.frameSerial == 2 && store.mapSerial == 1 && store.territorySerial == 1)
    }

    @Test func aLatestWinsRecordReplacesTheOneBeforeAndAFrameIsNoRevision() throws {
        let (header, records) = try decoded()
        var store = WorldStore()
        store.reset(header: header)
        let frames = records.compactMap { if case .frma(let f) = $0 { f } else { nil } }
        #expect(frames.map(\.step) == [100, 104])
        let before = store.revision
        let c0 = store.apply(.frma(frames[0]))
        #expect(c0 == .frame && store.revision == before, "the renderer reads frames by their serial")
        store.apply(.frma(frames[1]))
        #expect(store.frame?.step == 104 && store.replaced["FRMA"] == 1)
        let clocks = records.compactMap { if case .clck(let c) = $0 { c } else { nil } }
        let c1 = store.apply(.clck(clocks[0]))
        #expect(c1 == .clock && store.revision == before + 1)
    }

    @Test func aNewStreamForgetsTheOldOne() throws {
        let (header, records) = try decoded()
        var store = WorldStore()
        store.reset(header: header)
        for record in records { store.apply(record) }
        let (other, _) = try decoded("2d")
        store.reset(header: other)
        #expect(store.header == other)
        #expect(store.frame == nil && store.identities == nil && store.map == nil && store.clock == nil)
        #expect(store.chronicle.events.isEmpty && store.frameSerial == 0)
    }

    // MARK: Events

    @Test func eventsAreKeptOnceAndInOrderOfTheirIds() {
        var store = WorldStore()
        let fresh = store.apply(events: batch([event(5000), event(3000), event(4000)]))
        #expect(fresh.map(\.id) == [5000, 3000, 4000], "the new ones, in the order they arrived")
        #expect(store.chronicle.events.map(\.id) == [3000, 4000, 5000])
        #expect(store.chronicle.newestFirst.map(\.id) == [5000, 4000, 3000])
        // The same events again (a backlog that overlaps the live feed, a resume): nothing is new.
        #expect(store.apply(events: batch([event(4000), event(5000)])).isEmpty)
        #expect(store.chronicle.events.count == 3)
        // One new among known ones.
        #expect(store.apply(events: batch([event(5000), event(6000), event(3000)])).map(\.id) == [6000])
    }

    @Test func aBacklogFillsInBehindTheLiveEventsInOrder() {
        var store = WorldStore()
        store.apply(events: batch([event(9000)]))
        let fresh = store.apply(events: batch([event(2000), event(4000), event(9000)], backlog: true))
        #expect(fresh.map(\.id) == [2000, 4000])
        #expect(store.chronicle.events.map(\.id) == [2000, 4000, 9000])
    }

    @Test func theChronicleKeepsTheNewestFiveHundred() {
        var store = WorldStore()
        store.apply(events: batch((1...520).map { event(UInt64($0) * 1000) }))
        #expect(store.chronicle.events.count == Chronicle.capacity)
        #expect(store.chronicle.events.first?.id == 21_000 && store.chronicle.events.last?.id == 520_000)
        // An event older than the ring is not new, however often it is sent; a newer one is.
        #expect(store.apply(events: batch([event(5000)])).isEmpty)
        #expect(store.apply(events: batch([event(521_000)])).map(\.id) == [521_000])
        #expect(store.chronicle.events.count == Chronicle.capacity)
    }

    @Test func anEventBatchThroughApplyReportsTheChange() {
        var store = WorldStore()
        let first = store.apply(.evnt(batch([event(1000)])))
        #expect(first == .events)
        let again = store.apply(.evnt(batch([event(1000)])))
        #expect(again.isEmpty)
    }

    @Test func theFixturesEventsAreTheOnesItsSidecarSays() throws {
        let (header, records) = try decoded()
        var store = WorldStore()
        store.reset(header: header)
        let batches = records.compactMap { if case .evnt(let b) = $0 { b } else { nil } }
        #expect(batches.count == 2 && batches[0].isBacklog && !batches[1].isBacklog)
        for b in batches { store.apply(events: b) }
        #expect(store.chronicle.events.count == Set(batches.flatMap { $0.events.map(\.id) }).count)
        let ids = store.chronicle.events.map(\.id)
        #expect(ids == ids.sorted())
    }

    // MARK: Lookups

    @Test func aUidIsFoundInTheNewestFrameFromAHint() throws {
        let (header, records) = try decoded()
        var store = WorldStore()
        store.reset(header: header)
        for record in records { store.apply(record) }
        let frame = try #require(store.frame)
        for (index, uid) in frame.uid.enumerated() {
            #expect(store.listedIndex(ofUID: uid) == index)
            #expect(store.listedIndex(ofUID: uid, hint: frame.n - 1) == index)
            #expect(store.listedIndex(ofUID: uid, hint: 10_000) == index, "a hint past the end is clamped")
        }
        #expect(store.listedIndex(ofUID: -12345) == nil)
        #expect(WorldStore().listedIndex(ofUID: 1) == nil)
    }

    @Test func anIdentityIsFoundByUid() throws {
        let (header, records) = try decoded()
        var store = WorldStore()
        store.reset(header: header)
        for record in records { store.apply(record) }
        let row = try #require(store.identities?.rows.first)
        #expect(store.identity(ofUID: row.uid) == row)
        #expect(store.identity(ofUID: -1) == nil)
    }
}
