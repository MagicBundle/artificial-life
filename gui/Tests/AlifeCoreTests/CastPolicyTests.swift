import Foundation
import Testing

@testable import AlifeCore

/// The Cast tab's policy: every live storm, the eight largest kin groups.
@Suite struct CastPolicyTests {
    private func kin(slot: Int, size: Int, name: String? = nil, namedAfter: Int? = nil) -> IdentityTables.Kin {
        IdentityTables.Kin(slot: slot, id: 100 + slot, name: name ?? "Group \(slot)", namedAfter: namedAfter, size: size, clade: 0,
                           cladeShare: 1, simSelf: 1, x: Double(slot), y: Double(slot) + 0.5, z: nil, arch: nil, births: nil, deaths: nil)
    }

    private func identities(kin: [IdentityTables.Kin], houses: [IdentityTables.House] = [], alive: Int) -> PlayIdentities {
        PlayIdentities(
            head: RecordHead(step: 100, seq: 0, version: 1, flags: 0), snapshotStep: 96,
            tables: IdentityTables(houses: houses, exact: true, origin: 0, kin: kin, colonies: [], unknownImmigrant: 0),
            rows: (0..<alive).map { IdentityRow(uid: Int64($0), clade: 0, house: 0, kin: 0, colony: 255, flags: 0) })
    }

    @Test func theCastHoldsTheEightLargestLineagesLargestFirst() {
        // Twelve groups with distinct sizes, in the order the table lists them (slot order).
        let sizes = [5, 40, 12, 33, 7, 21, 2, 19, 26, 3, 15, 9]
        let record = identities(kin: sizes.enumerated().map { kin(slot: $0.offset, size: $0.element) }, alive: 300)
        let cast = CastPolicy.lineages(record, territory: nil)
        #expect(CastPolicy.maxLineages == 8 && cast.count == 8)
        #expect(cast.map(\.size) == [40, 33, 26, 21, 19, 15, 12, 9])
        #expect(cast.map(\.id) == [101, 103, 108, 105, 107, 110, 102, 111])
    }

    @Test func equalSizesAreOrderedBySlot() {
        let record = identities(kin: [kin(slot: 3, size: 10), kin(slot: 1, size: 10), kin(slot: 2, size: 10)], alive: 40)
        #expect(CastPolicy.lineages(record, territory: nil).map(\.slot) == [1, 2, 3])
    }

    @Test func aShareIsOverTheOrganismsTheSnapshotLists() {
        let record = identities(kin: [kin(slot: 0, size: 30)], alive: 120)
        #expect(CastPolicy.lineages(record, territory: nil).first?.share == 0.25)
        let empty = identities(kin: [kin(slot: 0, size: 30)], alive: 0)
        #expect(CastPolicy.lineages(empty, territory: nil).first?.share == 30, "no division by zero (the table outlives its rows)")
    }

    @Test func aGroupIsNamedAfterItsHouseWhenTheHouseIsListed() {
        let house = IdentityTables.House(slot: 0, id: 777, name: "Aldric", size: 9, clade: 1)
        let record = identities(kin: [kin(slot: 0, size: 9, namedAfter: 777), kin(slot: 1, size: 5, namedAfter: 999), kin(slot: 2, size: 4)],
                                houses: [house], alive: 18)
        #expect(CastPolicy.lineages(record, territory: nil).map(\.namedAfter) == ["Aldric", nil, nil])
    }

    @Test func theTerritoryCellsAreCountedBySlot() throws {
        let store = try WorldFixtures.store()
        let territory = try #require(store.territory)
        let cast = CastPolicy.lineages(store.identities, territory: territory)
        for lineage in cast {
            let owned = territory.owner.filter { Int($0) == lineage.slot }.count
            #expect(lineage.territoryCells == owned, "\(lineage.name)")
        }
        #expect(cast.map(\.territoryCells).reduce(0, +) == territory.owner.filter { $0 != Slot.none }.count)
        #expect(cast.allSatisfy { $0.territoryCells > 0 })
    }

    @Test func theFixturesCastIsWhatItsRecordsSay() throws {
        let store = try WorldFixtures.store()
        let cast = CastPolicy.lineages(store.identities, territory: store.territory)
        #expect(cast.map(\.name) == ["Aldric kin", "Brennar kin", "Cestria"])
        #expect(abs(cast[0].share - 3.0 / 7) < 1e-12 && cast[0].births == 2 && cast[0].deaths == [1, 0, 0, 0, 0])
        #expect(CastPolicy.lineages(nil, territory: nil).isEmpty)
    }

    @Test func deathsAreSummarisedByCause() {
        #expect(CastPolicy.deathSummary([3, 0, 1, 0, 0]) == "3 starved, 1 aged")
        #expect(CastPolicy.deathSummary([0, 0, 0, 0, 2]) == "2 storm")
        #expect(CastPolicy.deathSummary([0, 0, 0, 0]) == "none")
        #expect(CastPolicy.deathSummary(nil) == nil)
        #expect(CastPolicy.deathCauses == ["starved", "killed", "aged", "catastrophe", "storm"])
    }

    // MARK: Storms

    @Test func everyLiveStormIsInTheCastNamedFirst() throws {
        let store = try WorldFixtures.store()
        let cast = CastPolicy.storms(store.storms, baseRate: 14.4)
        #expect(cast.map(\.id) == [7, 9] && cast.map(\.title) == ["Merrow", "disturbance 9"])
        #expect(cast.map(\.named) == [true, false] && cast.map(\.hemisphere) == ["northern", "southern"])
        #expect(cast[1].coasting && !cast[0].coasting)
        #expect(CastPolicy.storms(nil, baseRate: 14.4).isEmpty)
    }

    @Test func aStrongerDisturbanceStaysBelowANamedStorm() throws {
        var store = try WorldFixtures.store()
        var record = try #require(store.storms)
        record.storms.storms[1].zeta = 0.45  // a disturbance just below the naming threshold
        record.storms.storms[0].zeta = 0.5
        store.apply(.strm(record))
        #expect(CastPolicy.storms(store.storms, baseRate: 14.4).map(\.id) == [7, 9])
        record.storms.storms[1].named = true
        record.storms.storms[1].zeta = 0.9
        store.apply(.strm(record))
        #expect(CastPolicy.storms(store.storms, baseRate: 14.4).map(\.id) == [9, 7], "named, then strongest first")
    }

    @Test func aStormsAgeIsGameMinutesWhateverTheCommandedRate() throws {
        // 1250 steps at the base rate of 14.4 per second is 1250 / 864 minutes; the commanded rate is not an input.
        let store = try WorldFixtures.store()
        let age = CastPolicy.storms(store.storms, baseRate: 14.4)[0].ageMinutes
        #expect(abs(age - 1250.0 / 864) < 1e-12)
        #expect(CastPolicy.ageMinutes(steps: 864, baseRate: 14.4) == 1)
        #expect(CastPolicy.ageMinutes(steps: 100, baseRate: 0) == 0)
    }

    @Test func theBarIsFullAtTheScaleAndNeverBeyond() {
        var store = WorldStore()
        let storm = PlayStorms.Storm(id: 1, name: "A", named: true, hemi: 0, x: 0, y: 0, r: 1, ext: [1, 1], zeta: 4, zetaNow: 4, wind: 0, rain: 0,
                                    age: 10, parent: 0, state: "active", track: [], pred: [])
        store.apply(.strm(PlayStormRecord(head: RecordHead(step: 0, seq: 0, version: 1, flags: 0), tick: 25,
                                          storms: PlayStorms(detectStep: 0, storms: [storm], ended: []))))
        #expect(CastPolicy.storms(store.storms, baseRate: 14.4)[0].peakFraction == 1)
        #expect(CastPolicy.fullBar == 1.5)
    }
}
