import Foundation
import Testing

@testable import AlifeCore

@Suite struct WorldPaletteTests {
    @Test func theTablesHaveTheSizesOfTheEngine() {
        #expect(WorldPalette.slots.count == 16)
        #expect(WorldPalette.archetypes.count == 8)  // none and the seven archetypes
        #expect(WorldPalette.intents.count == 11)  // none and the ten intents
        #expect(Set(WorldPalette.slots).count == 16, "no two slots look alike")
        #expect(!(WorldPalette.slots + WorldPalette.archetypes + WorldPalette.intents).contains(0xFF0000), "red is the hurt colour")
    }

    @Test func aSlotByteHasItsColourAndTheReservedOnesTheirs() {
        for slot in 0..<16 { #expect(WorldPalette.slotColor(UInt8(slot)) == WorldPalette.slots[slot]) }
        #expect(WorldPalette.slotColor(Slot.unknown) == WorldPalette.unknown)
        #expect(WorldPalette.slotColor(Slot.other) == WorldPalette.other)
        #expect(WorldPalette.slotColor(Slot.none) == WorldPalette.none)
        for slot in 16..<253 { #expect(WorldPalette.slotColor(UInt8(slot)) == nil) }
    }

    @Test func theComponentsAreRedGreenBlue() {
        let c = WorldPalette.components(0xE69F01)
        #expect(c.r == 0xE6 && c.g == 0x9F && c.b == 0x01)
    }

    // MARK: The territory image

    private func territory(owner: [UInt8], purity: [UInt8], ids: [Int] = [17, 21, 22], nx: Int = 2, ny: Int = 2) -> PlayTerritory {
        PlayTerritory(head: RecordHead(step: 0, seq: 0, version: 1, flags: 0), nx: nx, ny: ny, snapshotStep: 0, ids: ids,
                      owner: owner, purity: purity)
    }

    @Test func eachCellHasItsOwnersColourAndAnAlphaFromItsPurity() {
        let image = TerritoryImage.rgba(territory(owner: [0, 1, 255, 2], purity: [255, 0, 255, 128]))
        #expect(image.count == 16)
        func texel(_ i: Int) -> [UInt8] { Array(image[(4 * i)..<(4 * i + 4)]) }
        let orange = WorldPalette.components(WorldPalette.slots[0])
        #expect(texel(0) == [orange.r, orange.g, orange.b, 255], "pure: opaque")
        let sky = WorldPalette.components(WorldPalette.slots[1])
        #expect(texel(1) == [sky.r, sky.g, sky.b, 128], "contested: half")
        #expect(texel(2) == [0, 0, 0, 0], "no owner: clear")
        #expect(texel(3)[3] == 192 && texel(3)[0] == WorldPalette.components(WorldPalette.slots[2]).r)
    }

    @Test func aSlotPastTheIdsIsNoOwner() {
        let image = TerritoryImage.rgba(territory(owner: [3, 0, 0, 0], purity: [255, 255, 255, 255], ids: [5]))
        #expect(Array(image[0..<4]) == [0, 0, 0, 0], "slot 3 has no id in this record")
        #expect(image[3 + 4] == 255)
    }

    @Test func theFixturesTerritoryHasItsOwnersWhereItsSidecarSays() throws {
        let fixture = try PlayFixture(kind: "3d")
        let terr = try #require(try fixture.events().compactMap { event -> PlayTerritory? in
            if case .record(.terr(let t)) = event { t } else { nil }
        }.first)
        let image = TerritoryImage.rgba(terr)
        #expect(image.count == terr.nx * terr.ny * 4)
        // Row 3: cells 0 to 10 are slot 0, 11 to 21 slot 1, the rest nobody.
        func alpha(_ cell: Int, row: Int) -> UInt8 { image[4 * (row * terr.nx + cell) + 3] }
        #expect(alpha(5, row: 3) > 0 && alpha(16, row: 3) > 0 && alpha(28, row: 3) == 0)
        #expect(image[4 * (3 * terr.nx + 5)] == WorldPalette.components(WorldPalette.slots[0]).r)
        #expect(image[4 * (3 * terr.nx + 16)] == WorldPalette.components(WorldPalette.slots[1]).r)
    }
}
