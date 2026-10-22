import Foundation

/// The palettes of the play stream (DESIGN §33.2.2, normative): the server paints `FRMA.rgb` with them and the client draws
/// the cast swatches and the territory overlay with the same table, so a swatch is the colour of its organisms.
/// `tests/test_gui_sync.py` compares these tables with `game/colors.py`. Colours are `0xRRGGBB`.
package enum WorldPalette {
    /// Slot `s < 16` of a house, kin group or colony table (none is red: red is the hurt colour).
    package static let slots: [UInt32] = [
        0xE69F00, 0x56B4E9, 0x009E73, 0xF0E442, 0x0072B2, 0xCC79A7, 0x9467BD, 0x8C564B,
        0x7FC97F, 0xFDC086, 0x386CB0, 0xBEAED4, 0x17BECF, 0xBCBD22, 0xFF7FBF, 0xA6CEE3,
    ]
    /// An unknown immigrant lineage (slot 253).
    package static let unknown: UInt32 = 0x6E6E96
    /// Alive but listed in no top-16 table (slot 254).
    package static let other: UInt32 = 0x808080
    /// No slot (255) and unlabelled rows.
    package static let none: UInt32 = 0x383838
    /// By `rules.Archetype` value; index 0 is unlabelled.
    package static let archetypes: [UInt32] = [
        none, 0x009E73, 0xD55E00, 0xF0E442, 0x56B4E9, 0xCC79A7, 0x9467BD, 0xC8C8C8, 0x004C8C, 0x00D2DC, 0x8B5A2B,
    ]
    /// By `rules.Intent` value; index 0 is none.
    package static let intents: [UInt32] = [
        none, 0xFF7FBF, 0xD55E00, 0xF0E442, 0x0072B2, 0x7FC97F, 0x9467BD, 0x17BECF, 0x009E73, 0xE69F00, 0x606060,
    ]

    /// The colour of a slot byte: the table for 0 to 15, the three reserved values, nil for the rest (a decoder refuses
    /// those before they get here).
    package static func slotColor(_ slot: UInt8) -> UInt32? {
        switch slot {
        case 0..<16: return slots[Int(slot)]
        case Slot.unknown: return unknown
        case Slot.other: return other
        case Slot.none: return none
        default: return nil
        }
    }

    package static func components(_ rgb: UInt32) -> (r: UInt8, g: UInt8, b: UInt8) {
        (UInt8((rgb >> 16) & 255), UInt8((rgb >> 8) & 255), UInt8(rgb & 255))
    }
}

/// The territory texture: `TERR`'s owner map as RGBA8, `ny` rows of `nx` cells, row-major (DESIGN §33.1.5). A cell with an
/// owner has the owner's slot colour and an alpha from its purity (half to full: a contested cell is fainter); a cell
/// without one is clear. The shader multiplies the alpha by the level's territory alpha (25%).
package enum TerritoryImage {
    package static func rgba(_ territory: PlayTerritory) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: territory.nx * territory.ny * 4)
        for cell in 0..<(territory.nx * territory.ny) {
            let owner = territory.owner[cell]
            guard owner != Slot.none, Int(owner) < territory.ids.count, let color = WorldPalette.slotColor(owner) else { continue }
            let (r, g, b) = WorldPalette.components(color)
            let purity = Float(territory.purity[cell]) / 255
            bytes[4 * cell] = r
            bytes[4 * cell + 1] = g
            bytes[4 * cell + 2] = b
            bytes[4 * cell + 3] = UInt8((255 * (0.5 + 0.5 * purity)).rounded())
        }
        return bytes
    }
}
