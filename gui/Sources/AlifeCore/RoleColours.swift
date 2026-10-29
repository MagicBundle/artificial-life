import Foundation

/// The `role` colour mode of the bonds feature (DESIGN §40.7, §45): the server computes the colour of every agent (no protocol change),
/// so the client only names it. The legend lists the colours the server paints; an agent hurt in the last step is red in every mode.
package enum RoleColours {
    package struct Entry: Equatable, Sendable {
        package var name: String
        /// `0xRRGGBB`.
        package var rgb: UInt32
        package var meaning: String
    }

    /// Generalist grey, forager yellow, guard blue, germ pink (Okabe-Ito, as the archetype colours of the game).
    package static let entries: [Entry] = [
        Entry(name: "Generalist", rgb: 0xBEBEBE, meaning: "no committed role"),
        Entry(name: "Forager", rgb: 0xF0E442, meaning: "harvests more, defends less"),
        Entry(name: "Guard", rgb: 0x0072B2, meaning: "shields itself and its bonded partners"),
        Entry(name: "Germ", rgb: 0xCC79A7, meaning: "a germ line"),
    ]

    /// The mode is found by name in the header's `color_modes` (its index is whatever the header says; a world without the bonds feature
    /// does not list it).
    package static let mode = "role"

    package static func isRoleMode(_ name: String?) -> Bool { name == mode }
}
