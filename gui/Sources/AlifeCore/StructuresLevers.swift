import Foundation

// MARK: - Built structures

/// The levers of the built-structures layer (`[structures]`, DESIGN §38, §45): material that organisms deposit, which shelters from the
/// wind, slows movement and shields from predation. `structures.enabled` is the switch and fixes the brain layout (a fork cannot
/// change it). Ranges mirror `ext/structures.py` `_validate`.
///
/// `structures.silent_build` is command-line only: it zeroes the founders' build column at the start of a run (the control that
/// nobody builds until mutation finds it), a start-of-run choice that has no meaning for a fork.
package enum StructuresLevers {
    private static let on = FeatureLeverSpec.Gate.flag("structures.enabled")

    package static let group = FeatureLeverGroup(
        feature: "structures",
        title: "Structures",
        summary: "Material that organisms build: shelter from the wind, a drag barrier and a shield against predation. The overlay `build` switches it on.",
        switchKey: "structures.enabled",
        specs: [
            .flag("structures.enabled", "false", "Structures on",
                  "Adds the material grid, the material sensor and the build action. Every lever below is inert while this is off.", layout: true),
            .integer("structures.cap", "4096", "Cell capacity", "Material units a cell can hold, 256 to 32767.", low: 256, high: 32767, gate: on),
            .integer("structures.build_max", "16", "Build rate", "Units deposited per step at build output 1, 1 to 1024.", low: 1, high: 1024, gate: on),
            .integer("structures.build_cost", "2", "Build cost", "Quanta per unit built; 0 = free. build_cost x build_max must stay below 1048576.", low: 0, high: 32768, gate: on),
            .integer("structures.decay_q12", "16", "Decay", "Decay per step in 1/4096; material always decays, so at least 1.", low: 1, high: 4096, gate: on),
            .decimal("structures.shelter_gain", "0.8", "Shelter", "Wind force and air speed multiplied by 1 - gain x cover, 0 to 1.", high: 1, gate: on),
            .decimal("structures.barrier_gain", "3", "Barrier", "Drag multiplied by 1 + gain x material here, 0 to 3 (at most a 4x slowdown).", high: 3, gate: on),
            .decimal("structures.shield_gain", "0.5", "Shield",
                     "A victim's predation demand multiplied by 1 - gain x cover, 0 to 1. 1 needs combat.protect_floor (command line).", high: 1, gate: on),
            .decimal("structures.reach", "0.5", "Reach", "Weight of face neighbours in the cover, 0 to 1.", high: 1, gate: on),
            .decimal("structures.open_frac", "0.5", "Open threshold", "A cell counts as open below this fraction of the cap, in (0, 1].",
                     low: 0, high: 1, lowOpen: true, gate: on),
            .decimal("structures.aperture", "0.25", "Aperture", "Share of neighbours every cell keeps open (the anti-trapping rule), 0 to 1; 0 switches the rule off.",
                     high: 1, gate: on),
            .decimal("structures.exit_ease", "0.75", "Exit ease", "Leaving a wall costs (1 - exit_ease) of the entering drag, 0 to 1.", high: 1, gate: on),
        ],
        crossErrors: { values in
            var errors: [String] = []
            if let cost = Parse.integer(values["structures.build_cost"]), let rate = Parse.integer(values["structures.build_max"]),
               cost * rate >= 1 << 20 {
                errors.append("structures.build_cost x structures.build_max must be below 1048576.")
            }
            if Parse.decimal(values["structures.shield_gain"]) == 1, !(values.protectFloor > 0) {
                errors.append("structures.shield_gain = 1 needs combat.protect_floor > 0, which is not a lever here (use the command line).")
            }
            return errors
        },
        overlay: "build")
}
