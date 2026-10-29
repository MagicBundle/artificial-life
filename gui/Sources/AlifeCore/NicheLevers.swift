import Foundation

// MARK: - Niche construction: farming

/// The levers of farming (`[niche]`, DESIGN §41, §45): organisms put units into the soil of their food cell, which adds to its regrowth. `niche.farm`
/// is the switch and fixes the brain layout and the state shapes (a fork cannot change it). Ranges mirror `ext/niche.py` `_validate`.
///
/// Two keys are command-line only: `niche.silent_init` (founders' farm column zeroed: a start-of-run choice) and `niche.control` (`none`, `ablate`,
/// `mute`: a fork-only experiment arm). The cost of a unit scales with the regrowth: the 2D overlay `niche` has `farm_cost` 30, the 3D overlay
/// `niche3d` 150 (`NicheLevers.group.overlayName(dims:)`), and the engine warns when a cost is far below the design point.
package enum NicheLevers {
    private static let on = FeatureLeverSpec.Gate.flag("niche.farm")

    package static let group = FeatureLeverGroup(
        feature: "niche",
        title: "Farming",
        summary: "Organisms farm the soil of their food cell: a unit put into it decays slowly and adds to the cell's regrowth. The overlay `niche` (2D) or `niche3d` (3D) switches it on.",
        switchKey: "niche.farm",
        specs: [
            .flag("niche.farm", "false", "Farming on",
                  "Adds the soil, its sensor and the put action. Every lever below is inert while this is off.", layout: true),
            .integer("niche.farm_rate", "8", "Put rate", "Units put per step at output 1, 1 to 256.", low: 1, high: 256, gate: on),
            .integer("niche.farm_cap", "4095", "Soil cap", "The soil level a cell can hold, 1 to 32767.", low: 1, high: 32767, gate: on),
            .integer("niche.farm_cost", "30", "Put cost",
                     "Quanta per unit put; 0 = free. 30 is the 2D design point, about 150 in 3D (the regrowth is larger there).",
                     low: 0, high: 32768, gate: on),
            .integer("niche.farm_decay_q16", "16", "Soil decay", "Decay per step in 1/65536; soil always decays, so at least 1 (16 is a mean life of 4096 steps).",
                     low: 1, high: 65535, gate: on),
            .decimal("niche.farm_bonus_frac", "0.5", "Bonus", "The bonus at full soil, as a fraction of regen_max, on top of the regrowth, 0 to 1.", high: 1, gate: on),
        ],
        crossErrors: { values in
            var errors: [String] = []
            if let cost = Parse.integer(values["niche.farm_cost"]), let rate = Parse.integer(values["niche.farm_rate"]), cost * rate >= 1 << 20 {
                errors.append("niche.farm_cost x niche.farm_rate must be below 1048576.")
            }
            if let cap = Parse.integer(values["niche.farm_cap"]), let decay = Parse.integer(values["niche.farm_decay_q16"]), cap * decay > 1 << 30 {
                errors.append("niche.farm_cap x niche.farm_decay_q16 must be at most 2^30 (the dithered loss is int32).")
            }
            return errors
        },
        overlay: "niche",
        overlay3D: "niche3d")
}
