import Foundation

// MARK: - Arms races

/// The levers of the arms-race layer (`[arms]`, DESIGN §42, §45): heritable attack, armour, toxin-free key and lock, the Tullock contest.
/// `arms.enabled` is the switch; every other key is inert while it is off (the engine requires its default), so the GUI passes none
/// of them then. Ranges mirror `ext/arms.py` `_validate`. The layer has no stream record in this series (`FARS` is deferred): its
/// effects show in the metrics charts (`arms_*`), not on the map.
///
/// The keys it leaves out are command-line only: none (the A3/A4 keys `pool`, `pierce`, `perception` and `dump_n` are not in the engine yet).
package enum ArmsLevers {
    private static let on = FeatureLeverSpec.Gate.flag("arms.enabled")

    package static let group = FeatureLeverGroup(
        feature: "arms",
        title: "Arms race",
        summary: "Heritable attack and armour, a key and lock that must match, and the contest between them. Provisional until the calibration runs.",
        switchKey: "arms.enabled",
        specs: [
            .flag("arms.enabled", "false", "Arms race on",
                  "Adds the heritable traits and the contest. Every lever below is inert (and not passed) while this is off.", layout: true),
            .decimal("arms.kappa", "2", "Armour weight",
                     "Armour's weight in the contest, r = kappa A / (kappa A + a0 + ATK). 0 disables armour. kappa / (kappa + a0) must be at most 0.9.", gate: on),
            .decimal("arms.a0", "1", "Baseline attack", "The attacker's baseline strength in the contest's denominator, > 0.",
                     low: 0, lowOpen: true, gate: on),
            .decimal("arms.key_floor", "1", "Key/lock floor",
                     "Match floor of the key and lock, 0 to 1. 1 switches the axis off; lower makes a mismatch costlier. 0 needs combat.protect_floor (command line).",
                     high: 1, gate: on),
            .decimal("arms.speed_slow", "0", "Armour slows", "Thrust factor 1 - speed_slow * ARM: the cost of armour in speed, 0 to 0.9.",
                     high: 0.9, gate: on),
            .decimal("arms.upkeep_atk", "0", "Attack upkeep", "Quanta per step at attack 1, quadratic, below 32768.", high: 32767, gate: on),
            .decimal("arms.upkeep_arm", "0", "Armour upkeep", "Quanta per step at armour 1, quadratic, below 32768.", high: 32767, gate: on),
            .decimal("arms.init_atk", "0", "Founders' attack", "Starting attack of founders and immigrants, 0 to 1.", high: 1, gate: on),
            .decimal("arms.init_arm", "0", "Founders' armour", "Starting armour of founders and immigrants, 0 to 1.", high: 1, gate: on),
            .decimal("arms.sigma", "0", "Trait mutation", "Fixed mutation step of the attack and armour magnitudes, 0 to 0.5.", high: 0.5, gate: on),
            .decimal("arms.sigma_phase", "0", "Phase mutation", "Fixed mutation step of the key and lock phases (turns), 0 to 0.5.", high: 0.5, gate: on),
        ],
        crossErrors: { values in
            var errors: [String] = []
            if let kappa = Parse.decimal(values["arms.kappa"]), let a0 = Parse.decimal(values["arms.a0"]), a0 > 0,
               kappa / (kappa + a0) > 0.9 {
                errors.append("arms: kappa / (kappa + a0) must be at most 0.9 (no immunity).")
            }
            if Parse.decimal(values["arms.key_floor"]) == 0 {
                errors.append("arms.key_floor = 0 needs combat.protect_floor > 0, which is not a lever here (use the command line).")
            }
            return errors
        },
        overlay: nil)
}
