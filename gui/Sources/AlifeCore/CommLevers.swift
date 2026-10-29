import Foundation

// MARK: - Communication

/// The levers of the voice channel (`[comm]`, DESIGN §39, §45): W call channels heard by the neighbours inside a range. `comm.enabled`
/// and `comm.width` fix the brain layout (a fork cannot change them). Ranges mirror `ext/comm.py` `_validate`.
///
/// `comm.control` (`none`, `deaf`, `decoy`) is command-line only: it is the receiver-side ablation of an experiment, not something to
/// explore from a panel, and the engine allows it only with the feature on.
package enum CommLevers {
    private static let on = FeatureLeverSpec.Gate.flag("comm.enabled")

    package static let group = FeatureLeverGroup(
        feature: "comm",
        title: "Communication",
        summary: "A voice: W call intensities per organism, heard by the neighbours in range with a linear falloff. The overlay `comm` switches it on.",
        switchKey: "comm.enabled",
        specs: [
            .flag("comm.enabled", "false", "Voice on",
                  "Adds the voice channels and their brain inputs and outputs. Every lever below is inert while this is off.", layout: true),
            .integer("comm.width", "4", "Channels", "Call channels per organism (W), 1 to 8. Fixed for the run.", low: 1, high: 8, gate: on, layout: true),
            .decimal("comm.range", "0", "Hearing range", "0 = the sensing radius, else a distance up to it (the engine checks the radius).", gate: on),
            .decimal("comm.atten_pow", "1", "Falloff exponent", "Exponent of the linear-to-zero distance weight, > 0.", low: 0, lowOpen: true, gate: on),
            .decimal("comm.kin_exp", "0", "Listen to kin", "Hearing weight x kin similarity to this power; 0 = everyone is heard.", gate: on),
            .decimal("comm.cost", "4", "Call cost", "Quanta per unit of call intensity (dithered), >= 0; cost x width must stay below 32768.", gate: on),
        ],
        crossErrors: { values in
            if let cost = Parse.decimal(values["comm.cost"]), let width = Parse.integer(values["comm.width"]), cost * Double(width) >= 32768 {
                return ["comm.cost x comm.width must be below 32768 (the per-row due)."]
            }
            return []
        },
        overlay: "comm")
}
