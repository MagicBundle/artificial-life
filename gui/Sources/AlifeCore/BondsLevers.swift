import Foundation

// MARK: - Bonds that pay, and roles

/// The levers of the bond layer (`[bonds]` sharing keys and `[roles]`, DESIGN §40, §45), one feature in the engine (`ext/bonds.py`): the
/// conductive sharing along bonds (`bonds.share_mode`) and the committed forager/guard roles (`roles.enabled`). Either switches the
/// bond block of the brain on, which a fork cannot change (`FeatureLevers.layoutErrors`: the block on or off, not the mode 1 or 2). Ranges mirror `ext/bonds.py` `_validate`.
///
/// Not levers, because they are not in the feature's table: `bonds.share_rate` and the older bond keys (existing engine keys).
package enum BondsLevers {
    private static let sharing = FeatureLeverSpec.Gate.nonzero("bonds.share_mode")
    private static let roles = FeatureLeverSpec.Gate.flag("roles.enabled")

    package static let group = FeatureLeverGroup(
        feature: "bonds",
        title: "Bonds and roles",
        summary: "Sharing energy along bonds (0 the legacy gift, 1 conductive by mutual consent, 2 conductive and obligate) and committed forager/guard roles. The overlay `bonds` switches both on.",
        switchKey: "bonds.share_mode",
        specs: [
            .options("bonds.share_mode", "0", "Sharing mode", "0 the legacy gift, 1 conductive with mutual consent, 2 conductive and obligate.",
                     [.init(value: "0", label: "0 gift"), .init(value: "1", label: "1 consent"), .init(value: "2", label: "2 obligate")]),
            .decimal("bonds.share_kappa", "0.125", "Share fraction", "Conductive: fraction of the energy gap sent per step, in (0, 0.125].",
                     low: 0, high: 0.125, lowOpen: true, gate: sharing),
            .decimal("bonds.share_kin_floor", "1", "Non-kin efficiency", "Conductive: efficiency scale of a non-kin bond, 0 to 1; 1 is kin-blind.",
                     high: 1, gate: sharing),
            .flag("roles.enabled", "false", "Roles on",
                  "Adds the role state and the bond block of the brain. The role levers below are inert while this is off.", layout: true),
            .decimal("roles.forage_gain", "0", "Forager gain", "A forager's harvest cap grows by this fraction of harvest_max (ramped), >= 0.", gate: roles),
            .decimal("roles.forage_defend_pen", "0", "Forager defence penalty", "A forager's own defence is multiplied by 1 - this, in [0, 1).",
                     high: 1, highOpen: true, gate: roles),
            .decimal("roles.guard_harvest_pen", "0", "Guard harvest penalty", "A guard's harvest cap shrinks by this fraction, in [0, 1).",
                     high: 1, highOpen: true, gate: roles),
            .decimal("roles.guard_shield", "0", "Guard shield", "Protection a guard grants itself and each bonded partner (ramped), 0 to 0.9.", high: 0.9, gate: roles),
            .decimal("roles.synergy", "0", "Synergy", "Extra forager cap beside a guard whose ramp is at least half (fraction of harvest_max), >= 0.", gate: roles),
            .integer("roles.ramp_steps", "64", "Ramp", "Steps after a switch for a role's effects to reach full strength, 1 to 32767.", low: 1, high: 32767, gate: roles),
            .decimal("roles.threshold", "0.5", "Dead zone", "Dead zone of the two role outputs, in (0, 1): inside it the role is kept.",
                     low: 0, high: 1, lowOpen: true, highOpen: true, gate: roles),
            .integer("roles.switch_cost", "0", "Switch cost", "Quanta paid by a row that asks for another role, 0 to 32767.", low: 0, high: 32767, gate: roles),
            .flag("roles.inherit_role", "false", "Inherit role", "A bud starts with its parent's role (else generalist).", gate: roles),
        ],
        crossErrors: { _ in [] },
        overlay: "bonds")
}
