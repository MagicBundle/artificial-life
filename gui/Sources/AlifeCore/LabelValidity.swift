import Foundation

/// Whether the labels on screen are valid (DESIGN §33.7): the same gate N1 applies to the World Report, read from the stream.
///
/// A label (an archetype or intent byte) is valid only while all three hold: the writer's header says the held-out verdict
/// table has a row for this world and rule set (`PlayHeader.Labels.gate.keyFound`), that label's byte is among those that
/// pass, and the environment is at its defaults (`ENVS.levers` empty, N1's "environment changed" rule). Anything else is shown
/// as a provisional label: by default, and in every version-3 (storm) world until W2b's held-out windows add rows. The
/// client decides nothing about thresholds; it only reads the writer's list.
package struct LabelValidity: Equatable, Sendable {
    package var intentVersion: Int
    package var gate: PlayHeader.Labels.Gate?
    /// `ENVS.levers` is empty; false until the first `ENVS` (nothing is known to be at its default).
    package var environmentAtDefault: Bool

    package init(labels: PlayHeader.Labels?, environment: PlayEnvironment?) {
        intentVersion = labels?.intentVersion ?? 0
        gate = labels?.gate
        environmentAtDefault = environment.map { $0.levers.isEmpty } ?? false
    }

    private var gateOpen: Bool { (gate?.keyFound ?? false) && environmentAtDefault }

    /// The archetype byte passes the gate now (0, "none", never does).
    package func archetypeIsValid(_ byte: UInt8) -> Bool { byte > 0 && gateOpen && (gate?.archetypes.contains(Int(byte)) ?? false) }

    /// The intent byte passes the gate now.
    package func intentIsValid(_ byte: UInt8) -> Bool { byte > 0 && gateOpen && (gate?.intents.contains(Int(byte)) ?? false) }

    /// The row of a kin group's mean archetype mix (index `i` is archetype `i + 1`) is valid only if every archetype with
    /// a share of 1% or more is.
    package func mixIsValid(_ mix: [Double]) -> Bool {
        gateOpen && zip(mix, 1...).allSatisfy { $0.0 < 0.01 || archetypeIsValid(UInt8(clamping: $0.1)) }
    }

    /// Why a label is provisional, in a few words; nil when the gate is open (a label may still be missing from its list).
    package var reason: String? {
        guard let gate, gate.keyFound else { return "no held-out verdict for this world and rule set" }
        return environmentAtDefault ? nil : "the environment is away from its defaults"
    }

    /// The note under an organism's card.
    package func note(allValid: Bool) -> String {
        if allValid { return "labels are validated for this world (labels v\(intentVersion))" }
        if let reason { return "labels are provisional (labels v\(intentVersion): \(reason))" }
        return "labels are provisional (labels v\(intentVersion): no passing verdict for this label)"
    }
}
