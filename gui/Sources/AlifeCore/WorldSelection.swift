import Foundation

/// The numbers of the event registry of version 1 (DESIGN §33.1.8; the header lists them as `play.events`, and a test pins
/// these to it).
package enum EventKind {
    package static let kinBorn = 1, kinSplit = 2, kinMerged = 3, kinDissolved = 4
    package static let houseFounded = 5, houseExtinct = 6
    package static let stormBorn = 7, stormNamed = 8, stormMerged = 9, stormSplit = 10, stormDissolved = 11
    package static let directorEntry = 12
    package static let focusDeath = 13, focusHandoff = 14
    package static let checkpoint = 15, renumbered = 16, extinct = 17
    /// G3b: a kin thread (a group through its id changes) held together for three snapshots, or has been gone for four.
    package static let kinFormed = 18, kinEnded = 19
    /// G3b: a player's power took effect, or its effect ended.
    package static let powerApplied = 20, powerEnded = 21

    package static let registry: [Int: String] = [
        1: "kin_born", 2: "kin_split", 3: "kin_merged", 4: "kin_dissolved", 5: "house_founded", 6: "house_extinct",
        7: "storm_born", 8: "storm_named", 9: "storm_merged", 10: "storm_split", 11: "storm_dissolved",
        12: "director_entry", 13: "focus_death", 14: "focus_handoff", 15: "checkpoint", 16: "renumbered", 17: "extinct",
        18: "kin_formed", 19: "kin_ended", 20: "power_applied", 21: "power_ended",
    ]
}

/// What the player has picked. One thing at a time; the inspector shows it whatever tab is open.
package enum WorldSelection: Equatable, Sendable {
    case none
    /// An organism, by uid (a row is reused after a death, a uid never is).
    case organism(uid: Int64)
    case storm(id: Int)
    /// A kin group, by its id: its territory is highlighted.
    case kin(id: Int)
    case pool(index: Int)

    package var organismUID: Int64? {
        if case .organism(let uid) = self { return uid }
        return nil
    }

    package var isNone: Bool { self == .none }

    /// The server's hand-off: the followed organism died and the world continues on another (`EVNT focus_handoff`, `a` the
    /// old uid and `b` the new). Applied **only when `a` is what is selected now**, so that a hand-off that arrives after the
    /// player clicked something else never overrides the click. True if it was applied.
    package mutating func applyHandoff(_ event: WorldEvent) -> Bool {
        guard event.kind == EventKind.focusHandoff, event.b >= 0, case .organism(let uid) = self, uid == Int64(event.a) else {
            return false
        }
        self = .organism(uid: Int64(event.b))
        return true
    }
}

/// What a click on a chronicle row does (DESIGN §33.4.8): fly to where it happened and select the entity it is about.
package struct ChronicleAction: Equatable, Sendable {
    /// Where it happened, on the floor; nil for an event with no place (a checkpoint, a house).
    package var at: SIMD2<Float>?
    package var select: WorldSelection

    /// The action of an event: a storm's id for a storm event, a kin group's for a kin event, the organism the follow passed
    /// to for a hand-off (the old one is dead) and the dead organism for its death; a house, a director entry and a system
    /// event select nothing (a house is not a cast row).
    package static func of(_ event: WorldEvent) -> ChronicleAction {
        var at: SIMD2<Float>?
        if let x = event.x, let y = event.y { at = SIMD2(Float(x), Float(y)) }
        var select = WorldSelection.none
        switch event.fam {
        case "storm": if event.a > 0 { select = .storm(id: event.a) }
        case "kin": if event.a > 0 { select = .kin(id: event.a) }
        case "focus":
            // (a uid is a real number, 0 included: only the other families number their entities from 1)
            if event.kind == EventKind.focusHandoff, event.b >= 0 {
                select = .organism(uid: Int64(event.b))
            } else if event.a >= 0 {
                select = .organism(uid: Int64(event.a))
            }
        default: break
        }
        return ChronicleAction(at: at, select: select)
    }
}
