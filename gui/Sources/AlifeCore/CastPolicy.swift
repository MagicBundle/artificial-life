import Foundation

// The Cast tab: who is on the stage (DESIGN §33.4.8). Every live storm, the eight largest kin groups, and the one organism
// the player picked. A pure policy over the records, so what the tab lists is tested without a window.

/// A storm in the cast.
package struct CastStorm: Equatable, Identifiable, Sendable {
    package var id: Int
    package var title: String
    package var named: Bool
    package var hemisphere: String
    /// The peak lower-layer vorticity so far, over f0.
    package var peak: Double
    /// `peak` as a fraction of the bar's full scale.
    package var peakFraction: Double
    /// Game minutes since the track began.
    package var ageMinutes: Double
    package var coasting: Bool
    package var x: Double
    package var y: Double
}

/// A kin group in the cast.
package struct CastLineage: Equatable, Identifiable, Sendable {
    /// The kin group's id (the key of selection, territory and events).
    package var id: Int
    /// The slot in the last `IDNT` (the byte its rows and its territory carry).
    package var slot: Int
    package var name: String
    /// The name of the dominant house the group is named after, when it has one.
    package var namedAfter: String?
    package var size: Int
    /// Its share of the organisms the last `IDNT` lists.
    package var share: Double
    /// The mean archetype shares (seven values; provisional labels), nil while the labels warm up.
    package var archetypes: [Double]?
    package var births: Int?
    /// Deaths of the last window by cause (`CastPolicy.deathCauses`).
    package var deaths: [Int]?
    /// The cells of the territory map the group owns.
    package var territoryCells: Int
    package var x: Double?
    package var y: Double?
    package var z: Double?
}

package enum CastPolicy {
    /// Lineages listed.
    package static let maxLineages = 8
    /// A storm's peak at which its bar is full, in f0. Named strength is 0.5.
    package static let fullBar = 1.5
    /// The causes of the death counts of a kin group, in the order the record lists them.
    package static let deathCauses = ["starved", "killed", "aged", "catastrophe", "storm"]

    /// Game minutes of `steps`: `steps / (60·B)` at the base rate `B`, whatever rate the player has set.
    package static func ageMinutes(steps: Int, baseRate: Double) -> Double {
        baseRate > 0 ? Double(steps) / (60 * baseRate) : 0
    }

    /// All live storms, the named first (strongest first), then the disturbances by strength; ties by id.
    package static func storms(_ record: PlayStormRecord?, baseRate: Double) -> [CastStorm] {
        guard let storms = record?.storms.storms else { return [] }
        return storms.map { storm in
            CastStorm(
                id: storm.id, title: Names.storm(storm), named: storm.named, hemisphere: storm.hemi == 0 ? "northern" : "southern",
                peak: storm.zeta, peakFraction: min(max(storm.zeta / fullBar, 0), 1),
                ageMinutes: ageMinutes(steps: storm.age, baseRate: baseRate), coasting: storm.state == "coasting",
                x: storm.x, y: storm.y)
        }
        .sorted { a, b in
            if a.named != b.named { return a.named }
            if a.peak != b.peak { return a.peak > b.peak }
            return a.id < b.id
        }
    }

    /// The `maxLineages` largest kin groups of the last `IDNT`, largest first (ties by slot).
    package static func lineages(_ identities: PlayIdentities?, territory: PlayTerritory?) -> [CastLineage] {
        guard let identities else { return [] }
        let tables = identities.tables
        let alive = max(identities.rows.count, 1)
        var cells: [Int: Int] = [:]
        if let territory {
            for owner in territory.owner where Int(owner) < territory.ids.count { cells[Int(owner), default: 0] += 1 }
        }
        return tables.kin
            .sorted { a, b in a.size != b.size ? a.size > b.size : a.slot < b.slot }
            .prefix(maxLineages)
            .map { kin in
                CastLineage(
                    id: kin.id, slot: kin.slot, name: kin.name,
                    namedAfter: kin.namedAfter.flatMap { id in tables.houses.first { $0.id == id }?.name },
                    size: kin.size, share: Double(kin.size) / Double(alive), archetypes: kin.arch,
                    births: kin.births, deaths: kin.deaths, territoryCells: cells[kin.slot] ?? 0,
                    x: kin.x, y: kin.y, z: kin.z)
            }
    }

    /// `"3 starved, 1 aged"`: the non-zero deaths of a window by cause; nil without a count.
    package static func deathSummary(_ deaths: [Int]?) -> String? {
        guard let deaths else { return nil }
        let parts = zip(deaths, deathCauses).filter { $0.0 > 0 }.map { "\($0.0) \($0.1)" }
        return parts.isEmpty ? "none" : parts.joined(separator: ", ")
    }
}
