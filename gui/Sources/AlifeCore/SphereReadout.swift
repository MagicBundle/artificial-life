import Foundation
import simd

/// The words the planet's cells are described in (the inspector of a clicked cell, the tooltips): from the records, as sentences a player
/// reads. Pure.
package enum SphereReadout {
    /// "12.3N 45.6E · rain 35%, cloud 60% · House Foshiov territory · built 62% of the cap (about 2540 units) · farmed soil 100% ...", from
    /// the field pixel `pixel`: its place, the weather over it, whose territory it is, and what is built and farmed in the column above it.
    package static func cellText(pixel: Int, sphere: SphereHeader, store: WorldStore) -> String {
        guard (0..<sphere.fieldCells).contains(pixel) else { return "no such cell" }
        let centre = SphereCentres.shared.centres(nside: sphere.nsideField)[pixel]
        var parts = [SphereFrame.placeText(centre) + "  (cell \(pixel))"]
        if let map = store.sphereMap, map.npix == sphere.fieldCells {
            let rain = Int((Float(map.p[pixel]) / 255 * 100).rounded())
            let cloud = Int((Float(map.dc[pixel]) / 255 * 100).rounded())
            let sea = Int((Float(map.ts[pixel]) / 255 * 100).rounded())
            parts.append("rain \(rain)%, cloud \(cloud)%, warm sea \(sea)%")
        }
        if let terr = store.territory, let nside = terr.nside {
            let t = Healpix.pixelFloat(of: centre, nside: nside)
            let slot = Int(terr.owner[t])
            if slot < terr.ids.count, terr.ids[slot] >= 0 {
                let id = terr.ids[slot]
                let name = store.identities?.tables.kin.first { $0.id == id }?.name ?? "kin group \(id)"
                parts.append("\(name) territory")
            }
        }
        if let material = store.material, let cap = store.play?.structures?.cap, let level = columnLevel(material, pixel: pixel, sphere: sphere) {
            parts.append("built " + FeatureWords.cellText(level: level, cap: cap))
        }
        if let soil = store.soil, let cap = store.play?.niche?.cap, let level = columnLevel(soil, pixel: pixel, sphere: sphere) {
            parts.append("farmed " + FeatureWords.cellText(level: level, cap: cap))
        }
        return parts.joined(separator: " · ")
    }

    /// The highest level in the column of layers above a field pixel, nil when it is empty.
    package static func columnLevel(_ cells: PlayCells, pixel: Int, sphere: SphereHeader) -> UInt8? {
        guard cells.nx == sphere.fieldCells, cells.ny == 1 else { return nil }
        var best: UInt8 = 0
        for z in 0..<cells.nz { best = max(best, cells.level(x: pixel, y: 0, z: z)) }
        return best == 0 ? nil : best
    }

    /// "Northern hemisphere", "Southern hemisphere" or "Equator" for a latitude in degrees.
    package static func hemisphere(lat: Double) -> String {
        abs(lat) < 5 ? "Equator" : (lat > 0 ? "Northern hemisphere" : "Southern hemisphere")
    }
}
