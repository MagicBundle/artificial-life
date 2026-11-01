import Foundation
import simd

/// The colours of the planet's cells (DESIGN §46.8): one RGBA value per HEALPix pixel for the floor, for the cloud shell at the lid and for the
/// territory map, composed on the CPU from the records (`MAPH`, `STRC`, `FSOL`, `TERR`) with the box map's palette (`MapPalette`) so that a
/// storm looks like a storm in either world. The renderer uploads these arrays and the fragment shader looks a cell up by its pixel
/// (`healpix_ring`); the minimap paints the same arrays on the Mollweide chart. Pure and tested.
package enum SphereColours {
    /// The floor of a world with no weather map, and its hemisphere tints (`MapPalette`).
    package static let dryFloor = SIMD3<Float>(0.075, 0.095, 0.125)

    /// `smoothstep(-0.2, 0.2, sin(lat))`: 1 in the northern half (f > 0), 0 in the southern, smooth over the equator, as the box map's tint.
    package static func northness(of centre: SIMD3<Float>) -> Float {
        MapPalette.smoothstep(-0.2, 0.2, centre.z)
    }

    /// The floor: the sea's temperature and vorticity under the rain (a world with a weather map), else the dry floor with a faint
    /// hemisphere tint so that north reads without a label; then the built material and the farmed soil over it, as the box's cells.
    /// `layer` is the lattice layer shown (nil: the column's maximum, the top-down view of what has been built).
    package static func floor(sphere: SphereHeader, map: PlaySphereWeather?, material: PlayCells?, soil: PlayCells?,
                              display: WorldDisplay, layer: Int?, centres: [SIMD3<Float>]) -> [SIMD4<Float>] {
        let n = sphere.fieldCells
        precondition(centres.count == n, "one centre per field cell")
        var out = [SIMD4<Float>](repeating: SIMD4(0, 0, 0, 1), count: n)
        let showMap = display.map && map != nil && map!.npix == n
        for i in 0..<n {
            let north = northness(of: centres[i])
            var c: SIMD3<Float>
            if showMap, let map {
                c = MapPalette.color(
                    sea: Float(map.ts[i]) / 255, zeta: Float(map.zeta[i]) / 127, rain: Float(map.p[i]) / 255, cloud: 0,
                    northness: north, hemisphereWeight: 1, exposure: display.exposure)
            } else {
                let tint = MapPalette.hemisphereTint(northness: north, weight: 1)
                c = dryFloor + (tint.color - dryFloor) * tint.mix
                c = simd_clamp(c * display.exposure, SIMD3(repeating: 0), SIMD3(repeating: 1))
            }
            out[i] = SIMD4(c, 1)
        }
        if display.structures, let material { blend(&out, cells: material, colour: CellLook.structures, sphere: sphere, layer: layer) }
        if display.soil, let soil { blend(&out, cells: soil, colour: CellLook.soil, sphere: sphere, layer: layer) }
        return out
    }

    /// The cells of a sparse record over the floor: a cell's opacity is affine in its level (`CellLook.tileAlpha`); of the layers of a
    /// column the highest level counts, or only layer `layer`.
    package static func blend(_ floor: inout [SIMD4<Float>], cells: PlayCells, colour: SIMD3<Float>, sphere: SphereHeader, layer: Int?) {
        let npix = sphere.fieldCells
        guard cells.nx == npix, cells.ny == 1 else { return }
        var level = [UInt8](repeating: 0, count: npix)
        for k in 0..<cells.count {
            let flat = Int(cells.indices[k])
            let z = flat / npix, p = flat % npix
            if let layer, z != layer { continue }
            level[p] = max(level[p], cells.levels[k])
        }
        for p in 0..<npix where level[p] != 0 {
            let a = CellLook.tileAlpha(level: level[p])
            let c = SIMD3(floor[p].x, floor[p].y, floor[p].z)
            floor[p] = SIMD4(c + (colour - c) * a, 1)
        }
    }

    /// The cloud shell: white where the cloud is deep (`mapColorAt`: `0.88 smoothstep(0.03, 0.75, cloud)`), clear elsewhere. The rain is
    /// painted on the floor under it.
    package static func cloud(sphere: SphereHeader, map: PlaySphereWeather?, display: WorldDisplay) -> [SIMD4<Float>] {
        let n = sphere.fieldCells
        guard display.map, let map, map.npix == n else { return [SIMD4<Float>](repeating: .zero, count: n) }
        return (0..<n).map { i in
            let a = MapPalette.cloudMix * MapPalette.smoothstep(MapPalette.cloudEdge.x, MapPalette.cloudEdge.y, Float(map.dc[i]) / 255)
            let c = min(MapPalette.cloud * display.exposure, SIMD3(repeating: 1))
            return SIMD4(c, a)
        }
    }

    /// The territory map: an owner's slot colour with an alpha from the purity (half to full), clear without an owner; `highlight` (a kin
    /// group id) is the selected group, drawn at full alpha, the others fainter while one is selected.
    package static func territory(_ t: PlayTerritory?, nside: Int, highlight: Int? = nil) -> [SIMD4<Float>] {
        let n = Healpix.npix(nside)
        guard let t, t.nx * t.ny == n else { return [SIMD4<Float>](repeating: .zero, count: n) }
        return (0..<n).map { i in
            let slot = t.owner[i]
            guard slot != Slot.none, Int(slot) < t.ids.count, let rgb = WorldPalette.slotColor(slot) else { return .zero }
            let (r, g, b) = WorldPalette.components(rgb)
            var alpha = 0.5 + 0.5 * Float(t.purity[i]) / 255
            if let highlight { alpha = t.ids[Int(slot)] == highlight ? 1 : alpha * 0.45 }
            return SIMD4(Float(r) / 255, Float(g) / 255, Float(b) / 255, alpha)
        }
    }

    /// `0xAARRGGBB` of a colour (the minimap's image, a chart pixel).
    package static func packed(_ c: SIMD4<Float>) -> UInt32 {
        func byte(_ v: Float) -> UInt32 { UInt32((min(max(v, 0), 1) * 255).rounded()) }
        return byte(c.w) << 24 | byte(c.x) << 16 | byte(c.y) << 8 | byte(c.z)
    }

    /// The chart's colours: the floor with the cloud over it and the territory tint over that, opaque.
    package static func chart(floor: [SIMD4<Float>], cloud: [SIMD4<Float>], territory: [SIMD4<Float>], territoryNside: Int,
                              fieldNside: Int, centres: [SIMD3<Float>], territoryAlpha: Float) -> [UInt32] {
        (0..<floor.count).map { i in
            var c = SIMD3(floor[i].x, floor[i].y, floor[i].z)
            if i < cloud.count { c += (SIMD3(cloud[i].x, cloud[i].y, cloud[i].z) - c) * cloud[i].w }
            if territoryAlpha > 0, !territory.isEmpty {
                let t = territory[Healpix.pixelFloat(of: centres[i], nside: territoryNside)]
                c += (SIMD3(t.x, t.y, t.z) - c) * (t.w * territoryAlpha)
            }
            return packed(SIMD4(c, 1))
        }
    }
}
