import Foundation

// What the World window shows of the five feature layers (DESIGN §45): which layers the stream carries, how a cell level and a bond flow
// are drawn, and the words of the inspector for them. Pure values: the renderer and the panels read these constants and the tests pin them.

/// Which feature layers the world in view carries, from its header (`play.records`, `play.bonds`, `color_modes`). A world without a feature
/// has none of its records, so nothing here is offered for it.
package struct FeatureLayers: Equatable, Sendable {
    /// `STRC`: the built material.
    package var structures = false
    /// `FSOL`: the farmed soil.
    package var soil = false
    /// `FSIG`: the voice of the focused organism.
    package var voice = false
    /// `FRMA` carries the bond flow (`FLAG_BONDFLOW`).
    package var bondFlow = false
    /// The `role` colour mode is listed.
    package var roles = false

    package init() {}

    package init(header: StreamHeader?) {
        guard let header else { return }
        let records = header.play?.records ?? []
        structures = records.contains("STRC")
        soil = records.contains("FSOL")
        voice = records.contains("FSIG")
        bondFlow = header.play?.bonds?.flow == true
        roles = header.colorModes.contains(RoleColours.mode)
    }

    package var any: Bool { structures || soil || voice || bondFlow || roles }
}

/// How a layer of cells is drawn: a flat tile in 2D, a translucent cube of side `field_cell` in 3D, the colour fixed per layer and the
/// opacity a function of the level (a decaying wall visibly thins).
package enum CellLook {
    /// Warm grey: built material.
    package static let structures = SIMD3<Float>(0.82, 0.75, 0.62)
    /// Green: farmed soil.
    package static let soil = SIMD3<Float>(0.30, 0.78, 0.36)

    /// The opacity of a cell is affine in its level, `base + slope · level / 255` (the shader does this sum): a flat tile is faint at level 1
    /// and 0.78 at a full cell; a cube is lower, since cubes pile up along the view.
    package static let tile = (base: Float(0.18), slope: Float(0.60))
    package static let cube = (base: Float(0.06), slope: Float(0.34))

    package static func tileAlpha(level: UInt8) -> Float { level == 0 ? 0 : tile.base + tile.slope * Float(level) / 255 }
    package static func cubeAlpha(level: UInt8) -> Float { level == 0 ? 0 : cube.base + cube.slope * Float(level) / 255 }

    /// The most cubes one draw makes (`STRC_MAX_CELLS`).
    package static let maxInstances = PlayCells.maxCells
}

/// How the flow along a bond is drawn: each half of the bond line takes a colour for what it does, the donor's warm and the receiver's cool,
/// the more saturated the stronger the flow; no flow leaves the line in the colours of its agents.
package enum BondFlowLook {
    package static let donor = SIMD3<Float>(1.0, 0.55, 0.12)
    package static let receiver = SIMD3<Float>(0.18, 0.75, 1.0)

    /// The strength of a stored flow (`round(127 · clip(f / scale, -1, 1))`) in 0...1.
    package static func strength(_ flow: Int8) -> Float { min(abs(Float(flow)), 127) / 127 }

    /// The quanta per step the stored value stands for, given the header's `play.bonds.scale`.
    package static func quanta(_ flow: Int8, scale: Double) -> Double { Double(flow) / 127 * scale }
}

extension PlayCells {
    /// The level of cell `(x, y, z)`, 0 when it is not listed (or outside the grid). `indices` is ascending, so this is a binary search.
    package func level(x: Int, y: Int, z: Int) -> UInt8 {
        guard (0..<nx).contains(x), (0..<ny).contains(y), (0..<nz).contains(z) else { return 0 }
        let target = UInt32((z * ny + y) * nx + x)
        var low = 0, high = indices.count
        while low < high {
            let mid = (low + high) / 2
            if indices[mid] < target { low = mid + 1 } else { high = mid }
        }
        return low < indices.count && indices[low] == target ? levels[low] : 0
    }

    /// The level of the cell a point of the world is in: `x` and `y` wrap with the world (`field_cell` is the cell's length), `z` is bounded.
    package func level(atX x: Float, y: Float, z: Float, cell: Double) -> UInt8 {
        guard cell > 0, x.isFinite, y.isFinite, z.isFinite else { return 0 }
        func wrap(_ v: Float, _ n: Int) -> Int {
            let i = Int((Double(v) / cell).rounded(.down))
            return ((i % n) + n) % n
        }
        let zi = min(max(Int((Double(z) / cell).rounded(.down)), 0), nz - 1)
        return level(x: wrap(x, nx), y: wrap(y, ny), z: zi)
    }
}

extension WorldStore {
    package var layers: FeatureLayers { FeatureLayers(header: header) }
}

/// The words of the inspector for the feature layers.
package enum FeatureWords {
    /// "62% of the cap (about 2540 units)".
    package static func cellText(level: UInt8, cap: Int) -> String {
        let percent = Int((100 * Double(level) / 255).rounded())
        let units = Int(PlayCells.units(level, cap: cap).rounded())
        return "\(percent)% of the cap (about \(units) units)"
    }

    /// The role whose colour is `rgb` in the role colour mode, nil for another colour (an agent hurt in the last step is red).
    package static func role(forColour rgb: UInt32) -> String? {
        RoleColours.entries.first { $0.rgb == rgb & 0xFFFFFF }?.name
    }

    /// A voice channel's value as a fraction: `says` is `floor(256 v)` clipped to 255, `hears` is `round(255 h)`.
    package static func fraction(_ byte: UInt8) -> Double { Double(byte) / 255 }
}

/// The legend over the map for the feature layers in view: the roles when the colour mode is `role`, the swatch of each cell layer that is
/// drawn, and the two halves of a bond with flow. Nothing for a world without them.
package enum FeatureLegend {
    package struct Entry: Equatable, Sendable {
        package var title: String
        package var colour: SIMD3<Float>
    }

    package static func entries(layers: FeatureLayers, display: WorldDisplay, colourMode: String?) -> [Entry] {
        func colour(_ rgb: UInt32) -> SIMD3<Float> {
            SIMD3(Float((rgb >> 16) & 255), Float((rgb >> 8) & 255), Float(rgb & 255)) / 255
        }
        var entries: [Entry] = []
        if layers.roles, RoleColours.isRoleMode(colourMode) {
            entries += RoleColours.entries.map { Entry(title: $0.name, colour: colour($0.rgb)) }
        }
        if layers.structures, display.structures { entries.append(Entry(title: "Built material", colour: CellLook.structures)) }
        if layers.soil, display.soil { entries.append(Entry(title: "Farmed soil", colour: CellLook.soil)) }
        if layers.bondFlow, display.bondFlow, display.bonds {
            entries.append(Entry(title: "Bond: gives", colour: BondFlowLook.donor))
            entries.append(Entry(title: "Bond: receives", colour: BondFlowLook.receiver))
        }
        return entries
    }
}
