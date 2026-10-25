import Foundation

/// The organism glow of the globe (DESIGN §36.10, the spec's 2.4 layer 5): where the organisms are too small to be drawn one by one, the
/// map carries their density as light. The alive rows are binned on the CPU into a `cells x cells` grid over the torus
/// (`byte = 255 log1p(n) / log1p(16)`), and the map's shader adds it as warm white where the instances have faded out: the glow takes
/// `1 - alpha` of the same ramp that fades the instances in (`OrganismSizePolicy.instanceAlpha`), so the two are complementary by construction.
package enum GlowHistogram {
    /// Cells on a side (160 for a world of 320: cells of 2 units).
    package static let cells = 160
    /// The count that saturates a cell.
    package static let saturation: Float = 16

    /// The byte of a cell holding `count` rows.
    package static func byte(count: Int) -> UInt8 {
        UInt8(min(255 * log1p(Float(count)) / log1p(saturation), 255).rounded())
    }

    /// The grid of `n` rows at `positions` (`3n` floats, x y z) in a world of `size`, row-major from y = 0, as bytes.
    package static func bin(positions: [Float], n: Int, size: Float, cells: Int = GlowHistogram.cells) -> [UInt8] {
        guard size > 0, cells > 0 else { return [] }
        var counts = [UInt16](repeating: 0, count: cells * cells)
        let scale = Float(cells) / size
        let side = Float(cells)
        let top = cells - 1
        positions.withUnsafeBufferPointer { p in
            counts.withUnsafeMutableBufferPointer { c in
                for i in 0..<min(n, p.count / 3) {
                    var fx = p[3 * i] * scale, fy = p[3 * i + 1] * scale
                    guard fx.isFinite, fy.isFinite else { continue }
                    fx -= (fx / side).rounded(.down) * side
                    fy -= (fy / side).rounded(.down) * side
                    let at = min(Int(fy), top) * cells + min(Int(fx), top)
                    if c[at] < UInt16.max { c[at] += 1 }
                }
            }
        }
        let table = (0...Int(saturation)).map { byte(count: $0) }  // (a count of 16 or more is 255)
        return counts.map { table[min(Int($0), table.count - 1)] }
    }

    /// The light of a cell as the shader adds it: `smoothstep(0.05, 0.9, g)` for `g` the byte over 255 (a cell of 4 rows or more is at 0.6).
    package static func light(byte: UInt8) -> Float {
        let g = Float(byte) / 255
        let t = min(max((g - 0.05) / (0.9 - 0.05), 0), 1)
        return t * t * (3 - 2 * t)
    }
}
