import Foundation
import simd

/// The height colour mode (DESIGN §36.5, the owner's request after the play of 2026-10-24: "still no intuitive sense of space and
/// dimensions"): an organism's colour is its altitude, `z` in `[0, H]` of a 3D world, on a perceptually uniform ramp. It is computed in
/// the client from the `FRMA` positions (the vertex shaders of the organisms and of their bonds; the engine's colour modes are not
/// touched), so it costs a few instructions a row and no stream, config or engine change.
///
/// **The ramp is viridis** (van der Walt and Smith, matplotlib, CC0), sampled at 17 stops and interpolated linearly in sRGB between
/// them. Why this one: its lightness rises steadily with the value (a greyscale print, or a viewer who sees no colour, still reads
/// height as brightness, and the highest organisms, nearest the eye, are the brightest), it is safe for the common colour-vision
/// deficiencies, and it has **no red**: the engine paints an organism hurt last step pure red (`FF0000`) in every mode, and that stays
/// unmistakable (the shader keeps that colour as it is). Its cost is a teal in the middle that sits close to the warm sea
/// (`MapPalette.warmSea`) where a warm pool is below; the organism's outline (§36.7) is what separates it there.
package enum HeightColour {
    /// The name of the mode in the View menu's list and the `c` key's cycle, beside the engine's own modes.
    package static let modeName = "height"
    /// What the menu calls it.
    package static let title = "Height"

    /// Whether the mode is the one a 3D world opens in, until the player picks another. **It is not** (DESIGN §36.5 says why: the colour
    /// is the one channel that names the kin groups the Cast, the territory and the player's decisions are about, and half of the
    /// recorded planet's organisms are in the lowest fifth of the volume, so as the default it would paint most of the picture in the
    /// darkest end of the ramp over a dark sea); the player turns it on with `c` or View > Colour, and the choice is kept. This is the
    /// one constant to change if the owner prefers it the other way. The mode is available in any world with some height.
    package static let isDefaultIn3D = false

    /// The mode exists in a world of this height (a 2D world has none).
    package static func isAvailable(dims: Int, height: Float) -> Bool { dims == 3 && height > 0 }

    /// Viridis at 0, 1/16, ..., 1, as `RRGGBB`.
    package static let hexStops: [UInt32] = [
        0x440154, 0x48186A, 0x472D7B, 0x424086, 0x3B528B, 0x33638D, 0x2C728E, 0x26828E, 0x21918C,
        0x1F9F88, 0x27AD81, 0x3DBC74, 0x5CC863, 0x81D34D, 0xAADC32, 0xD5E21A, 0xFDE725,
    ]

    /// The stops as sRGB in 0...1 (what the shader's table holds).
    package static let stops: [SIMD3<Float>] = hexStops.map { hex in
        SIMD3(Float((hex >> 16) & 255), Float((hex >> 8) & 255), Float(hex & 255)) / 255
    }

    /// The colour at `t` in 0...1 (clamped): linear between the two stops around it, in sRGB, as the shader does.
    package static func color(_ t: Float) -> SIMD3<Float> {
        guard !t.isNaN else { return stops[0] }
        let scaled = min(max(t, 0), 1) * Float(stops.count - 1)
        let i = min(Int(scaled), stops.count - 2)
        let f = scaled - Float(i)
        return stops[i] + (stops[i + 1] - stops[i]) * f
    }

    /// The position on the ramp of an organism at height `z` in a world `height` thick.
    package static func fraction(z: Float, height: Float) -> Float {
        guard height > 0, z.isFinite else { return 0 }
        return min(max(z / height, 0), 1)
    }

    /// The table as the Metal shader source writes it (`Shaders.source` interpolates this, so the two cannot differ).
    package static var shaderTable: String {
        stops.map { String(format: "float3(%.5f, %.5f, %.5f)", $0.x, $0.y, $0.z) }.joined(separator: ", ")
    }

    /// The ticks of the legend: heights 0, H/2 and H, as the labels say them.
    package static func ticks(height: Float) -> [(fraction: Float, label: String)] {
        [(0, "0"), (0.5, String(format: "%.0f", height / 2)), (1, String(format: "%.0f", height))]
    }

    /// Whether the colour is the engine's hurt red, which the height mode leaves alone.
    package static func isHurtRed(_ rgb: UInt32) -> Bool { rgb & 0xFFFFFF == 0xFF0000 }
}
