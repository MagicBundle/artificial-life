import Foundation
import simd

/// The outline of an organism and the colour arithmetic it rests on (DESIGN §36.7). Pure: the impostor shaders compute the same
/// rule per pixel, and a test compares the two.
///
/// An outline of one drawable pixel is added to both impostor kinds. Its colour is chosen from the luminance `L` (linearised sRGB
/// relative luminance) of the **shaded** colour of the pixel next to the ring, so that the edge contrast is at least 3:1 for any
/// colour, any depth cue and any shading:
///
/// ```text
/// shaded  = base * depthCue * (0.3 + 0.7 n.v)             a sphere (a flat disc: (1 - 0.3 r^2))
/// outline = L(shaded) > 0.17 ? (0.06, 0.06, 0.06) : (0.92, 0.92, 0.92)
/// ```
///
/// A rule on the base colour fails under the shading the renderer keeps (3 of 25 palette colours under 3:1 at a depth cue of 0.62,
/// 15 of 25 at a sphere's rim); on the shaded colour it cannot.
///
/// The threshold is the luminance at which the two outlines have the same contrast: 0.17 gives 4.0 to either side. The pixel next
/// to the ring is 1 to 2 px inside the edge, wherever the pixel grid falls on the silhouette, and the rule is evaluated at 1.5 px:
/// the true luminance of that pixel is within about 1.4 times the estimate, which the equal-contrast threshold absorbs (the light
/// outline holds up to L = 0.243, the dark one down to 0.114). The first draft's 0.22 gave 4.9 and 3.3 and measured 2.88 on the GPU.
package enum OutlineContrast {
    /// The luminance above which the dark outline is used.
    package static let threshold: Float = 0.17
    package static let dark = SIMD3<Float>(repeating: 0.06)
    package static let light = SIMD3<Float>(repeating: 0.92)
    /// The ring's width in drawable pixels.
    package static let width: Float = 1
    /// The depth cue of an organism at height `z` of `height`: `0.85 + 0.15 z / height` (v1's cue was 0.62 + 0.38 z / height, which
    /// dimmed 58% of a recorded population by more than 25%).
    package static let depthCueBase: Float = 0.85
    package static let depthCueSlope: Float = 0.15

    /// The linear value of an sRGB-encoded channel in 0...1.
    package static func linear(_ c: Float) -> Float {
        c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
    }

    /// The relative luminance (WCAG) of an sRGB colour in 0...1 per channel.
    package static func luminance(_ rgb: SIMD3<Float>) -> Float {
        0.2126 * linear(rgb.x) + 0.7152 * linear(rgb.y) + 0.0722 * linear(rgb.z)
    }

    /// The WCAG contrast ratio of two sRGB colours, 1 to 21.
    package static func contrast(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        let la = luminance(a), lb = luminance(b)
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)
    }

    /// The luma Y' = 0.2126 R + 0.7152 G + 0.0722 B of the encoded values (what the brightness floors of the tests measure).
    package static func luma(_ rgb: SIMD3<Float>) -> Float {
        0.2126 * rgb.x + 0.7152 * rgb.y + 0.0722 * rgb.z
    }

    /// The outline colour for a pixel whose shaded colour next to the ring is `shaded`.
    package static func outline(forShaded shaded: SIMD3<Float>) -> SIMD3<Float> {
        luminance(shaded) > threshold ? dark : light
    }

    /// The brightness factor of an organism at height `z` of a world `height` high (1 in a 2D world, which has none).
    package static func depthCue(z: Float, height: Float) -> Float {
        guard height > 0 else { return 1 }
        return depthCueBase + depthCueSlope * min(max(z / height, 0), 1)
    }

    /// The headlight factor of a sphere's surface whose normal makes `normalDotView` with the view direction.
    package static func sphereShade(normalDotView nv: Float) -> Float { 0.3 + 0.7 * max(nv, 0) }

    /// The shading factor of a flat disc at `r` (0 at the centre, 1 at the rim) of its radius.
    package static func discShade(r: Float) -> Float { 1 - 0.3 * r * r }

    /// n.v of a sphere of `radiusPixels` at the centre of the pixel next to the ring: `width + 0.5` pixels inside the edge,
    /// where the shaded colour the outline is chosen against is taken (0.55 at 9 px, 0.87 at 3 px).
    package static func ringNormalDotView(radiusPixels r: Float) -> Float {
        guard r > 0 else { return 0 }
        let k = min(max((r - width - 0.5) / r, 0), 1)
        return (1 - k * k).squareRoot()
    }

    /// The `r` of a flat disc of `radiusPixels` at the centre of the pixel next to the ring.
    package static func ringDiscRadius(radiusPixels r: Float) -> Float {
        guard r > 0 else { return 0 }
        return min(max((r - width - 0.5) / r, 0), 1)
    }

    /// The shaded colour of a sphere impostor: `base * depthCue * (0.3 + 0.7 n.v)`.
    package static func shadedSphere(base: SIMD3<Float>, depthCue: Float, normalDotView nv: Float) -> SIMD3<Float> {
        base * depthCue * sphereShade(normalDotView: nv)
    }
}
