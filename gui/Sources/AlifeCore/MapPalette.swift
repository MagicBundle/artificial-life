import Foundation
import simd

/// The weather map's colours (DESIGN §36.6): the Swift mirror of the constants in `mapFragment` (`Shaders.source`), which a test
/// compares with the numbers parsed from the shader, and the same arithmetic on the CPU for the brightness and contrast checks.
///
/// The map has its own luminance range, independent of the clear colour. v1's cold sea (5, 17, 35) sat 1.05:1 from the clear
/// colour (8, 10, 14), and the map was faded to 25% of that at the organism level; the raised cold sea is (20, 59, 107) at Y' 0.212.
package enum MapPalette {
    // MARK: Sea

    /// The sea over the cold half of the world (no warm pool), and over a warm pool.
    package static let coldSea = SIMD3<Float>(0.08, 0.23, 0.42)
    package static let warmSea = SIMD3<Float>(0.10, 0.46, 0.55)
    /// The sea is `mix(cold, warm, smoothstep(seaEdge.x, seaEdge.y, T_s))`.
    package static let seaEdge = SIMD2<Float>(0.1, 0.9)

    // MARK: Vorticity (signed)

    /// Cyclonic water (zeta > 0) is warmer, anticyclonic cooler: `mix(sea, tint, 0.30 smoothstep(0, 0.8, +-zeta))`.
    package static let cyclonic = SIMD3<Float>(0.92, 0.62, 0.30)
    package static let anticyclonic = SIMD3<Float>(0.32, 0.38, 0.92)
    package static let vorticityMix: Float = 0.30
    package static let vorticityEdge: Float = 0.8

    // MARK: Hemispheres

    /// A faint tint of the sea by hemisphere, so that two hemispheres show without a label: blue-violet in the one with f > 0
    /// (0 < y < L/2), teal in the other, 3% each. It is full from `hemisphereFullDistance` (a fraction of L) up and gone at
    /// `hemisphereGoneDistance` and below.
    package static let hemisphereNorth = SIMD3<Float>(0.42, 0.36, 0.92)
    package static let hemisphereSouth = SIMD3<Float>(0.10, 0.72, 0.66)
    package static let hemisphereMix: Float = 0.03
    package static let hemisphereGoneDistance: Float = 0.12
    package static let hemisphereFullDistance: Float = 0.20

    // MARK: Rain and cloud

    /// Rain is a slate blue wash: `mix(colour, rain, clamp(1.4 rain, 0, 0.75))`.
    package static let rain = SIMD3<Float>(0.40, 0.55, 0.82)
    package static let rainGain: Float = 1.4
    package static let rainMax: Float = 0.75
    /// Cloud is nearly white: `mix(colour, cloud, 0.88 smoothstep(0.03, 0.75, cloud))`.
    package static let cloud = SIMD3<Float>(0.95, 0.97, 1.00)
    package static let cloudMix: Float = 0.88
    package static let cloudEdge = SIMD2<Float>(0.03, 0.75)

    // MARK: Ground

    /// The clear colour (`Renderer.background`), and the floor of a world with no weather map.
    package static let background = SIMD3<Float>(0.035, 0.04, 0.055)
    package static let floor = SIMD3<Float>(0.045, 0.055, 0.075)
    /// The two equators (y = 0 and y = L/2): a thin pale line.
    package static let equator = SIMD3<Float>(0.75, 0.80, 0.90)
    package static let equatorMix: Float = 0.35

    // MARK: Territory

    /// The tint of a territory cell in its owner's colour (the "tint" state; the "borders" state has none), and the border
    /// between cells of different owners, white.
    package static let territoryTint: Float = 0.10
    package static let territoryBorder: Float = 0.55

    // MARK: Arithmetic

    package static func smoothstep(_ lo: Float, _ hi: Float, _ x: Float) -> Float {
        let t = min(max((x - lo) / (hi - lo), 0), 1)
        return t * t * (3 - 2 * t)
    }

    /// The hemisphere tint's weight at camera distance `distance` (a fraction of L): 0 up to `hemisphereGoneDistance`, 1 from
    /// `hemisphereFullDistance`.
    package static func hemisphereWeight(distance: Float) -> Float {
        smoothstep(hemisphereGoneDistance, hemisphereFullDistance, distance)
    }

    /// The hemisphere tint at a point with `northness` (1 in the half with f > 0, 0 in the other; the shader smooths the change
    /// over the equators), scaled by the camera's `hemisphereWeight`.
    package static func hemisphereTint(northness: Float, weight: Float) -> (color: SIMD3<Float>, mix: Float) {
        (mix(hemisphereSouth, hemisphereNorth, northness), weight * hemisphereMix)
    }

    /// The map's colour (0...1 per channel) for sea temperature `sea`, signed vorticity `zeta` in -1...1, `rain` and `cloud`,
    /// at a point of `northness` (see `hemisphereTint`) under the hemisphere tint's `hemisphereWeight`. The order is the shader's:
    /// sea, hemisphere, vorticity, rain, cloud, then the exposure.
    package static func color(sea temperature: Float, zeta: Float, rain rainDepth: Float, cloud cloudDepth: Float,
                              northness: Float = 0.5, hemisphereWeight: Float = 0, exposure: Float = 1) -> SIMD3<Float> {
        var c = mix(coldSea, warmSea, smoothstep(seaEdge.x, seaEdge.y, temperature))
        let tint = hemisphereTint(northness: northness, weight: hemisphereWeight)
        c = mix(c, tint.color, tint.mix)
        c = mix(c, cyclonic, vorticityMix * smoothstep(0, vorticityEdge, zeta))
        c = mix(c, anticyclonic, vorticityMix * smoothstep(0, vorticityEdge, -zeta))
        c = mix(c, rain, min(max(rainGain * rainDepth, 0), rainMax))
        c = mix(c, cloud, cloudMix * smoothstep(cloudEdge.x, cloudEdge.y, cloudDepth))
        return simd_clamp(c * exposure, SIMD3(repeating: 0), SIMD3(repeating: 1))
    }

    /// What the world pass writes for a map colour at map opacity `opacity`: the colour mixed back towards the clear colour.
    package static func shown(_ color: SIMD3<Float>, opacity: Float) -> SIMD3<Float> {
        mix(background, color, opacity)
    }

    /// 8-bit values as the render target keeps them.
    package static func eightBit(_ c: SIMD3<Float>) -> SIMD3<Float> {
        simd_clamp((c * 255).rounded(.toNearestOrEven),SIMD3(repeating: 0), SIMD3(repeating: 255)) / 255
    }

    private static func mix(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ t: Float) -> SIMD3<Float> { a + (b - a) * t }
}
