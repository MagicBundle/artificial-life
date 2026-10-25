import Foundation

/// The look of the globe through the zoom (DESIGN §36.10, the spec's 3.5): every quantity is a function of the one scalar `s`, with no
/// switch. The renderer reads them into `GlobeUniforms`; the tests pin their ends and that none steps.
package enum GlobeLook {
    private static func ss(_ lo: Float, _ hi: Float, _ x: Float) -> Float { Morph.smoothstep(lo, hi, x) }

    /// Space against the haze sky (the colour behind the map): the haze sky is the flat close-up's horizon.
    package static func space(s: Float) -> Float { ss(0.1, 0.7, s) }
    /// The stars and the atmosphere's glow fade in over `s` in [0.2, 0.8].
    package static func stars(s: Float) -> Float { ss(0.2, 0.8, s) }
    /// The horizon haze belongs to the flat close-up: full to `s` = 0.3, gone from 0.7.
    package static func fogStrength(s: Float) -> Float { 1 - ss(0.3, 0.7, s) }
    /// The haze at the limb of the rolled plane (a ball of radius R / s): from the first morph on, so that the edge against the sky is haze.
    package static func limbFog(s: Float) -> Float { ss(0, 0.04, s) }
    /// The clear half extent of the haze, in fractions of `L`: 0.5 (`WorldFog.clear`) on the flat map, 0.25 by `s` = 0.3. The mesh of the
    /// globe path is three worlds wide at most (its copies fade out over `s` in [0.1, 0.3]), so its edge must be hidden by then.
    package static func fogClear(s: Float) -> Float { WorldFog.clear - 0.25 * ss(0, 0.3, s) }
    /// The width of the fade, in fractions of `L`: 0.2 (`WorldFog.width`), 0.1 by `s` = 0.3.
    package static func fogWidth(s: Float) -> Float { WorldFog.width - 0.10 * ss(0, 0.3, s) }
    /// The lighting of the map (the key light, ambient 0.66): none on the flat map, in full from `s` = 0.8.
    package static func lit(s: Float) -> Float { ss(0.05, 0.8, s) }
    /// The blue rim at the limb.
    package static func rim(s: Float) -> Float { ss(0.2, 0.9, s) }
    /// The graticule, the gold equator and the magenta seam rows; the flat map has its two equator lines instead.
    package static func graticule(s: Float) -> Float { ss(0.4, 0.9, s) }
    /// The pole caps: needed once the mesh's edge rows close up on the sphere.
    package static func caps(s: Float) -> Float { ss(0.55, 0.9, s) }
    /// The territory borders' alpha: 0.55 (F3) on the flat map, easing to 0.25 by `s` = 1.
    package static func territoryBorder(s: Float, flat: Float = MapPalette.territoryBorder) -> Float { flat + (0.25 - flat) * ss(0, 1, s) }
    /// Bonds are drawn while `s` is under this.
    package static let bondsUntil: Float = 0.5
    /// Whether the neighbours of the central tile of the map in x (and the copies across the wrap in y) are drawn at all.
    package static func tilesAlpha(s: Float) -> Float { SeamCopies.alpha(s: s) }
}
