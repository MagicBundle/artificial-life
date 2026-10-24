import Foundation
import simd

/// The horizon fade of a tilted view (DESIGN §36.5, camera controls): what replaces the footprint clamp of the tilt, so that the view
/// may be tilted to edge-on and the edge of the drawn world is never a hard line.
///
/// **The fog is a function of where a floor point is in the world, not of where the eye is**: its Chebyshev distance `C` (the larger of
/// `|dx|` and `|dy|`) from the point looked at, unwrapped. Up to `clear * L` it is nothing, then it grows as
/// `1 - exp(-((C - clear L) / (width L))^2)` (exponential in the square of the excess, so that the clear box stays clear and the
/// fade is complete over a short range): 22% at 0.6 L, 63% at 0.7 L, 98% at 0.9 L, 99.8% at L. Why not the eye's distance: the Top
/// view reaches `0.49 L` from the point looked at at its farthest (`WorldPose.reach`), so any fog that starts nearer than that
/// shows in the Top view, and any fog tied to the eye is nothing at the Top view and strongest where the eye is far from the point
/// looked at, which is exactly where the near edge of the patch would show. In the world's frame the clear box is the one world
/// everything is drawn in (every organism, storm and bond is at its minimum image about the point looked at, so none is beyond
/// `L/2` of it); beyond it the map is the same map again, which the fog hides.
///
/// **The mesh behind it reaches farther than the fog does**: the map's quad is `meshHalfX * L` either side of the point looked at in
/// x and, with the copies across the wrap in y, `1.5 L` in y; the fog is 99.8% there or sooner, so no edge of the mesh is ever
/// visible, whatever the tilt and the heading. The eye is kept over the clear box (`eyeReach`, `WorldPose.maxDistance`), so the fog
/// never covers the floor under the eye.
///
/// The haze is a colour between the sea and the clear colour, scaled by the brightness, and the sky above the horizon runs from it
/// (at the horizon) to the clear colour, with the elevation: `exp(-elevation / skyScale)`.
package enum WorldFog {
    /// The half extent of the clear box, in fractions of `L`.
    package static let clear: Float = 0.5
    /// The scale of the fade, in fractions of `L`.
    package static let width: Float = 0.2
    /// The floor farther than this fraction of `L` from the point looked at (Chebyshev) is not read: the pointer there has no
    /// coordinates, no cell and no place for a power (the haze is 43% there), and a drag that takes hold there does not move the map.
    package static let readReach: Float = 0.65
    /// The eye's horizontal offset from the point looked at, `distance * sin(tilt)`, is at most this fraction of `L`: the eye is
    /// over the clear box (at 0.45 L it is 0.05 L inside the edge for the least favourable heading).
    package static let eyeReach: Float = 0.45
    /// The map's mesh is this many `L` either side of the point looked at in x (the texture repeats, so it is seamless).
    package static let meshHalfX: Float = 1.5
    /// The haze above the horizon is `exp(-elevation / skyScale)`; the elevation is in radians.
    package static let skyScale: Float = 0.22
    /// The haze: the mean of the sea (cold, warm) and the clear colour at 35%, so that the horizon is a dusk of the same water.
    package static let hazeColour = SIMD3<Float>(0.071, 0.229, 0.332)

    /// The Chebyshev length of an offset.
    package static func chebyshev(_ offset: SIMD2<Float>) -> Float { max(abs(offset.x), abs(offset.y)) }

    /// The fog at a floor point of Chebyshev distance `c` from the point looked at, in a world of `size`: 0...1.
    package static func amount(c: Float, size: Float) -> Float {
        guard size > 0 else { return 0 }
        let x = max(c - clear * size, 0) / (width * size)
        return 1 - exp(-x * x)
    }

    /// The same, for a floor point at `point` with the point looked at at `centre`.
    package static func amount(at point: SIMD2<Float>, centre: SIMD2<Float>, size: Float) -> Float {
        amount(c: chebyshev(point - centre), size: size)
    }

    /// The haze as the world pass writes it: the haze scaled by the map's brightness, mixed towards the clear colour at the map's
    /// opacity (the ground is, so the haze and the sky agree with it).
    package static func colour(exposure: Float, opacity: Float, background: SIMD3<Float> = MapPalette.background) -> SIMD3<Float> {
        let lit = simd_clamp(hazeColour * exposure, SIMD3(repeating: 0), SIMD3(repeating: 1))
        return background + (lit - background) * opacity
    }

    /// The sky's haze at an elevation (radians above the horizon) of the ray: 1 at and below the horizon, falling away above it.
    package static func skyHaze(elevation: Float) -> Float {
        elevation <= 0 ? 1 : exp(-elevation / skyScale)
    }

    /// Whether the floor point `offset` from the point looked at is near enough to be read (a coordinate, a cell, a place for a power).
    package static func isReadable(offset: SIMD2<Float>, size: Float) -> Bool {
        chebyshev(offset) <= readReach * size
    }
}
