import Foundation
import simd

/// The periodic floor of the world: x and y are periodic on `[0, L)`, z is not (DESIGN §33.4.6). Everything the World window
/// draws or picks is placed at the *minimum image* of the camera target: the image of the point nearest the target. The
/// same function runs in the shaders (`rint`, ties to even) and here.
package enum Torus {
    /// `v` wrapped into `[0, size)`.
    package static func wrap(_ v: Float, size: Float) -> Float {
        let w = v - size * (v / size).rounded(.down)
        return w >= size ? 0 : w  // (a tiny negative value wraps to size in float32)
    }

    package static func wrap(_ p: SIMD2<Float>, size: Float) -> SIMD2<Float> {
        SIMD2(wrap(p.x, size: size), wrap(p.y, size: size))
    }

    /// The offset `d` brought into `[-size/2, size/2]` (ties to even, as the shaders' `rint`).
    package static func wrapDelta(_ d: Float, size: Float) -> Float {
        d - size * (d / size).rounded(.toNearestOrEven)
    }

    package static func wrapDelta(_ d: SIMD2<Float>, size: Float) -> SIMD2<Float> {
        SIMD2(wrapDelta(d.x, size: size), wrapDelta(d.y, size: size))
    }

    /// The image of `point` nearest `reference`.
    package static func nearestImage(of point: SIMD2<Float>, to reference: SIMD2<Float>, size: Float) -> SIMD2<Float> {
        reference + wrapDelta(point - reference, size: size)
    }

    /// The image of `point` nearest `reference` in x and y (z is not periodic).
    package static func nearestImage(of point: SIMD3<Float>, to reference: SIMD3<Float>, size: Float) -> SIMD3<Float> {
        let xy = nearestImage(of: SIMD2(point.x, point.y), to: SIMD2(reference.x, reference.y), size: size)
        return SIMD3(xy.x, xy.y, point.z)
    }
}

extension Camera {
    /// Keeps the target's x and y in `[0, size)`. The view is the same: everything is drawn at the images nearest the target.
    package mutating func wrapTarget(size: Float) {
        target = SIMD3(Torus.wrap(target.x, size: size), Torus.wrap(target.y, size: size), target.z)
    }
}
