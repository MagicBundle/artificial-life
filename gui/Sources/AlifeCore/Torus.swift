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

    /// The scene a camera on the torus is fitted to: the clip planes must hold the whole tiled map, which extends two and a
    /// half worlds from the cell of the target in every direction (the repeating quad of `Shaders.mapVertex`), seen from
    /// anywhere between the floor and the farthest zoom. `radius` is that extent's half diagonal, with a margin.
    package static func tiledScene(size: Float, height: Float) -> Scene {
        Scene(center: SIMD3(size / 2, size / 2, height / 2), radius: tiledHalfExtent * size * Float(2).squareRoot() * 1.02)
    }

    /// The tiled quad's half side, in worlds.
    package static let tiledHalfExtent: Float = 2.5

    /// The starting pose of the planet: the map seen from above, x to the right and y up (the way a map reads), the whole
    /// world in view in the smaller dimension of a window of `aspect`.
    package static func planet(size: Float, height: Float = 0, aspect: Float = 1, fovY: Float = Camera.defaultFovY) -> Camera {
        let halfY = atan(min(1, aspect) * tan(fovY / 2))
        let distance = size / 2 * 1.04 / tan(halfY)
        return Camera(target: SIMD3(size / 2, size / 2, 0), distance: distance, yaw: .pi / 2, pitch: -Camera.maxPitch,
                      fovY: fovY, near: 0.1, far: 8 * max(size, height), scene: tiledScene(size: size, height: height))
    }

    /// The starting pose of a world for the World window: the planet seen from above for a 3D world, and the face-on view of
    /// the 2D world (the same as the Lab's, with the clip scene of the torus).
    package static func world(for header: StreamHeader, aspect: Float = 1) -> Camera {
        var camera: Camera
        if header.dims == 3 {
            camera = planet(size: Float(header.size), height: Float(header.height), aspect: aspect)
        } else {
            camera = topDown(size: Float(header.size), aspect: aspect)
            camera.scene = tiledScene(size: Float(header.size), height: 0)
        }
        return camera
    }
}
