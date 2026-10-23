import Foundation
import simd

/// Whether organisms are drawn as symbols whose size is capped (the default) or at their physical size (DESIGN §36.7, decision D3).
package enum OrganismSizeMode: Int, CaseIterable, Sendable {
    /// The radius is capped by coverage and by an absolute size that falls as the camera rises, so single organisms stay separable.
    case capped
    /// The organism's radius, as the camera sees it: v1's picture (a carpet of overlapping spheres at the region level).
    case physical
}

/// How big an organism is drawn (DESIGN §36.7). Pure: the shaders take the numbers as uniforms and `WorldModel` counts the rows.
///
/// For a view of area `A` pixels with `N` rows inside it, the physical radius `r_phys = radius * pixelsPerUnit(depth)` is kept
/// where it is small and capped elsewhere:
///
/// ```text
/// r_cover  = sqrt(c A / (pi N))           c = 0.35: the most the organisms may cover if they do not overlap
/// r_abs(d) = 4.5 pt                       for d >= 0.20 L
///          = r_phys(floor) (0.08 L / d)^2  for d <= 0.08 L, which grows faster than r_phys and so never binds on the floor;
///            between, log-linear in d from 4.5 pt to r_phys(floor) at 0.08 L
/// r_draw   = clamp(r_phys, r_min, max(r_min, min(r_cover, r_abs(d))))        r_min = min(2.5 px, max(r_cover, 1.5 px))
/// ```
///
/// Above the organism level the sizes are symbols, not physical sizes: the picture is separable (at least 80% of the organisms in
/// view keep half their disc visible, where physical sizes leave 21% to 30%) and the price is that what is drawn is not the size
/// of the organism until the camera is within 0.08 L. Pixels are the drawable's.
package enum OrganismSizePolicy {
    /// The most of the view the organisms may cover.
    package static let coverage: Float = 0.35
    /// The least radius of a capped organism, in drawable pixels, where the coverage rule does not ask for less
    /// (`minimumRadius(cover:)`).
    package static let minimumRadius: Float = 2.5
    /// The least radius of an organism at its physical size (v1's smallest disc, `WorldLevels.minDiscPixels`).
    package static let physicalMinimumRadius: Float = 1.2
    /// The absolute cap far from the organisms, in points (4.5 pt is 9 px on a Retina drawable).
    package static let farRadiusPoints: Float = 4.5
    /// The camera distance, as a fraction of `L`, from which the absolute cap is `farRadiusPoints`.
    package static let farDistance: Float = 0.20
    /// The camera distance, as a fraction of `L`, at which the absolute cap meets the physical radius of an organism on the floor (the organism level's fly-to); closer it grows faster than that radius.
    package static let nearDistance: Float = 0.08
    /// The instance alpha is a ramp of the drawn radius, in pixels: nothing at `rampLow`, opaque from `rampHigh`.
    package static let rampLow: Float = 1.5
    package static let rampHigh: Float = 3.5
    /// The rows in view are counted at most this often (seconds), and the radius they give follows with this time constant.
    package static let countPeriod: Double = 0.1
    package static let smoothing: Float = 0.25

    /// The radius, in pixels, at which `rows` organisms cover `coverage` of `area` square pixels when they do not overlap;
    /// unbounded for no rows. `area` and `rows` are those of the visible footprint (the whole view in the flat regime).
    package static func coverRadius(area: Float, rows: Int) -> Float {
        guard rows > 0 else { return .infinity }
        return (coverage * max(area, 0) / (Float.pi * Float(rows))).squareRoot()
    }

    /// The least radius for a coverage radius `cover`: `minimumRadius`, but not above the coverage radius itself (so the carpet of
    /// a crowded small window does not return: review of A0, finding 7), and not below `rampLow`, where the alpha ramp has already
    /// taken the organism to nothing.
    package static func minimumRadius(cover: Float) -> Float {
        guard cover.isFinite else { return minimumRadius }
        return min(minimumRadius, max(cover, rampLow))
    }

    /// The pixels per world unit of something `depth` units in front of the eye, in a view `viewHeight` pixels high.
    package static func pixelsPerUnit(depth: Float, viewHeight: Float, fovY: Float) -> Float {
        0.5 * viewHeight / (max(depth, 1e-6) * tan(fovY / 2))
    }

    /// The absolute cap at camera distance `distance` (a world of size `size`), in pixels: `farRadiusPoints` far away, the
    /// physical radius of an organism at the floor's depth at `nearDistance * size` at the organism level's edge, log-linear in
    /// the distance between, and closer that radius times `(nearDistance * size / distance)^2` (it grows faster than any physical
    /// radius, so it does not bind there, but it is continuous: no step at `nearDistance * size`). Continuous at every distance.
    package static func absoluteRadius(distance: Float, size: Float, agentRadius: Float, viewHeight: Float, fovY: Float,
                                       pixelsPerPoint: Float) -> Float {
        let far = farRadiusPoints * max(pixelsPerPoint, 0.1)
        let farAt = farDistance * size, nearAt = nearDistance * size
        guard farAt > nearAt, nearAt > 0 else { return far }
        if distance >= farAt { return far }
        let near = agentRadius * pixelsPerUnit(depth: nearAt, viewHeight: viewHeight, fovY: fovY)
        guard near.isFinite, near > 0 else { return far }
        if distance <= nearAt {
            // Closer than the organism level's edge the cap grows with the cube of the zoom: above the physical radius of an
            // organism on the floor (which grows with its square), so it never binds a row on the floor, yet it is a continuation
            // of the blend, not a switch: a row above the floor (its physical radius above the floor's) is not held at `near`
            // just above `nearAt` and released just below it (review of A0, finding 1).
            let d = max(distance, 1e-3)
            return agentRadius * pixelsPerUnit(depth: d, viewHeight: viewHeight, fovY: fovY) * (nearAt / d) * (nearAt / d)
        }
        let t = log(farAt / distance) / log(farAt / nearAt)
        return exp(log(far) + t * (log(near) - log(far)))
    }

    /// The radius an organism is drawn at: its physical one, at least `minimum`, at most the larger of `cap` and `minimum`.
    package static func radius(physical: Float, cap: Float, minimum: Float) -> Float {
        min(max(physical, minimum), max(cap, minimum))
    }

    /// The opacity of an instance as a ramp of its drawn radius, in pixels (`smoothstep(1.5, 3.5, r)`): no level or distance
    /// switch, so organisms fade in as the picture gets to where they can be seen. The globe's glow takes `1 - alpha`.
    package static func instanceAlpha(radius: Float) -> Float {
        let t = min(max((radius - rampLow) / (rampHigh - rampLow), 0), 1)
        return t * t * (3 - 2 * t)
    }

    /// What the shaders and the picker are given for `mode`: the cap on the drawn radius (infinite at the physical size) and the
    /// least radius, in pixels. `cover` is the (smoothed) `coverRadius` of the footprint.
    package static func limits(mode: OrganismSizeMode, cover: Float, distance: Float, size: Float, agentRadius: Float,
                               viewHeight: Float, fovY: Float, pixelsPerPoint: Float) -> (cap: Float, minimum: Float) {
        switch mode {
        case .physical:
            return (.infinity, physicalMinimumRadius)
        case .capped:
            let absolute = absoluteRadius(distance: distance, size: size, agentRadius: agentRadius, viewHeight: viewHeight,
                                          fovY: fovY, pixelsPerPoint: pixelsPerPoint)
            return (min(cover, absolute), minimumRadius(cover: cover))
        }
    }

    /// The number of listed rows whose centre is in view, each at its minimum image about the camera target: the `N` of the
    /// coverage rule in the flat regime. A pass over the positions (about 0.1 ms for 42,000 rows).
    package static func rowsInView(frame: PlayFrame, camera: Camera, aspect: Float, size: Float) -> Int {
        let m = camera.viewProjectionMatrix(aspect: aspect)
        // The rows of the matrix that give clip x, y and w.
        let rx = SIMD4(m.columns.0.x, m.columns.1.x, m.columns.2.x, m.columns.3.x)
        let ry = SIMD4(m.columns.0.y, m.columns.1.y, m.columns.2.y, m.columns.3.y)
        let rw = SIMD4(m.columns.0.w, m.columns.1.w, m.columns.2.w, m.columns.3.w)
        let tx = camera.target.x, ty = camera.target.y
        var count = 0
        frame.pos.withUnsafeBufferPointer { pos in
            for i in 0..<min(frame.n, pos.count / 3) {
                let px = pos[3 * i], py = pos[3 * i + 1], pz = pos[3 * i + 2]
                let dx = px - tx, dy = py - ty
                let x = tx + dx - size * (dx / size).rounded(.toNearestOrEven)
                let y = ty + dy - size * (dy / size).rounded(.toNearestOrEven)
                let p = SIMD4(x, y, pz, 1)
                let w = simd_dot(rw, p)
                guard w > 0 else { continue }
                if abs(simd_dot(rx, p)) <= w, abs(simd_dot(ry, p)) <= w { count += 1 }
            }
        }
        return count
    }

    /// One step of the one-pole filter that takes the cover radius from `current` to `target` over `dt` seconds (time
    /// constant `smoothing`); the target itself when either is not finite (the first count, or an empty view).
    package static func smoothed(_ current: Float, toward target: Float, dt: Float) -> Float {
        guard current.isFinite, target.isFinite else { return target }
        let k = 1 - exp(-max(dt, 0) / smoothing)
        return current + (target - current) * k
    }
}
