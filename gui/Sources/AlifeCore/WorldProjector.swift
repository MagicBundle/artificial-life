import CoreGraphics
import Foundation
import simd

/// A storm as the overlay draws it, in the view's points (origin at the top left).
package struct StormGlyph: Equatable, Sendable {
    package var id: Int
    package var name: String?
    package var named: Bool
    package var hemisphere: Int
    package var center: CGPoint
    /// The ring's radius: the storm's `r` in points, never under 5.
    package var radius: CGFloat
    /// The track, continuous across the seam, ending at `center`.
    package var track: [CGPoint]
    /// The ring is a circle on the floor: its height over its width on the screen, 1 looking straight down and towards 0 at the
    /// horizon (never under 0.12).
    package var flatten: CGFloat = 1
    package var zeta: Double
    package var coasting: Bool
}

/// World to view for the `Canvas` overlay and the picker: the camera's projection onto the free rectangle, with every point
/// at the minimum image of the camera target (DESIGN §33.4.6). Pure, in the view's points: the view's size is the free
/// rectangle, so the overlay, the Metal view and the picker agree on one rectangle.
package struct WorldProjector: Equatable, Sendable {
    package var camera: Camera
    /// `L`.
    package var size: Float
    /// The free rectangle's size, in points.
    package var view: CGSize
    /// Whether a weather map is in view (the planet level exists only then, `WorldLevels.level`).
    package var mapShown: Bool
    /// The view as the globe's (nil: a camera only, the flat map at every distance). With it, a projector at `s > 0` projects and picks
    /// through the morph (`WorldVantage`); at `s = 0` it is the flat camera's, byte for byte.
    package var vantage: WorldVantage?

    package init(camera: Camera, size: Float, view: CGSize, mapShown: Bool = true) {
        self.camera = camera
        self.size = size
        self.view = view
        self.mapShown = mapShown
    }

    /// The projector of a pose on a world of `size` x `size` x `height`; `globe` is whether the globe exists (`s` then follows the
    /// distance), else the view is the flat map.
    package init(pose: WorldPose, size: Float, height: Float, view: CGSize, mapShown: Bool = true, globe: Bool = false) {
        self.camera = pose.camera(size: size, height: height)
        self.size = size
        self.view = view
        self.mapShown = mapShown
        let aspect = view.height >= 1 ? Float(view.width / view.height) : 1
        let vantage = WorldVantage(pose: pose, size: size, height: height, aspect: aspect, globe: globe && mapShown)
        self.vantage = vantage.isGlobe ? vantage : nil
    }

    /// The morph scalar: 0 for the flat map.
    package var s: Float { vantage?.s ?? 0 }

    /// The view is on the globe or between (the camera's flat projection no longer applies).
    package var isMorphing: Bool { s > 0 }

    package var aspect: Float { view.height >= 1 ? Float(view.width / view.height) : 1 }

    package var level: WorldLevel {
        if let vantage {
            // With the globe the derived level: organism under 0.12 L, planet from s = 0.5, else region.
            if camera.distance < WorldLevels.organismDistance * size { return .organism }
            return vantage.s >= 0.5 ? .planet : .region
        }
        return WorldLevels.level(distance: camera.distance, fovY: camera.fovY, aspect: aspect, size: size, mapShown: mapShown)
    }

    /// The image of a world point nearest the camera target.
    package func image(of point: SIMD3<Float>) -> SIMD3<Float> {
        Torus.nearestImage(of: point, to: camera.target, size: size)
    }

    /// The point in the view where `world` (at its minimum image) is drawn; nil behind the eye.
    package func screen(_ world: SIMD3<Float>) -> CGPoint? {
        if isMorphing, let vantage { return vantage.screen(world, in: view) }
        guard let ndc = camera.ndc(of: image(of: world), aspect: aspect) else { return nil }
        return CGPoint(x: CGFloat(ndc.x * 0.5 + 0.5) * view.width, y: CGFloat(0.5 - ndc.y * 0.5) * view.height)
    }

    /// The point of the plane at height `z` that is under `point` of the view (origin at the top left), wrapped into the world
    /// `[0, L)^2`. Nil when the ray from the eye through the point does not reach the plane in front of it (a view at the horizon,
    /// or a plane behind the eye), and nil where the plane is read no longer: farther than `WorldFog.readReach` of the point looked
    /// at, in the haze (a point of the screen toward the horizon is a world away; it has no coordinates and takes no power). The
    /// inverse of `screen` on that plane, up to the wrap: it is how a click on the map names a place for a power.
    package func world(at point: CGPoint, height z: Float = 0) -> SIMD2<Float>? {
        guard view.width >= 1, view.height >= 1 else { return nil }
        if isMorphing, let vantage {
            guard let hit = vantage.pick(at: WorldPose.ndc(of: point, in: view), height: z) else { return nil }
            return SIMD2(hit.x, hit.y)
        }
        let nx = Float(point.x / view.width) * 2 - 1
        let ny = 1 - Float(point.y / view.height) * 2
        let half = tan(camera.fovY / 2)
        let ray = simd_normalize(camera.forward + camera.right * (nx * half * aspect) + camera.up * (ny * half))
        guard abs(ray.z) > 1e-6 else { return nil }
        let t = (z - camera.position.z) / ray.z
        guard t > 0 else { return nil }
        let hit = camera.position + ray * t
        guard size > 0 else { return SIMD2(hit.x, hit.y) }
        guard WorldFog.isReadable(offset: SIMD2(hit.x - camera.target.x, hit.y - camera.target.y), size: size) else { return nil }
        func wrapped(_ v: Float) -> Float {
            let r = v - size * (v / size).rounded(.down)
            return r >= size ? 0 : r
        }
        return SIMD2(wrapped(hit.x), wrapped(hit.y))
    }

    /// The length in points of `units` world units across the view at `world`.
    package func points(ofUnits units: Float, at world: SIMD3<Float>) -> CGFloat? {
        if isMorphing, let vantage {
            // The circle of one unit on the surface is an ellipse; the screen-right length is its major axis.
            let o = vantage.offsets(x: world.x, y: world.y)
            return vantage.ellipse(e: o.e, n: o.n, z: world.z, in: view).map { CGFloat($0.major * units) }
        }
        let at = image(of: world)
        // Both ends are placed as they are: `screen` would wrap the far end again, a world away when `at` is near the
        // half-world from the target.
        guard let a = screenOfImage(at), let b = screenOfImage(at + camera.right * units) else { return nil }
        return hypot(a.x - b.x, a.y - b.y)
    }

    /// The storms of a `STRM` record as glyphs. A storm behind the eye is left out.
    package func glyphs(of storms: PlayStorms) -> [StormGlyph] {
        storms.storms.compactMap { storm in
            let floor = SIMD3<Float>(Float(storm.x), Float(storm.y), 0)
            guard let center = screen(floor) else { return nil }
            let physical = points(ofUnits: Float(storm.r), at: floor) ?? 0
            let radius = max(physical, 5)
            return StormGlyph(id: storm.id, name: storm.name, named: storm.named, hemisphere: storm.hemi, center: center,
                              radius: radius, track: track(storm), flatten: physical > 5 ? flatten(at: floor) : 1,
                              zeta: storm.zeta, coasting: storm.state == "coasting")
        }
    }

    /// The height over the width, on the screen, of a circle on the floor at `world`: the points between the centre and a point a unit
    /// ahead (along the heading) over those between the centre and a point a unit to the right. 1 looking straight down, towards 0 at
    /// the horizon; never under 0.12, and 1 where it cannot be told.
    package func flatten(at world: SIMD3<Float>) -> CGFloat {
        if isMorphing, let vantage {
            let o = vantage.offsets(x: world.x, y: world.y)
            guard let axes = vantage.ellipse(e: o.e, n: o.n, z: world.z, in: view), axes.major > 1e-6 else { return 1 }
            return CGFloat(min(max(axes.minor / axes.major, 0.12), 1))
        }
        let forward = SIMD2(camera.forward.x, camera.forward.y)
        let up = SIMD2(camera.up.x, camera.up.y)
        let alongFloor = simd_length(forward) > 0.05 ? forward : up
        guard simd_length(alongFloor) > 1e-4 else { return 1 }
        let ahead = simd_normalize(alongFloor)
        let at = image(of: world)
        guard let c = screenOfImage(at), let a = screenOfImage(at + SIMD3(ahead.x, ahead.y, 0)),
              let r = screenOfImage(at + camera.right) else { return 1 }
        let across = hypot(r.x - c.x, r.y - c.y), along = hypot(a.x - c.x, a.y - c.y)
        guard across > 1e-6, along.isFinite else { return 1 }
        return min(max(along / across, 0.12), 1)
    }

    /// The storm's track as points: unwrapped, so continuous across the seam; it is shifted whole by the multiple of `L`
    /// that brings the storm's current position to its minimum image, then wrapped by nothing.
    package func track(_ storm: PlayStorms.Storm) -> [CGPoint] {
        guard let last = storm.track.last, last.count >= 3 else { return [] }
        let here = Torus.nearestImage(of: SIMD2(Float(storm.x), Float(storm.y)),
                                      to: SIMD2(camera.target.x, camera.target.y), size: size)
        let shift = here - SIMD2(Float(last[1]), Float(last[2]))
        return storm.track.compactMap { entry in
            guard entry.count >= 3 else { return nil }
            return screenOfImage(SIMD3(Float(entry[1]) + shift.x, Float(entry[2]) + shift.y, 0))
        }
    }

    /// Like `screen`, for a point that is already at the image to draw (no further wrapping).
    package func screenOfImage(_ world: SIMD3<Float>) -> CGPoint? {
        if isMorphing, let vantage { return vantage.screenOfImage(world, in: view) }
        guard let ndc = camera.ndc(of: world, aspect: aspect) else { return nil }
        return CGPoint(x: CGFloat(ndc.x * 0.5 + 0.5) * view.width, y: CGFloat(0.5 - ndc.y * 0.5) * view.height)
    }

    /// The four corners in points of cell `(cx, cy)` of an `nx` x `ny` territory map, at the image of the cell nearest the
    /// camera target moved by `shift` worlds (`(1, 0)` is the copy one world to the east); nil if a corner is behind the eye.
    /// The corners are in order: south-west, south-east, north-east, north-west.
    package func cellCorners(cx: Int, cy: Int, nx: Int, ny: Int, shift: SIMD2<Int> = .zero) -> [CGPoint]? {
        guard nx > 0, ny > 0 else { return nil }
        let w = size / Float(nx), h = size / Float(ny)
        let centre = Torus.nearestImage(of: SIMD2((Float(cx) + 0.5) * w, (Float(cy) + 0.5) * h),
                                        to: SIMD2(camera.target.x, camera.target.y), size: size)
            + SIMD2(Float(shift.x), Float(shift.y)) * size
        var out: [CGPoint] = []
        for (sx, sy) in [(-1, -1), (1, -1), (1, 1), (-1, 1)] as [(Float, Float)] {
            guard let at = screenOfImage(SIMD3(centre.x + sx * w / 2, centre.y + sy * h / 2, 0)) else { return nil }
            out.append(at)
        }
        return out
    }
}
