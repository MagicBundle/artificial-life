import CoreGraphics
import Foundation
import simd

/// Where the World window looks, and how (DESIGN §36.5): the point of the floor at the middle of the view, its height, the
/// `heading` (the map turned about the view axis, 0 = north, +y, up), the `tilt` from straight down and the `distance`. It is the
/// state of the view; `Camera` is what is derived from it for the renderer, the projector and the picker.
///
/// `x` and `y` are the engine's torus coordinates, in `[0, L)`. `ybar` (the signed distance from the equator, `y = 0`, in
/// `[-L/2, L/2)`) is a readout: the equator is where the weather is, and the wrap in `y` at `L/2` is the world's seam (the poles
/// of the globe of PR B).
///
/// Everything here is exact on the plane: `grab` puts the floor point that was under the pointer back under it for any pointer,
/// heading and tilt, and `zoom(by:about:)` is a change of `distance` followed by the same grab. The clamps are the footprint's:
/// one world is ever drawn (`Patch`), so the view may not reach past `reach * L` of its centre in either axis for any heading.
package struct WorldPose: Equatable, Sendable {
    package var x: Float
    package var y: Float
    /// The height of the point looked at (the followed organism's), in `[0, H]`.
    package var z: Float
    /// The map turned about the view axis, in `[-pi, pi)`; 0 has north (+y) up and east (+x) to the right.
    package var heading: Float
    /// From looking straight down, in `[0, maxTilt]`.
    package var tilt: Float
    /// From the point looked at back to the eye, in world units.
    package var distance: Float
    package var fovY: Float
    /// A 2D world is seen from below with y down (`Camera.topDown`): the same pose, the other way up.
    package var faceDown: Bool

    package init(x: Float, y: Float, z: Float = 0, heading: Float = 0, tilt: Float = 0, distance: Float,
                 fovY: Float = Camera.defaultFovY, faceDown: Bool = false) {
        self.x = x
        self.y = y
        self.z = z
        self.heading = heading
        self.tilt = tilt
        self.distance = distance
        self.fovY = fovY
        self.faceDown = faceDown
    }

    /// The fraction of `L` the view may reach from its centre, along either axis, for any heading: the one-world patch is
    /// `L` wide and the margin keeps its edge and the seam's neighbours out of the picture.
    package static let reach: Float = 0.49
    /// The turn of a heading or a tilt drag, radians per point (the Lab's `radiansPerPixel`).
    package static let radiansPerPoint: Float = 0.008
    /// The steepest the view may be tilted, before the footprint's limit.
    package static let maxTilt: Float = 85 * .pi / 180

    // MARK: Readouts

    /// The signed distance from the equator, in `[-L/2, L/2)`: positive in the hemisphere with f > 0 ("northern").
    package func ybar(size: Float) -> Float { Torus.wrapDelta(y, size: size) }

    /// The hemisphere of the centre: 0 for `0 < y < L/2` (f > 0), else 1 (the engine's `hemi`).
    package func hemisphere(size: Float) -> Int { Self.hemisphere(y: y, size: size) }

    package static func hemisphere(y: Float, size: Float) -> Int {
        let w = Torus.wrap(y, size: size)
        return w > 0 && w < size / 2 ? 0 : 1
    }

    /// The height of the eye over the floor, in world units: `camera.position.z` (the point looked at is at `z`, the eye is further up
    /// by `distance` tilted; not the distance itself). For a world seen from below (2D) it is the distance to the plane.
    package func eyeHeight(size: Float, height: Float) -> Float {
        faceDown ? distance : camera(size: size, height: height).position.z
    }

    // MARK: The camera

    /// The camera of this view on a world of `size` x `size` x `height`: the target at the centre, looking straight down
    /// (or up, for a 2D world) turned by `heading` and tilted by `tilt`. Its clip planes hold the one-world patch.
    package func camera(size: Float, height: Float) -> Camera {
        let pitch = faceDown ? Float.pi / 2 - tilt : -Float.pi / 2 + tilt
        return Camera(target: SIMD3(x, y, z), distance: distance, yaw: .pi / 2 + heading, pitch: pitch, fovY: fovY, near: 0.1,
                      far: 8 * max(size, height, 1), scene: Camera.patchScene(center: SIMD2(x, y), size: size, height: height))
    }

    /// Takes the pose of `camera` (the target, the distance, the heading and the tilt). The inverse of `camera(size:height:)`.
    package mutating func apply(_ camera: Camera) {
        x = camera.target.x
        y = camera.target.y
        z = camera.target.z
        distance = camera.distance
        fovY = camera.fovY
        heading = Self.wrap(camera.yaw - .pi / 2)
        tilt = max(faceDown ? Float.pi / 2 - camera.pitch : camera.pitch + .pi / 2, 0)
    }

    package static func wrap(_ angle: Float) -> Float {
        let turn = 2 * Float.pi
        var a = angle.truncatingRemainder(dividingBy: turn)
        if a >= .pi { a -= turn }
        if a < -.pi { a += turn }
        return a
    }

    // MARK: The footprint and the clamps

    /// The farthest the eye may be from the point looked at, straight down (tilt 0), at `heading` and height `z`: the
    /// footprint of the view on the floor, a rectangle turned by the heading, stays within `reach * L` of the centre along both
    /// axes. The eye is `d + z` over the floor, so the floor spans `(d + z) tan(fov/2)` per half height. Never under
    /// `WorldLevels.minDistance`.
    package static func maxDistance(size: Float, fovY: Float, aspect: Float, heading: Float = 0, z: Float = 0) -> Float {
        let t = tan(fovY / 2)
        let a = max(aspect, 1e-3)
        let c = abs(cos(heading)), s = abs(sin(heading))
        let m = max(a * c + s, a * s + c)  // the larger half extent of the turned rectangle, in half heights
        return max(Patch.reach(size: size) / (t * m) - max(z, 0), WorldLevels.minDistance)
    }

    /// The most `distance` this view may have now.
    package func distanceLimit(size: Float, aspect: Float) -> Float {
        Self.maxDistance(size: size, fovY: fovY, aspect: aspect, heading: heading, z: z)
    }

    /// The largest offset from the centre, along x and along y, of the floor points the four corners of the view see: how far
    /// the picture reaches in the world's axes. Nil when a corner ray does not meet the floor in front of the eye (the horizon
    /// is in view): the picture then has no edge to bound.
    package func footprint(aspect: Float, size: Float, height: Float) -> SIMD2<Float>? {
        let camera = camera(size: size, height: height)
        let half = tan(fovY / 2)
        let eye = camera.position
        var extent = SIMD2<Float>.zero
        for (sx, sy) in [(-1, 1), (1, 1), (-1, -1), (1, -1)] as [(Float, Float)] {
            let ray = camera.forward + (sx * aspect * half) * camera.right + (sy * half) * camera.up
            guard abs(ray.z) > 1e-6 else { return nil }
            let t = -eye.z / ray.z
            guard t > 0, t.isFinite else { return nil }
            let hit = eye + t * ray
            extent = max(extent, SIMD2(abs(hit.x - x), abs(hit.y - y)))
        }
        return extent
    }

    /// Whether the picture stays within one world of its centre.
    package func fits(aspect: Float, size: Float, height: Float) -> Bool {
        guard let extent = footprint(aspect: aspect, size: size, height: height) else { return false }
        let limit = Patch.reach(size: size) * 1.0005
        return extent.x <= limit && extent.y <= limit
    }

    /// Tilts the view back towards straight down until the picture stays within one world (the footprint's limit, which
    /// replaces v1's tiled-quad reach): the largest tilt that fits, to a thousandth of a degree. The distance must already be
    /// within `distanceLimit`, which every tilt-0 view meets. True if the tilt changed.
    @discardableResult
    package mutating func limitTilt(aspect: Float, size: Float, height: Float) -> Bool {
        tilt = min(max(tilt, 0), Self.maxTilt)
        guard tilt > 0, !fits(aspect: aspect, size: size, height: height) else { return false }
        var low: Float = 0, high = tilt
        var probe = self
        for _ in 0..<28 {
            let mid = 0.5 * (low + high)
            probe.tilt = mid
            if probe.fits(aspect: aspect, size: size, height: height) { low = mid } else { high = mid }
        }
        tilt = low
        return true
    }

    /// The clamps of every edit: `x` and `y` into the world, `z` into `[0, H]`, the heading into `[-pi, pi)`, the distance
    /// between a few agent radii and the footprint's limit, then the tilt.
    package mutating func clamp(aspect: Float, size: Float, height: Float) {
        if size > 0 {
            x = Torus.wrap(x.isFinite ? x : 0, size: size)
            y = Torus.wrap(y.isFinite ? y : 0, size: size)
        }
        z = min(max(z.isFinite ? z : 0, 0), max(height, 0))
        heading = Self.wrap(heading.isFinite ? heading : 0)
        let limit = distanceLimit(size: size, aspect: aspect)
        distance = min(max(distance.isFinite ? distance : limit, WorldLevels.minDistance), max(limit, WorldLevels.minDistance))
        limitTilt(aspect: aspect, size: size, height: height)
    }

    // MARK: Grab, pan, zoom, turn

    /// The pointer as the normalised device coordinates of the view (`point` in points, origin at the top left).
    package static func ndc(of point: CGPoint, in view: CGSize) -> SIMD2<Float> {
        guard view.width >= 1, view.height >= 1 else { return .zero }
        return SIMD2(Float(point.x / view.width) * 2 - 1, 1 - Float(point.y / view.height) * 2)
    }

    /// Where the ray through `ndc` meets the plane through the centre (height `z`; or the horizontal plane at `planeZ`, the floor
    /// for the minimap), as an offset from the centre. It does not depend on where the centre is, only on the orientation and the
    /// distance: the camera is translated rigidly. Nil when the ray does not reach the plane in front of the eye.
    package func offset(at ndc: SIMD2<Float>, aspect: Float, planeZ: Float? = nil) -> SIMD2<Float>? {
        var origin = self
        origin.x = 0
        origin.y = 0
        let camera = origin.camera(size: 1, height: 0)
        let half = tan(fovY / 2)
        let ray = simd_normalize(camera.forward + camera.right * (ndc.x * half * aspect) + camera.up * (ndc.y * half))
        guard abs(ray.z) > 1e-6 else { return nil }
        let t = ((planeZ ?? z) - camera.position.z) / ray.z
        guard t > 0, t.isFinite else { return nil }
        let hit = camera.position + ray * t
        return SIMD2(hit.x, hit.y)
    }

    /// The point of the plane through the centre that is under the pointer, wrapped into the world. Nil off the plane.
    package func point(at ndc: SIMD2<Float>, aspect: Float, size: Float) -> SIMD2<Float>? {
        offset(at: ndc, aspect: aspect).map { Torus.wrap(SIMD2(x, y) + $0, size: size) }
    }

    /// Moves the centre so that `anchor` (a point of the plane, as `point(at:)` named it when the drag began) is under the
    /// pointer at `ndc`. Exact for any pointer, heading and tilt. False when the pointer is off the plane.
    @discardableResult
    package mutating func grab(anchor: SIMD2<Float>, at ndc: SIMD2<Float>, aspect: Float, size: Float) -> Bool {
        guard let offset = offset(at: ndc, aspect: aspect) else { return false }
        x = Torus.wrap(anchor.x - offset.x, size: size)
        y = Torus.wrap(anchor.y - offset.y, size: size)
        return true
    }

    /// A pan by `dx`, `dy` points of a view `view` points across (+y down): the map follows the fingers, as a grab of the
    /// centre carried to the pointer's displacement.
    package mutating func pan(dx: Float, dy: Float, view: CGSize, aspect: Float, size: Float) {
        guard view.width >= 1, view.height >= 1 else { return }
        let ndc = SIMD2(2 * dx / Float(view.width), -2 * dy / Float(view.height))
        grab(anchor: SIMD2(x, y), at: ndc, aspect: aspect, size: size)
    }

    /// Multiplies the distance by `factor` (below 1 comes closer; the result is clamped to the footprint's limit) and moves the
    /// centre so that the point that was under the pointer at `ndc` is under it again. Without a pointer on the plane (nil, or
    /// off it) the zoom is about the centre. Exact for any pointer when the pointer's ray meets the plane after the zoom: the tilt
    /// is limited first (zooming out can ask for a steeper view, and the limit does not depend on where the centre is) and the
    /// grab is made last, with the orientation the view ends with. The distance itself may be clamped (the point then moves with the
    /// clamp, as a grab at the new distance puts it back under the pointer).
    package mutating func zoom(by factor: Float, at ndc: SIMD2<Float>?, aspect: Float, size: Float, height: Float) {
        guard factor > 0, factor.isFinite else { return }
        let anchor = ndc.flatMap { point(at: $0, aspect: aspect, size: size) }
        let limit = distanceLimit(size: size, aspect: aspect)
        distance = min(max(distance * factor, WorldLevels.minDistance), max(limit, WorldLevels.minDistance))
        limitTilt(aspect: aspect, size: size, height: height)
        if let anchor, let ndc { grab(anchor: anchor, at: ndc, aspect: aspect, size: size) }
    }

    /// A heading and tilt drag of `dx`, `dy` points (+y down): dragging right turns the view towards the right, dragging up
    /// tilts it (as v1's orbit did).
    package mutating func turn(dx: Float, dy: Float) {
        heading = Self.wrap(heading - dx * Self.radiansPerPoint)
        tilt += (faceDown ? dy : -dy) * Self.radiansPerPoint
        tilt = min(max(tilt, 0), Self.maxTilt)
    }

    // MARK: Home

    /// The flat overview: the whole width of one world, north up, straight down, at the equator (`y = 0`, where the weather is; the
    /// middle of a 2D world), facing `populatedX` when the caller knows where the organisms are. It is the farthest the view may be.
    package static func home(for header: StreamHeader, aspect: Float, populatedX: Float? = nil,
                             fovY: Float = Camera.defaultFovY) -> WorldPose {
        let size = Float(header.size)
        let faceDown = header.dims != 3
        var view = WorldPose(x: Torus.wrap(populatedX ?? size / 2, size: size), y: faceDown ? size / 2 : 0, distance: 1,
                             fovY: fovY, faceDown: faceDown)
        view.distance = view.distanceLimit(size: size, aspect: aspect)
        return view
    }

    /// The circular mean of the `x` of the listed rows (the side of the world the organisms are on); nil for no rows.
    package static func populatedX(frame: PlayFrame, size: Float) -> Float? {
        guard frame.n > 0, size > 0 else { return nil }
        let k = 2 * Float.pi / size
        var s: Float = 0, c: Float = 0
        for i in 0..<frame.n {
            let a = frame.position(i).x * k
            s += sin(a)
            c += cos(a)
        }
        guard s != 0 || c != 0 else { return nil }
        return Torus.wrap(atan2(s, c) / k, size: size)
    }
}

/// The one-world patch of the weather map that replaces v1's quad of 5 x 5 worlds (§36.5): at most one world is ever drawn,
/// centred on the view in x, so no storm is drawn twice.
package enum Patch {
    /// How far from its centre the view may reach in either axis, in world units.
    package static func reach(size: Float) -> Float { WorldPose.reach * size }
}

extension Camera {
    /// The scene a camera on the one-world patch is fitted to: the patch is `L` wide around the target, the view reaches no
    /// further than `WorldPose.reach * L` from it along either axis, and the agents stand up to `height` over the floor.
    package static func patchScene(center: SIMD2<Float>, size: Float, height: Float) -> Scene {
        Scene(center: SIMD3(center.x, center.y, height / 2),
              radius: (0.5 * size * Float(2).squareRoot() + 0.5 * height) * 1.05)
    }
}
