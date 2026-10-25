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
/// heading and tilt, and `zoom(by:about:)` is a change of `distance` followed by the same grab. The clamps: looking straight down the
/// view may not reach past `reach * L` of its centre in either axis for any heading (one world is ever drawn, `Patch`); a tilted
/// view may be tilted to edge-on, and keeps its eye over the clear box instead (`maxDistance`): the far floor is hidden by the
/// horizon fade (`WorldFog`), not by a limit of the tilt.
package struct WorldPose: Equatable, Sendable {
    package var x: Float
    package var y: Float
    /// The height of the point looked at (the followed organism's), in `[0, H]`.
    package var z: Float
    /// The map turned about the view axis, in `[-pi, pi)`; 0 has north (+y) up and east (+x) to the right.
    package var heading: Float
    /// From looking straight down, in `[0, maxTilt]`: 0 is the Top view (straight down), pi/4 the Oblique one, pi/2 looks along the
    /// floor, edge-on (the camera's pitch is `tilt - pi/2`).
    package var tilt: Float
    /// From the point looked at back to the eye, in world units.
    package var distance: Float
    package var fovY: Float
    /// A 2D world is seen from below with y down (`Camera.topDown`): the same pose, the other way up.
    package var faceDown: Bool
    /// Whether `z` is the height of what the camera follows (an organism): the point looked at is then that organism, and the edge-on
    /// lift (`lift`) does not apply. The model sets it with the follow.
    package var followsHeight = false
    /// How far the point looked at is raised over `z` near edge-on (`edgeOnLift`): 0 up to a tilt of `liftFrom`, then rising smoothly
    /// to the middle of the height at edge-on, so that the eye is not on the floor with the whole of the life crushed into a line (the
    /// view from the Side preset, reached by the handle or an Option drag as well). It is a function of the tilt and `z`, kept in step
    /// by `clamp`, so it goes as the tilt is relaxed. Nothing for a followed height, a 2D world or a world with no height.
    package private(set) var lift: Float = 0
    /// The unit direction of the centre while it is inside a pole cap of the globe (`|latitude| >= phi_max`, 85.05 degrees), where
    /// `(x, y)` can only read the seam row: a grab that turns the globe over a pole needs a centre that the torus has no row for
    /// (DESIGN §36.10). Nil everywhere else, and always nil on the flat map (`clamp` drops it at `s = 0`).
    package var cap: SIMD3<Float>?

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
    /// The most the view may be tilted: edge-on, the view axis along the floor. (A1 stopped at 85 degrees and at the footprint's limit,
    /// 52 to 57 degrees at 0.12 L; the horizon fade replaced that limit.)
    package static let maxTilt: Float = .pi / 2

    /// The tilt from which the point looked at rises toward the middle of the height (`lift`): 70 degrees, the Low view's, which is
    /// untouched.
    package static let liftFrom: Float = 70 * .pi / 180

    /// The share, 0...1, of the way from the point looked at to the middle of the height that the lift has gone at `tilt`:
    /// a smoothstep from `liftFrom` to `maxTilt`.
    package static func edgeOnLift(tilt: Float) -> Float {
        let t = min(max((tilt - liftFrom) / (maxTilt - liftFrom), 0), 1)
        return t * t * (3 - 2 * t)
    }

    /// The height of the point the camera looks at: `z` and the lift. (`z` itself is the height the player asked for, or an
    /// organism's.)
    package var lookZ: Float { z + lift }

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

    // MARK: Readouts of the angle

    /// The tilt in whole degrees, 0 (straight down) to 90 (edge-on).
    package var tiltDegrees: Int { Int((tilt * 180 / .pi).rounded()) }

    /// The compass bearing the view looks towards, in whole degrees clockwise from north, 0 to 359: north up is 0, and a view turned so
    /// that east is up looks east, 90. (The heading is the map's turn, 0 north up and positive towards the west; the bearing is its
    /// opposite.)
    package var bearingDegrees: Int {
        var degrees = Int((-heading * 180 / .pi).rounded()) % 360
        if degrees < 0 { degrees += 360 }
        return degrees
    }

    // MARK: The camera

    /// The camera of this view on a world of `size` x `size` x `height`: the target at the centre, looking straight down
    /// (or up, for a 2D world) turned by `heading` and tilted by `tilt`. Its clip planes hold the one-world patch.
    package func camera(size: Float, height: Float) -> Camera {
        let pitch = faceDown ? Float.pi / 2 - tilt : -Float.pi / 2 + tilt
        return Camera(target: SIMD3(x, y, lookZ), distance: distance, yaw: .pi / 2 + heading, pitch: pitch, fovY: fovY, near: 0.1,
                      far: 8 * max(size, height, 1), scene: Camera.patchScene(center: SIMD2(x, y), size: size, height: height))
    }

    /// Takes the pose of `camera` (the target, the distance, the heading and the tilt). The inverse of `camera(size:height:)`.
    package mutating func apply(_ camera: Camera) {
        x = camera.target.x
        y = camera.target.y
        z = max(camera.target.z - lift, 0)
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

    /// The farthest the eye may be from the point looked at at `heading`, height `z` and `tilt`. Looking straight down: the footprint of
    /// the view on the floor, a rectangle turned by the heading, stays within `reach * L` of the centre along both axes. The eye is
    /// `d + z` over the floor, so the floor spans `(d + z) tan(fov/2)` per half height. Tilted: also the eye's horizontal offset from
    /// the point looked at, `d sin(tilt)`, stays within `WorldFog.eyeReach * L`, so the eye is over the clear box (the near edge of the
    /// patch and the floor under the eye are never in the haze). The two limits meet without a step (the second is no limit at a
    /// tilt of 0). Never under `WorldLevels.minDistance`.
    package static func maxDistance(size: Float, fovY: Float, aspect: Float, heading: Float = 0, z: Float = 0, tilt: Float = 0,
                                    globe: Bool = false) -> Float {
        // With the globe the farthest view is the globe at 36% of the height, whatever the tilt: no patch edge to keep out of the
        // picture, and the haze is the limb's (the camera orbits the planet).
        if globe { return max(Morph.maxDistanceFraction * size, WorldLevels.minDistance) }
        let t = tan(fovY / 2)
        let a = max(aspect, 1e-3)
        let c = abs(cos(heading)), s = abs(sin(heading))
        let m = max(a * c + s, a * s + c)  // the larger half extent of the turned rectangle, in half heights
        // (the height of the point looked at takes a view down by `z` only as far as the view is straight down: edge-on it is the eye's)
        var limit = Patch.reach(size: size) / (t * m) - max(z, 0) * cos(min(max(tilt, 0), Float.pi / 2))
        let sine = sin(min(max(tilt, 0), Float.pi / 2))
        if sine > 1e-4 { limit = min(limit, WorldFog.eyeReach * size / sine) }
        return max(limit, WorldLevels.minDistance)
    }

    /// The most `distance` this view may have now.
    package func distanceLimit(size: Float, aspect: Float, globe: Bool = false) -> Float {
        Self.maxDistance(size: size, fovY: fovY, aspect: aspect, heading: heading, z: z, tilt: tilt, globe: globe)
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

    /// The floor point (as an offset from the centre) under the pointer at `ndc`, always: where the ray meets the plane `planeZ` in
    /// front of the eye within `limit` of the centre (Chebyshev, in world units); where it misses it (the horizon is in the picture)
    /// or meets it farther, the point of its own horizontal line from the eye where that line leaves the box of `limit`. It is the
    /// footprint of a view that may be tilted to edge-on, clipped to the clear box: the part of the picture that is not haze. The eye
    /// is over the box (`WorldFog.eyeReach`), so the line does leave it.
    package func clippedOffset(at ndc: SIMD2<Float>, aspect: Float, planeZ: Float = 0, limit: Float) -> SIMD2<Float> {
        var origin = self
        origin.x = 0
        origin.y = 0
        let camera = origin.camera(size: 1, height: 0)
        let half = tan(fovY / 2)
        let ray = simd_normalize(camera.forward + camera.right * (ndc.x * half * aspect) + camera.up * (ndc.y * half))
        let eye = SIMD2(camera.position.x, camera.position.y)
        if abs(ray.z) > 1e-6 {
            let t = (planeZ - camera.position.z) / ray.z
            if t > 0, t.isFinite {
                let hit = eye + SIMD2(ray.x, ray.y) * t
                if WorldFog.chebyshev(hit) <= limit { return hit }
            }
        }
        // The line leaves the box where one of its coordinates reaches the box's side (the eye is inside; a ray that goes straight
        // up or down has no line: the eye's own place).
        let line = SIMD2(ray.x, ray.y)
        var exit = Float.infinity
        for axis in 0..<2 where abs(line[axis]) > 1e-6 {
            let s = ((line[axis] > 0 ? limit : -limit) - eye[axis]) / line[axis]
            if s >= 0 { exit = min(exit, s) }
        }
        guard exit.isFinite else { return SIMD2(min(max(eye.x, -limit), limit), min(max(eye.y, -limit), limit)) }
        return eye + line * exit
    }

    /// The clamps of every edit: `x` and `y` into the world, `z` into `[0, H]`, the heading into `[-pi, pi)`, the tilt into
    /// `[0, maxTilt]`, then the distance between a few agent radii and the limit of that tilt (`maxDistance`: a tilted view is held
    /// nearer, not tilted back).
    package mutating func clamp(aspect: Float, size: Float, height: Float, globe: Bool = false) {
        if size > 0 {
            x = Torus.wrap(x.isFinite ? x : 0, size: size)
            y = Torus.wrap(y.isFinite ? y : 0, size: size)
        }
        if !globe || faceDown { cap = nil }
        if let c = cap, !(c.x.isFinite && c.y.isFinite && c.z.isFinite) || simd_length(c) < 0.5 { cap = nil }
        z = min(max(z.isFinite ? z : 0, 0), max(height, 0))
        heading = Self.wrap(heading.isFinite ? heading : 0)
        tilt = min(max(tilt.isFinite ? tilt : 0, 0), Self.maxTilt)
        lift = followsHeight || faceDown || height <= 0 ? 0 : Self.edgeOnLift(tilt: tilt) * max(height / 2 - z, 0)
        let limit = distanceLimit(size: size, aspect: aspect, globe: globe && !faceDown)
        distance = min(max(distance.isFinite ? distance : limit, WorldLevels.minDistance), max(limit, WorldLevels.minDistance))
    }

    // MARK: Grab, pan, zoom, turn

    /// The pointer as the normalised device coordinates of the view (`point` in points, origin at the top left).
    package static func ndc(of point: CGPoint, in view: CGSize) -> SIMD2<Float> {
        guard view.width >= 1, view.height >= 1 else { return .zero }
        return SIMD2(Float(point.x / view.width) * 2 - 1, 1 - Float(point.y / view.height) * 2)
    }

    /// The tilt from which the plane a grab takes hold of is the floor, not the plane through the centre.
    package static let gripTilt: Float = 80 * .pi / 180

    /// The height of the plane the pointer takes hold of: the one through the centre (so that a followed organism's plane is
    /// grabbed), except that from `gripTilt` (a view that looks nearly along the floor, where the plane through the centre is
    /// the eye's own and is met by no ray) it is the floor.
    package var gripZ: Float { tilt < Self.gripTilt ? z : 0 }

    /// Where the ray through `ndc` meets the plane through the centre (`gripZ`; or the horizontal plane at `planeZ`, the floor
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
        let t = ((planeZ ?? gripZ) - camera.position.z) / ray.z
        guard t > 0, t.isFinite else { return nil }
        let hit = camera.position + ray * t
        return SIMD2(hit.x, hit.y)
    }

    /// The point of the grip plane that is under the pointer, wrapped into the world. Nil off the plane, and nil where the floor is not
    /// read (beyond `WorldFog.readReach`: in the haze, or toward the horizon, where a point of the screen is a world away).
    package func point(at ndc: SIMD2<Float>, aspect: Float, size: Float) -> SIMD2<Float>? {
        guard let hit = offset(at: ndc, aspect: aspect), WorldFog.isReadable(offset: hit, size: size) else { return nil }
        return Torus.wrap(SIMD2(x, y) + hit, size: size)
    }

    /// Moves the centre so that `anchor` (a point of the plane, as `point(at:)` named it when the drag began) is under the
    /// pointer at `ndc`. Exact for any pointer, heading and tilt. False when the pointer is off the plane, or in the haze (the map then
    /// stays where it is: a point of the screen that far is a world away).
    @discardableResult
    package mutating func grab(anchor: SIMD2<Float>, at ndc: SIMD2<Float>, aspect: Float, size: Float) -> Bool {
        guard let offset = offset(at: ndc, aspect: aspect), WorldFog.isReadable(offset: offset, size: size) else { return false }
        x = Torus.wrap(anchor.x - offset.x, size: size)
        y = Torus.wrap(anchor.y - offset.y, size: size)
        return true
    }

    /// A pan by `dx`, `dy` points of a view `view` points across (+y down): the map follows the fingers, as a grab of the point of
    /// the plane under the middle of the picture carried to the pointer's displacement. A view that looks nearly along the floor
    /// (from `gripTilt`) has no such point to speak of (the middle of the picture is at the horizon): it moves along the floor by
    /// the displacement at the distance of the point looked at, sideways for `dx` and along the heading for `dy`.
    package mutating func pan(dx: Float, dy: Float, view: CGSize, aspect: Float, size: Float) {
        guard view.width >= 1, view.height >= 1 else { return }
        let ndc = SIMD2(2 * dx / Float(view.width), -2 * dy / Float(view.height))
        if tilt < Self.gripTilt, let middle = offset(at: .zero, aspect: aspect) {
            grab(anchor: SIMD2(x, y) + middle, at: ndc, aspect: aspect, size: size)
        } else {
            let perPoint = 2 * distance * tan(fovY / 2) / Float(view.height)
            let right = SIMD2(cos(heading), sin(heading)), ahead = SIMD2(-sin(heading), cos(heading))
            let moved = (-dx * right + (faceDown ? -dy : dy) * ahead) * perPoint
            x = Torus.wrap(x + moved.x, size: size)
            y = Torus.wrap(y + moved.y, size: size)
        }
    }

    /// Multiplies the distance by `factor` (below 1 comes closer; the result is clamped to the limit of the tilt) and moves the
    /// centre so that the point that was under the pointer at `ndc` is under it again. Without a pointer on the plane (nil, or
    /// off it) the zoom is about the centre. Exact for any pointer when the pointer's ray meets the plane after the zoom: the grab is
    /// made last, with the orientation the view ends with. The distance itself may be clamped (the point then moves with the clamp, as a
    /// grab at the new distance puts it back under the pointer).
    package mutating func zoom(by factor: Float, at ndc: SIMD2<Float>?, aspect: Float, size: Float, height: Float) {
        guard factor > 0, factor.isFinite else { return }
        let anchor = ndc.flatMap { point(at: $0, aspect: aspect, size: size) }
        let limit = distanceLimit(size: size, aspect: aspect)
        distance = min(max(distance * factor, WorldLevels.minDistance), max(limit, WorldLevels.minDistance))
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
                             fovY: Float = Camera.defaultFovY, globe: Bool = false) -> WorldPose {
        let size = Float(header.size)
        let faceDown = header.dims != 3
        var view = WorldPose(x: Torus.wrap(populatedX ?? size / 2, size: size), y: faceDown ? size / 2 : 0, distance: 1,
                             fovY: fovY, faceDown: faceDown)
        // With the globe, Home is the globe overview: the sphere at 80% of the smaller dimension (`Morph.globeDistance`, 0.296 L).
        view.distance = globe && !faceDown ? Morph.globeDistance(L: size, fovY: fovY, aspect: aspect)
                                           : view.distanceLimit(size: size, aspect: aspect)
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
    /// The scene a camera on the one-world patch is fitted to: the map is drawn until the fog has taken it (`WorldFog`: 99.8% at `L`
    /// from the point looked at along either axis, so a corner of that box is `L sqrt 2` away), the clear box is `L` wide around the
    /// target, and the agents stand up to `height` over the floor. The far plane is behind everything the haze has not hidden, so
    /// the plane never cuts the floor in view as a line.
    package static func patchScene(center: SIMD2<Float>, size: Float, height: Float) -> Scene {
        Scene(center: SIMD3(center.x, center.y, height / 2),
              radius: (size * Float(2).squareRoot() + 0.5 * height) * 1.05)
    }
}
