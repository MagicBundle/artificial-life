import Foundation
import simd

// The camera of the sphere world (DESIGN §46.8): one pose for the whole range from the planet seen whole to a hover a few body lengths
// above the floor. There is no morph and no tangent-plane chart: the camera descends, the horizon curves.
//
//   T   = Rp * up                        the floor point looked at
//   eye = T + d * (cos(tilt) * up - sin(tilt) * heading)
//   forward = -cos(tilt) * up + sin(tilt) * heading
//
// At tilt 0 the eye is straight above T looking down, with `heading` up the screen; at tilt 90 it hovers behind T and looks along `heading`.
// `up` and `heading` are unit vectors, `heading` perpendicular to `up`; keeping both (and not a longitude and a bearing) is what makes the
// pole an ordinary place: a drag turns them together by a rotation about the planet's centre, which is the parallel transport a trackball
// gives, and the compass is derived from them.

/// What the camera looks at and from where. Pure value; all rules (clamps, grabs, zooms, flights) are methods.
package struct SpherePose: Equatable, Sendable {
    /// Unit vector from the planet's centre to the point looked at.
    package var up: SIMD3<Float>
    /// Unit vector perpendicular to `up`: the way up the screen at tilt 0, the way the eye looks at tilt 90.
    package var heading: SIMD3<Float>
    /// Radians from straight down (0) to the horizon (`maxTilt`).
    package var tilt: Float
    /// Distance from the floor point looked at to the eye, in world units.
    package var distance: Float
    package var fovY: Float

    package static let defaultFovY: Float = 45 * .pi / 180
    package static let maxTilt: Float = 82 * .pi / 180
    /// The nearest the eye comes to the floor point: a few body lengths.
    package static let minDistance: Float = 3

    package init(up: SIMD3<Float>, heading: SIMD3<Float>, tilt: Float = 0, distance: Float, fovY: Float = SpherePose.defaultFovY) {
        let u = simd_normalize(up)
        var h = heading - simd_dot(heading, u) * u
        if simd_length(h) < 1e-6 { h = SphereFrame.basis(at: u).north }
        self.up = u
        self.heading = simd_normalize(h)
        self.tilt = min(max(tilt, 0), Self.maxTilt)
        self.distance = max(distance, Self.minDistance)
        self.fovY = fovY
    }

    /// Looking at latitude `lat`, longitude `lon` (degrees), north up the screen.
    package init(lat: Double, lon: Double, tilt: Float = 0, distance: Float, fovY: Float = SpherePose.defaultFovY) {
        let u = SphereFrame.direction(lat: lat, lon: lon)
        self.init(up: u, heading: SphereFrame.basis(at: u).north, tilt: tilt, distance: distance, fovY: fovY)
    }

    /// The compass: the bearing of the screen's up direction, radians from north towards east, in [-pi, pi). 0 is north up.
    package var bearing: Float {
        let b = SphereFrame.basis(at: up)
        return atan2(simd_dot(heading, b.east), simd_dot(heading, b.north))
    }

    /// Whether the heading is north (to a milliradian).
    package var isNorthUp: Bool { simd_length(heading - SphereFrame.basis(at: up).north) < 1e-3 }

    /// The heading `fraction` (0...1) of the way from `from` to north about `up`, by the signed angle between them (the shorter way round, as the box's
    /// `HeadingEase` does by angle): a blend of the two vectors would stay put when they are opposite and then flip. `from` is taken
    /// at the point looked at now (its part along `up` dropped); a heading with no tangent part goes straight to north.
    package static func easedHeading(from: SIMD3<Float>, up: SIMD3<Float>, fraction: Float) -> SIMD3<Float> {
        let north = SphereFrame.basis(at: up).north
        let tangent = from - simd_dot(from, up) * up
        guard simd_length(tangent) > 1e-6 else { return north }
        let h = simd_normalize(tangent)
        let angle = atan2(simd_dot(simd_cross(h, north), up), simd_dot(h, north))  // signed, in (-pi, pi]
        let q = simd_quatf(angle: angle * min(max(fraction, 0), 1), axis: up)
        return simd_normalize(q.act(h))
    }

    /// The pose with `heading` turned to north (`n`, Keep north up). At a pole north is the engine's (east is (1, 0, 0)).
    package func northUp() -> SpherePose {
        var p = self
        p.heading = SphereFrame.basis(at: up).north
        return p
    }

    // MARK: Rotation about the centre (the grab)

    /// The pose with the eye carried by `q` about the planet's centre: `up` and `heading` turn together.
    package func rotated(by q: simd_quatf) -> SpherePose {
        var p = self
        p.up = simd_normalize(q.act(up))
        var h = q.act(heading)
        h -= simd_dot(h, p.up) * p.up  // (rounding)
        p.heading = simd_normalize(h)
        return p
    }
}

/// A pose on a planet: the matrices, rays and the clamps that need the planet's radius and the lid.
package struct SphereCamera: Equatable, Sendable {
    package var pose: SpherePose
    package var planetRadius: Float
    package var lidRadius: Float

    package init(pose: SpherePose, planetRadius: Float, lidRadius: Float) {
        self.pose = pose
        self.planetRadius = planetRadius
        self.lidRadius = max(lidRadius, planetRadius)
    }

    // MARK: Basis

    package var target: SIMD3<Float> { pose.up * planetRadius }
    package var forward: SIMD3<Float> {
        -cos(pose.tilt) * pose.up + sin(pose.tilt) * pose.heading
    }
    /// The screen's up direction in the world.
    package var screenUp: SIMD3<Float> { cos(pose.tilt) * pose.heading + sin(pose.tilt) * pose.up }
    package var right: SIMD3<Float> { simd_normalize(simd_cross(forward, screenUp)) }
    package var eye: SIMD3<Float> {
        target + pose.distance * (cos(pose.tilt) * pose.up - sin(pose.tilt) * pose.heading)
    }
    /// The eye's height above the floor (its distance from the centre less the planet's radius).
    package var altitude: Float { simd_length(eye) - planetRadius }

    // MARK: Matrices

    package func viewMatrix() -> simd_float4x4 {
        let r = right, u = screenUp, f = forward, p = eye
        return simd_float4x4(columns: (
            SIMD4(r.x, u.x, -f.x, 0),
            SIMD4(r.y, u.y, -f.y, 0),
            SIMD4(r.z, u.z, -f.z, 0),
            SIMD4(-simd_dot(r, p), -simd_dot(u, p), simd_dot(f, p), 1)))
    }

    /// Near and far planes: the far plane just behind the farthest thing the planet can hold (the far side of the lid), the near plane in
    /// front of the nearest, so that the depth buffer's range is spent on the planet whether it is seen whole or from a hover.
    package var clipPlanes: (near: Float, far: Float) {
        let r = simd_length(eye)
        let far = r + lidRadius
        let outside = r - lidRadius
        let near = outside > 1 ? max(0.1, 0.35 * outside) : 0.1
        return (near, far)
    }

    package func projectionMatrix(aspect: Float) -> simd_float4x4 {
        let fy = 1 / tan(pose.fovY / 2)
        let (near, far) = clipPlanes
        let range = near - far
        return simd_float4x4(columns: (
            SIMD4(fy / aspect, 0, 0, 0),
            SIMD4(0, fy, 0, 0),
            SIMD4(0, 0, far / range, -1),
            SIMD4(0, 0, near * far / range, 0)))
    }

    package func viewProjectionMatrix(aspect: Float) -> simd_float4x4 {
        projectionMatrix(aspect: aspect) * viewMatrix()
    }

    /// Normalised device coordinates of a planet-frame point: x and y in [-1, 1] inside the view; nil for a point behind the eye.
    package func ndc(of point: SIMD3<Float>, aspect: Float) -> SIMD2<Float>? {
        let clip = viewProjectionMatrix(aspect: aspect) * SIMD4(point, 1)
        guard clip.w > 0 else { return nil }
        return SIMD2(clip.x, clip.y) / clip.w
    }

    /// The ray through the point `ndc` of the view (x right, y up, both in [-1, 1]): the eye and the unit direction.
    package func ray(ndc: SIMD2<Float>, aspect: Float) -> (origin: SIMD3<Float>, direction: SIMD3<Float>) {
        let half = tan(pose.fovY / 2)
        let d = forward + right * (ndc.x * half * aspect) + screenUp * (ndc.y * half)
        return (eye, simd_normalize(d))
    }

    // MARK: Ray and sphere

    /// The parameters of the entry and exit of the ray `origin + t * direction` through the sphere of `radius` about the centre; nil if it
    /// misses it.
    package static func intersect(origin: SIMD3<Float>, direction: SIMD3<Float>, radius: Float) -> (near: Float, far: Float)? {
        let b = simd_dot(origin, direction)
        let c = simd_dot(origin, origin) - radius * radius
        let disc = b * b - c
        guard disc >= 0 else { return nil }
        let root = disc.squareRoot()
        return (-b - root, -b + root)
    }

    /// The floor point the ray hits first (in front of the eye): the direction from the centre, or nil.
    package func floorDirection(ndc: SIMD2<Float>, aspect: Float) -> SIMD3<Float>? {
        let r = ray(ndc: ndc, aspect: aspect)
        guard let hit = Self.intersect(origin: r.origin, direction: r.direction, radius: planetRadius) else { return nil }
        let t = hit.near > 0 ? hit.near : hit.far
        guard t > 0 else { return nil }
        return simd_normalize(r.origin + t * r.direction)
    }

    /// The direction under the view point `ndc` for the grab: the floor hit, or, where the ray passes beside the planet (the pointer is
    /// outside the disc), the direction of the point of the ray nearest the centre, which is continuous with the hit at the silhouette.
    package func grabDirection(ndc: SIMD2<Float>, aspect: Float) -> SIMD3<Float> {
        if let hit = floorDirection(ndc: ndc, aspect: aspect) { return hit }
        let r = ray(ndc: ndc, aspect: aspect)
        let t = max(-simd_dot(r.origin, r.direction), 0)
        let p = r.origin + t * r.direction
        return simd_length(p) > 1e-6 ? simd_normalize(p) : pose.up
    }

    /// Whether the point is hidden from the eye by the planet itself (the floor sphere stands between): the markers and labels of the far
    /// side are not drawn. The lid is translucent and hides nothing.
    package func isOccluded(_ point: SIMD3<Float>) -> Bool {
        let to = point - eye
        let distance = simd_length(to)
        guard distance > 1e-6 else { return false }
        guard let hit = Self.intersect(origin: eye, direction: to / distance, radius: planetRadius * 0.999) else { return false }
        return hit.near > 0 && hit.near < distance - 1e-3
    }

    // MARK: Gestures

    /// Carries the camera so that `anchor` (a unit direction, grabbed at the press) is under the view point `ndc`: the pointer drags the
    /// planet. Exact for a pointer on the planet, continuous beyond the silhouette.
    package mutating func grab(anchor: SIMD3<Float>, at ndc: SIMD2<Float>, aspect: Float) {
        let under = grabDirection(ndc: ndc, aspect: aspect)
        pose = pose.rotated(by: SphereFrame.rotation(from: under, to: anchor))
    }

    /// A pan of `dx`, `dy` points (+y down) of a view `size` points: the content follows the fingers, as a grab from the centre.
    package mutating func pan(dx: Float, dy: Float, size: SIMD2<Float>) {
        guard size.x >= 1, size.y >= 1, dx.isFinite, dy.isFinite else { return }
        let aspect = size.x / size.y
        let centre = SIMD2<Float>(0, 0)
        let anchor = grabDirection(ndc: centre, aspect: aspect)
        grab(anchor: anchor, at: SIMD2(2 * dx / size.x, -2 * dy / size.y), aspect: aspect)
    }

    /// Multiplies the distance by `factor` (below 1 comes closer) about the view point `ndc`: the floor point under it stays under it.
    package mutating func zoom(by factor: Float, at ndc: SIMD2<Float>? = nil, aspect: Float) {
        guard factor.isFinite, factor > 0 else { return }
        let before = ndc.map { grabDirection(ndc: $0, aspect: aspect) }
        pose.distance = min(max(pose.distance * factor, SpherePose.minDistance), maxDistance(aspect: aspect))
        if let ndc, let before { grab(anchor: before, at: ndc, aspect: aspect) }
    }

    /// Turns the view about the floor point looked at: `dx` points turn the heading (a drag to the right turns the planet to the right), `dy`
    /// points tilt it (a drag up tips the view towards the horizon).
    package mutating func turn(dx: Float, dy: Float) {
        let angle = dx * 0.006
        let q = simd_quatf(angle: -angle, axis: pose.up)
        pose.heading = simd_normalize(q.act(pose.heading))
        pose.tilt = min(max(pose.tilt - dy * 0.006, 0), SpherePose.maxTilt)
    }

    // MARK: Limits and the standard views

    /// The farthest the eye is let go: the planet seen whole with some room (`wholeDistance` is the Home view).
    package func maxDistance(aspect: Float) -> Float {
        Self.fitDistance(lidRadius: lidRadius, planetRadius: planetRadius, fovY: pose.fovY, aspect: aspect, fill: 0.45)
    }

    /// The distance from the floor point looked at (straight down) at which the lid sphere fills `fill` of the smaller half field of view.
    package static func fitDistance(lidRadius: Float, planetRadius: Float, fovY: Float, aspect: Float, fill: Float = 0.8) -> Float {
        let half = atan(min(1, aspect) * tan(fovY / 2))
        // the eye at r from the centre sees a sphere of radius R at the half angle asin(R / r): r = R / sin(fill * half)
        let r = lidRadius / sin(max(fill * half, 0.01))
        return max(r - planetRadius, SpherePose.minDistance)
    }

    /// The Home view: the planet whole, centred on `centre` (the populated side), north up.
    package static func home(planetRadius: Float, lidRadius: Float, centre: SIMD3<Float>, aspect: Float,
                             fovY: Float = SpherePose.defaultFovY) -> SpherePose {
        let u = simd_length(centre) > 1e-6 ? simd_normalize(centre) : SIMD3<Float>(0, 0, 1)
        return SpherePose(up: u, heading: SphereFrame.basis(at: u).north, tilt: 0,
                          distance: fitDistance(lidRadius: lidRadius, planetRadius: planetRadius, fovY: fovY, aspect: aspect),
                          fovY: fovY)
    }

    /// Clamps the distance (and the tilt) of the pose to the limits.
    package mutating func clamp(aspect: Float) {
        pose.distance = min(max(pose.distance, SpherePose.minDistance), max(maxDistance(aspect: aspect), SpherePose.minDistance))
        pose.tilt = min(max(pose.tilt, 0), SpherePose.maxTilt)
    }
}

/// How far in the view is (the HUD's word and the rule for what is drawn).
package enum SphereLevel: String, Equatable, Sendable {
    /// The planet seen whole or nearly.
    case planet
    /// A region: the curvature of the horizon shows.
    case region
    /// Close enough to see single organisms.
    case organism

    package static func level(distance: Float, planetRadius: Float, lidRadius: Float) -> SphereLevel {
        if distance > 0.9 * lidRadius { return .planet }
        if distance > 0.6 * (lidRadius - planetRadius) + 8 { return .region }
        return .organism
    }
}

/// The flight of the camera to a place (`flyTo`, Home, a click on a storm): a great circle with the distance rising in the middle of a long
/// way and settling at the goal, eased at both ends.
package struct SphereFlight: Equatable, Sendable {
    package var from: SpherePose
    package var to: SpherePose
    package var duration: Float
    package var peak: Float

    /// A flight from `from` to `to`. `straight` (reduced motion) is a plain 0.25 s ease with no zoom out.
    package init(from: SpherePose, to: SpherePose, planetRadius: Float, straight: Bool = false) {
        self.from = from
        self.to = to
        let angle = SphereFrame.angle(from.up, to.up)
        if straight {
            duration = 0.25
            peak = 0
        } else {
            duration = min(max(0.55 + 0.9 * angle / .pi + 0.25 * abs(log(max(to.distance, 1) / max(from.distance, 1))), 0.5), 2.2)
            // a long way is flown higher: the planet's width in the middle of the flight is a fraction of the angle flown
            peak = min(max(planetRadius * 2.2 * angle / .pi, 0) * 0.6, planetRadius * 2)
        }
    }

    /// The smooth step at both ends.
    package static func ease(_ s: Float) -> Float {
        let t = min(max(s, 0), 1)
        return t * t * (3 - 2 * t)
    }

    /// The pose `elapsed` seconds in; the goal itself from `duration` on.
    package func pose(at elapsed: Float) -> SpherePose {
        let s = Self.ease(duration > 0 ? elapsed / duration : 1)
        if s >= 1 { return to }
        let up = SphereFrame.slerp(from.up, to.up, s)
        // carry the heading along the flight (the rotation that takes from.up to up), then blend towards the goal's
        let q = SphereFrame.rotation(from: from.up, to: up)
        let carried = q.act(from.heading)
        let goalHeading = to.heading
        let heading = simd_normalize(simd_mix(carried, goalHeading, SIMD3(repeating: s)))
        let logd = (1 - s) * log(from.distance) + s * log(to.distance)
        let bump = peak * sin(.pi * s)
        return SpherePose(up: up, heading: heading, tilt: from.tilt + (to.tilt - from.tilt) * s,
                          distance: exp(logd) + bump, fovY: to.fovY)
    }
}
