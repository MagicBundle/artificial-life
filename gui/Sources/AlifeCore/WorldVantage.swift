import CoreGraphics
import Foundation
import simd

/// The view of the world at every zoom, from the flat close-up to the globe (DESIGN §36.10; the spec's `WorldView` of 3.1, named
/// `WorldVantage` because the SwiftUI view of the window already is `WorldView`). Pure.
///
/// The state is `WorldPose` (the centre `(x, y)`, or its direction while it is inside a pole cap, the heading, the tilt, the distance
/// `d`); `Morph` turns the distance into the scalar `s` and the surface of the scene. The scene is drawn in *sphere units* in the
/// centre's frame: the centre's surface point is at `(0, 0, R)`, the eye `E = M d` away from the look-at point, the screen axes are
/// `x` right and `y` up with the heading already applied (`Morph.position` bakes it in), so the camera is `yaw = pi/2`,
/// `pitch = -pi/2 + tilt` at every `s`: **the tilt orbits the camera about the look-at point of the planet, as it does over the flat
/// map** (the tilt is not eased out by `s`; the player's controls work the same at the globe: tilt, turn, the four standard views and
/// the 3D button).
///
/// What this type answers, by regime (the spec's 2.6 and 5.1):
/// - `screen`: where a torus point is drawn (the copy of the seam rule 2.2, nil behind the limb or behind the eye);
/// - `pick`: the front-most surface point under a pointer: the plane at `s = 0`, the sphere's analytic answer at `s = 1`, in the
///   morph the **two-start Newton** (the roll sphere's and the Mercator sphere's analytic answers; the converged root of least depth;
///   a 16 x 16 grid fallback);
/// - `grab` and `zoom(by:at:)`: the **regime solver**. `s = 0`: the exact translation of the plane (the existing `WorldPose.grab`);
///   `s <= 0.5`: that translation refined by a damped Gauss-Newton on `(x, ybar)`, the heading fixed; `s >= 0.9`: the exact rotation
///   that takes the picked direction to the pointer's, the heading carried; in between the two orientations blended by
///   `smoothstep(0.5, 0.9, s)`.
package struct WorldVantage: Equatable, Sendable {
    package var pose: WorldPose { didSet { refresh() } }
    /// `L`.
    package let size: Float
    /// `H`, the world's height.
    package let height: Float
    /// The window's aspect (width over height).
    package let aspect: Float
    /// Whether the globe exists for this world (3D, with a map). Off: the view is the flat map at every distance and `s` is 0.
    package let globe: Bool

    /// The morph scalar.
    package private(set) var s: Float = 0
    /// The centre's frame (`Morph.Frame`), with the heading.
    package private(set) var frame: Morph.Frame
    /// The camera in the centre's frame (sphere units).
    package private(set) var camera: Camera
    private var viewProjection = matrix_identity_float4x4

    /// What is under a pointer.
    package struct Hit: Equatable, Sendable {
        /// The torus point, `x` and `y` in `[0, L)`. In a pole cap (`seam`) the seam row `L/2` and the longitude's column.
        package var x: Float
        package var y: Float
        package var z: Float
        /// Inside a pole cap: the point names no place (the tooltip says "wrap").
        package var seam: Bool
        /// The offsets from the centre in torus units (east, north: `n` is the unwrapped row offset).
        package var e: Float
        package var n: Float
        /// The point in the centre's frame (sphere units).
        package var point: SIMD3<Float>
        /// The world-fixed unit direction of the point on the sphere.
        package var direction: SIMD3<Float>
        /// Along the ray.
        package var depth: Float
    }

    /// What a grab holds: a torus point, its height and its direction on the sphere.
    package struct Anchor: Equatable, Sendable {
        package var x: Float
        package var y: Float
        package var z: Float
        package var direction: SIMD3<Float>
    }

    package init(pose: WorldPose, size: Float, height: Float, aspect: Float, globe: Bool) {
        self.pose = pose
        self.size = size
        self.height = height
        self.aspect = max(aspect, 1e-3)
        self.globe = globe && size > 0 && !pose.faceDown && height > 0
        frame = Morph.Frame(x: 0, ybar: 0, L: max(size, 1))
        camera = Camera(target: .zero, distance: 1, yaw: 0, pitch: 0)
        refresh()
    }

    // MARK: Derived state

    package var isGlobe: Bool { globe }

    /// The least square of the distance a pose is held at: the unit of the Newton tolerances below.
    private static let newtonStep: Float = 0.05
    private static let newtonIterations = 10

    private mutating func refresh() {
        let L = size
        s = globe ? Morph.s(distance: pose.distance, L: L, fovY: pose.fovY, aspect: aspect) : 0
        guard L > 0 else { return }
        if let c = validCap {
            frame = Morph.Frame(centre: c, heading: pose.heading, L: L)
        } else {
            frame = Morph.Frame(x: pose.x, ybar: pose.ybar(size: L), heading: pose.heading, L: L)
        }
        let R = GlobeMap.radius(L: L)
        let E = Morph.eyeDistance(distance: pose.distance)
        let target = R + Morph.zeta(s: s, L: L, height: height) * pose.lookZ
        let reach = max(R + E, 0.9 * Morph.M * L / 2) * 1.05 + Morph.M * max(height, 0)
        camera = Camera(target: SIMD3(0, 0, target), distance: E, yaw: .pi / 2, pitch: -.pi / 2 + pose.tilt, fovY: pose.fovY, near: 0.1,
                        far: 8 * reach, scene: Camera.Scene(center: .zero, radius: reach))
        viewProjection = camera.viewProjectionMatrix(aspect: aspect)
    }

    /// The pose's cap direction while it is consistent with `(x, y)` (an edit of `x` or `y` since the grab that set it drops it).
    package var validCap: SIMD3<Float>? {
        guard let c = pose.cap, size > 0, abs(Torus.wrapDelta(pose.y - size / 2, size: size)) < 0.05 else { return nil }
        let column = Torus.wrap(size * atan2(c.y, c.x) / (2 * .pi), size: size)
        return abs(Torus.wrapDelta(pose.x - column, size: size)) < 0.05 ? c : nil
    }

    /// The unit direction of the centre.
    package var centreDirection: SIMD3<Float> { frame.centre }

    /// The position in the centre's frame of the mesh vertex at the offsets `(e, n)`, height `z`.
    package func position(e: Float, n: Float, z: Float) -> SIMD3<Float> {
        Morph.position(e: e, n: n, z: z, frame: frame, s: s, L: size, height: height)
    }

    // MARK: Torus to offsets

    /// The offsets `(e, n)` from the centre of the torus point `(x, y)`: the nearest column; the nearest row too while the seam copies
    /// exist (`s` under 0.3, SeamCopies), else the principal one (`|ybar| <= L/2`: the Mercator sphere has no copy past a pole).
    package func offsets(x: Float, y: Float) -> (e: Float, n: Float) {
        let e = Torus.wrapDelta(x - frame.x, size: size)
        let n0 = Torus.wrapDelta(y, size: size) - frame.ybar
        guard SeamCopies.alpha(s: s) > 0 else { return (e, n0) }
        var n = n0
        for k: Float in [-1, 1] where abs(n0 + k * size) < abs(n) { n = n0 + k * size }
        return (e, n)
    }

    /// The offsets of an image that is already placed near the centre (no wrapping: `x`, `y` as `Torus.nearestImage` returns them).
    package func offsets(image x: Float, _ y: Float) -> (e: Float, n: Float) {
        (x - frame.x, y - pose.y + Torus.wrapDelta(pose.y, size: size) - frame.ybar)
    }

    /// The torus point of the offsets `(e, n)`: wrapped, with `seam` once the row is past a pole and the copies are gone.
    package func torus(e: Float, n: Float) -> (x: Float, y: Float, seam: Bool) {
        let rowUnwrapped = frame.ybar + n
        return (Torus.wrap(frame.x + e, size: size), Torus.wrap(rowUnwrapped, size: size),
                s >= SeamCopies.fadeEnd && abs(rowUnwrapped) >= size / 2)
    }

    // MARK: Screen

    package func point(ofNDC ndc: SIMD3<Float>, in view: CGSize) -> CGPoint {
        CGPoint(x: CGFloat(ndc.x * 0.5 + 0.5) * view.width, y: CGFloat(0.5 - ndc.y * 0.5) * view.height)
    }

    /// The point in the view and the depth of the mesh point `(e, n, z)`; nil behind the eye, and behind the limb (the surface faces
    /// away from the eye).
    package func screen(e: Float, n: Float, z: Float, in view: CGSize) -> (point: CGPoint, depth: Float)? {
        let p = position(e: e, n: n, z: z)
        let normal = Morph.normal(e: e, n: n, frame: frame, s: s, L: size)
        // (a point exactly edge-on, the look-at point of the Side view, is on the horizon and is kept)
        let toEye = camera.position - p
        guard simd_dot(normal, toEye) > -1e-3 * simd_length(toEye) else { return nil }
        let clip = viewProjection * SIMD4(p, 1)
        guard clip.w > 0 else { return nil }
        let ndc = SIMD3(clip.x, clip.y, clip.z) / clip.w
        return (point(ofNDC: ndc, in: view), clip.w)
    }

    /// Where the torus point is drawn.
    package func screen(_ world: SIMD3<Float>, in view: CGSize) -> CGPoint? {
        let o = offsets(x: world.x, y: world.y)
        return screen(e: o.e, n: o.n, z: world.z, in: view)?.point
    }

    /// Where an image placed near the centre is drawn.
    package func screenOfImage(_ world: SIMD3<Float>, in view: CGSize) -> CGPoint? {
        let o = offsets(image: world.x, world.y)
        return screen(e: o.e, n: o.n, z: world.z, in: view)?.point
    }

    /// The screen vectors, in points, of one torus unit east and one north at `(e, n)` (central differences over half a unit).
    package func jacobian(e: Float, n: Float, z: Float, in view: CGSize) -> (east: SIMD2<Float>, north: SIMD2<Float>)? {
        let h: Float = 0.5
        func at(_ e: Float, _ n: Float) -> SIMD2<Float>? {
            // The hidden-side test is the centre's: the neighbours of a visible point are drawn with it.
            let p = position(e: e, n: n, z: z)
            let clip = viewProjection * SIMD4(p, 1)
            guard clip.w > 0 else { return nil }
            return SIMD2((clip.x / clip.w * 0.5 + 0.5) * Float(view.width), (0.5 - clip.y / clip.w * 0.5) * Float(view.height))
        }
        guard let a = at(e - h, n), let b = at(e + h, n), let c = at(e, n - h), let d = at(e, n + h) else { return nil }
        return ((b - a) / (2 * h), (d - c) / (2 * h))
    }

    /// The semi-axes in points of the ellipse a circle of one torus unit on the surface at `(e, n)` is drawn as (major, minor).
    package func ellipse(e: Float, n: Float, z: Float, in view: CGSize) -> (major: Float, minor: Float)? {
        guard let j = jacobian(e: e, n: n, z: z, in: view) else { return nil }
        let a = j.east.x, c = j.east.y, b = j.north.x, d = j.north.y  // columns (a, c) east, (b, d) north
        let trace = a * a + b * b + c * c + d * d
        let det = a * d - b * c
        let root = max(trace * trace / 4 - det * det, 0).squareRoot()
        return ((trace / 2 + root).squareRoot(), max(trace / 2 - root, 0).squareRoot())
    }

    // MARK: Rays

    /// The ray through `ndc`: the eye and the unit direction, in the centre's frame.
    package func ray(at ndc: SIMD2<Float>) -> (origin: SIMD3<Float>, direction: SIMD3<Float>) {
        let half = tan(camera.fovY / 2)
        let r = simd_normalize(camera.forward + camera.right * (ndc.x * half * aspect) + camera.up * (ndc.y * half))
        return (camera.position, r)
    }

    // MARK: Pick

    /// The result of a pick and the Newton iterations it took (the most over the starts), for the tests.
    package struct Pick: Sendable {
        package var hit: Hit?
        package var iterations: Int
    }

    /// The front-most surface point of height `z` under `ndc`; nil off the planet (outside the disc, behind the limb), at the
    /// horizon of a flat view, and where the floor is not read (`WorldFog.readReach`).
    package func pick(at ndc: SIMD2<Float>, height z: Float = 0) -> Hit? { pickCounted(at: ndc, height: z).hit }

    package func pickCounted(at ndc: SIMD2<Float>, height z: Float = 0) -> Pick {
        guard size > 0 else { return Pick(hit: nil, iterations: 0) }
        let (o, r) = ray(at: ndc)
        let z = min(max(z, 0), max(height, 0))
        if s <= 0 { return Pick(hit: pickPlane(o, r, z), iterations: 0) }
        if s >= 1 { return Pick(hit: pickSphere(o, r, z), iterations: 0) }
        return pickMorph(o, r, z)
    }

    private func finish(e: Float, n: Float, z: Float, o: SIMD3<Float>, r: SIMD3<Float>) -> Hit? {
        guard WorldFog.isReadable(offset: SIMD2(e, n), size: size) else { return nil }
        let p = position(e: e, n: n, z: z)
        let depth = simd_dot(p - o, r)
        guard depth > 0 else { return nil }
        let t = torus(e: e, n: n)
        let q = GlobeMap.direction(longitude: GlobeMap.longitude(x: frame.x + e, L: size),
                                   latitude: GlobeMap.latitude(ybar: frame.ybar + n, L: size))
        return Hit(x: t.x, y: t.y, z: z, seam: t.seam, e: e, n: n, point: p, direction: q, depth: depth)
    }

    /// `s = 0`: the ray against the plane `z`, turned by the heading.
    private func pickPlane(_ o: SIMD3<Float>, _ r: SIMD3<Float>, _ z: Float) -> Hit? {
        guard abs(r.z) > 1e-6 else { return nil }
        let R = GlobeMap.radius(L: size)
        let t = (R + Morph.M * z - o.z) / r.z
        guard t > 0, t.isFinite else { return nil }
        let hit = o + r * t
        let ch = cos(frame.heading), sh = sin(frame.heading)
        let e = (hit.x * ch - hit.y * sh) / Morph.M, n = (hit.x * sh + hit.y * ch) / Morph.M
        return finish(e: e, n: n, z: z, o: o, r: r)
    }

    /// `s = 1`: the ray against the sphere of radius `R + kz z`; the hit's direction names the place (a cap names the seam).
    private func pickSphere(_ o: SIMD3<Float>, _ r: SIMD3<Float>, _ z: Float) -> Hit? {
        let radius = GlobeMap.radius(L: size) + GlobeMap.shellScale(L: size, height: height) * z
        let b = simd_dot(o, r), h = b * b - (simd_dot(o, o) - radius * radius)
        guard h >= 0 else { return nil }
        let t = -b - h.squareRoot()
        guard t > 0 else { return nil }
        let local = o + r * t
        let d = simd_normalize(local)
        let world = d.x * frame.right + d.y * frame.up + d.z * frame.centre
        let place = GlobeMap.torus(direction: world, L: size)
        let ybar = place.seam ? (world.z < 0 ? -size / 2 : size / 2) : Torus.wrapDelta(place.y, size: size)
        let e = Torus.wrapDelta(place.x - frame.x, size: size), n = ybar - frame.ybar
        return Hit(x: place.x, y: place.y, z: z, seam: place.seam, e: e, n: n, point: local, direction: world, depth: t)
    }

    private struct Root {
        var e: Float
        var n: Float
        var depth: Float
    }

    private func pickMorph(_ o: SIMD3<Float>, _ r: SIMD3<Float>, _ z: Float) -> Pick {
        let a1 = simd_normalize(simd_cross(r, camera.up)), a2 = simd_cross(r, a1)
        var roots: [Root] = []
        var iterations = 0
        for start in [rollStart(o, r, z), mercatorStart(o, r, z)] {
            guard let start else { continue }
            let found = newton(from: start, o: o, r: r, z: z, a1: a1, a2: a2)
            iterations = max(iterations, found.iterations)
            if let root = found.root { roots.append(root) }
        }
        if roots.isEmpty, let seed = gridSeed(o, r, z, a1, a2) {
            let found = newton(from: seed, o: o, r: r, z: z, a1: a1, a2: a2)
            iterations = max(iterations, found.iterations)
            if let root = found.root { roots.append(root) }
        }
        guard let best = roots.min(by: { $0.depth < $1.depth }) else { return Pick(hit: nil, iterations: iterations) }
        return Pick(hit: finish(e: best.e, n: best.n, z: z, o: o, r: r), iterations: iterations)
    }

    /// The perpendicular miss of the mesh point `(e, n)` from the ray, in the plane `(a1, a2)` across it.
    private func miss(_ e: Float, _ n: Float, _ z: Float, _ o: SIMD3<Float>, _ a1: SIMD3<Float>, _ a2: SIMD3<Float>) -> SIMD2<Float> {
        let d = position(e: e, n: n, z: z) - o
        return SIMD2(simd_dot(d, a1), simd_dot(d, a2))
    }

    /// Newton on the two offsets (the Jacobian by central differences of the position, 0.05 units; at most 10 iterations; stopping at
    /// a step under 1e-3 units); a halving line search keeps the miss falling. The root is accepted when it is a point of the visible
    /// sheet: the miss under 0.02 sphere units, in front of the eye and facing it.
    private func newton(from start: (e: Float, n: Float), o: SIMD3<Float>, r: SIMD3<Float>, z: Float, a1: SIMD3<Float>,
                        a2: SIMD3<Float>) -> (root: Root?, iterations: Int) {
        var e = start.e, n = start.n
        var g = miss(e, n, z, o, a1, a2)
        var iterations = 0
        let h = Self.newtonStep
        while iterations < Self.newtonIterations {
            let ge1 = miss(e + h, n, z, o, a1, a2), ge0 = miss(e - h, n, z, o, a1, a2)
            let gn1 = miss(e, n + h, z, o, a1, a2), gn0 = miss(e, n - h, z, o, a1, a2)
            let je = (ge1 - ge0) / (2 * h), jn = (gn1 - gn0) / (2 * h)
            let det = je.x * jn.y - jn.x * je.y
            iterations += 1
            guard abs(det) > 1e-9 else { break }
            // solve [je jn] step = -g
            var step = SIMD2((-g.x * jn.y + g.y * jn.x) / det, (-je.x * g.y + je.y * g.x) / det)
            let limit = size / 4
            let length = simd_length(step)
            if length > limit { step *= limit / length }
            var factor: Float = 1
            var next = miss(e + step.x, n + step.y, z, o, a1, a2)
            var tries = 0
            while simd_length(next) > simd_length(g), tries < 4 {
                factor /= 2
                next = miss(e + step.x * factor, n + step.y * factor, z, o, a1, a2)
                tries += 1
            }
            e += step.x * factor
            n += step.y * factor
            g = next
            if simd_length(step * factor) < 1e-3 { break }
        }
        guard simd_length(g) < 0.02 else { return (nil, iterations) }
        let p = position(e: e, n: n, z: z)
        let depth = simd_dot(p - o, r)
        let normal = Morph.normal(e: e, n: n, frame: frame, s: s, L: size)
        guard depth > 0, simd_dot(normal, o - p) > 0 else { return (nil, iterations) }
        return (Root(e: e, n: n, depth: depth), iterations)
    }

    /// The roll sphere's analytic answer: the ray against the sphere of radius `R/s` (plus the lifted height) tangent at the centre,
    /// inverted through the exponential map; the closest approach when the ray misses. The plane's answer when `s` is nearly 0.
    private func rollStart(_ o: SIMD3<Float>, _ r: SIMD3<Float>, _ z: Float) -> (e: Float, n: Float)? {
        let R = GlobeMap.radius(L: size)
        let zeta = Morph.zeta(s: s, L: size, height: height)
        let a = Morph.M * Morph.lambda(s: s, latitude: frame.latitude)
        let ch = cos(frame.heading), sh = sin(frame.heading)
        var ux: Float, uy: Float
        if s < 1e-4 {
            guard abs(r.z) > 1e-6 else { return nil }
            let t = (R + zeta * z - o.z) / r.z
            guard t > 0 else { return nil }
            let hit = o + r * t
            ux = hit.x
            uy = hit.y
        } else {
            let rho = R / s
            let c0 = SIMD3<Float>(0, 0, R - rho)
            let radius = rho + zeta * z
            let oc = o - c0
            let b = simd_dot(oc, r), h = b * b - (simd_dot(oc, oc) - radius * radius)
            let t = h >= 0 ? -b - h.squareRoot() : -b
            let q = oc + r * t
            let length = simd_length(q)
            guard length > 1e-6 else { return nil }
            let theta = acos(min(max(q.z / length, -1), 1))
            let azimuth = atan2(q.y, q.x)
            ux = rho * theta * cos(azimuth)
            uy = rho * theta * sin(azimuth)
        }
        return ((ux * ch - uy * sh) / a, (ux * sh + uy * ch) / a)
    }

    /// The Mercator sphere's analytic answer: the ray against the sphere of radius `R + kz z`, inverted through `GlobeMap` (the
    /// closest approach when the ray misses).
    private func mercatorStart(_ o: SIMD3<Float>, _ r: SIMD3<Float>, _ z: Float) -> (e: Float, n: Float)? {
        let radius = GlobeMap.radius(L: size) + GlobeMap.shellScale(L: size, height: height) * z
        let b = simd_dot(o, r), h = b * b - (simd_dot(o, o) - radius * radius)
        let t = h >= 0 ? -b - h.squareRoot() : -b
        let local = o + r * t
        let length = simd_length(local)
        guard length > 1e-6 else { return nil }
        let d = local / length
        let world = d.x * frame.right + d.y * frame.up + d.z * frame.centre
        let place = GlobeMap.torus(direction: world, L: size)
        let ybar = place.seam ? (world.z < 0 ? -size / 2 : size / 2) : Torus.wrapDelta(place.y, size: size)
        return (Torus.wrapDelta(place.x - frame.x, size: size), ybar - frame.ybar)
    }

    /// The fallback: the best of a 16 x 16 grid of seeds over the world around the centre, by the miss, among the points that face
    /// the eye and are in front of it.
    private func gridSeed(_ o: SIMD3<Float>, _ r: SIMD3<Float>, _ z: Float, _ a1: SIMD3<Float>, _ a2: SIMD3<Float>) -> (e: Float, n: Float)? {
        var best: (e: Float, n: Float, miss: Float)?
        for i in 0..<16 {
            for j in 0..<16 {
                let e = (Float(i) + 0.5) / 16 * size - size / 2, n = (Float(j) + 0.5) / 16 * size - size / 2
                let p = position(e: e, n: n, z: z)
                guard simd_dot(p - o, r) > 0, simd_dot(Morph.normal(e: e, n: n, frame: frame, s: s, L: size), o - p) > 0 else { continue }
                let m = simd_length(miss(e, n, z, o, a1, a2))
                if best == nil || m < best!.miss { best = (e, n, m) }
            }
        }
        return best.map { ($0.e, $0.n) }
    }

    // MARK: Anchors, grab, zoom, pan

    /// The pointer as the normalised device coordinates of the view.
    package static func ndc(of point: CGPoint, in view: CGSize) -> SIMD2<Float> { WorldPose.ndc(of: point, in: view) }

    /// The grip height: the plane through the centre (the followed organism's), the floor from `gripTilt`.
    private var gripZ: Float { min(max(pose.gripZ, 0), max(height, 0)) }

    /// The surface point under `ndc` that a drag would hold.
    package func anchor(at ndc: SIMD2<Float>) -> Anchor? {
        if s <= 0 {
            // the flat map's own rule (the grip plane, the haze), so the close-up is the old close-up
            guard let p = pose.point(at: ndc, aspect: aspect, size: size) else { return nil }
            return Anchor(x: p.x, y: p.y, z: gripZ, direction: GlobeMap.direction(x: p.x, y: p.y, L: size))
        }
        guard let hit = pick(at: ndc, height: gripZ) else { return nil }
        return Anchor(x: hit.x, y: hit.y, z: hit.z, direction: hit.direction)
    }

    /// Moves the view so that `anchor` is under the pointer at `ndc` (the regime solver). False, and nothing changed, when the pointer
    /// holds no surface now.
    ///
    /// `s = 0`: the plane's exact translation. Otherwise one *step* of the solver (below), repeated up to `refinements` times while it
    /// brings the anchor nearer to the pointer: the blend of the two exact solutions is exact only at the ends (the translation does
    /// not hold at a high latitude, the rotation is of a sphere), and a second and third step correct the rest.
    @discardableResult
    package mutating func grab(anchor: Anchor, at ndc: SIMD2<Float>) -> Bool {
        if s <= 0 {
            var moved = pose
            guard moved.grab(anchor: SIMD2(anchor.x, anchor.y), at: ndc, aspect: aspect, size: size) else { return false }
            pose = moved
            return true
        }
        guard var here = pick(at: ndc, height: anchor.z) else { return false }
        var bestMiss = miss(of: here, anchor: anchor)
        for _ in 0..<Self.refinements {
            if bestMiss < 1e-6 { break }
            let from = (centre: frame.centre, heading: pose.heading)
            // the blend first; where it stalls (a translation cannot cross a pole), the exact rotation of the sphere
            var targets = [solvedOrientation(anchor: anchor, at: ndc, here: here)]
            if Morph.smoothstep(0.5, 0.9, s) < 1 { targets.append(rotationSolution(anchor: anchor, picked: here.direction)) }
            var improved = false
            search: for target in targets {
                // a step that does not bring the anchor nearer is shortened (towards the orientation it started from) before it is given up
                for factor in [Float(1), 0.5, 0.25] {
                    let mixed = from.centre + (target.0 - from.centre) * factor
                    guard simd_length(mixed) > 1e-6 else { continue }
                    setOrientation(centre: simd_normalize(mixed),
                                   heading: WorldPose.wrap(from.heading + Torus.wrapDelta(target.1 - from.heading, size: 2 * .pi) * factor))
                    if let next = pick(at: ndc, height: anchor.z) {
                        let m = miss(of: next, anchor: anchor)
                        if m < bestMiss {
                            (here, bestMiss, improved) = (next, m, true)
                            break search
                        }
                    }
                }
            }
            if !improved {
                setOrientation(centre: from.centre, heading: from.heading)
                break
            }
        }
        return true
    }

    /// From this `s` the centre may be inside a pole cap (the rotation regime).
    package static let capFrom: Float = 0.9

    /// The most steps of the regime solver in a grab.
    package static let refinements = 12

    /// How far a pick is from the anchor: the torus offset while the copies exist, the distance of the directions after.
    private func miss(of hit: Hit, anchor: Anchor) -> Float {
        if s <= SeamCopies.fadeEnd {
            return hypot(Torus.wrapDelta(hit.x - anchor.x, size: size), Torus.wrapDelta(hit.y - anchor.y, size: size))
        }
        return simd_length(hit.direction - anchor.direction)
    }

    /// One step: the translation's orientation and the rotation's, blended by `smoothstep(0.5, 0.9, s)`.
    private func solvedOrientation(anchor: Anchor, at ndc: SIMD2<Float>, here: Hit) -> (SIMD3<Float>, Float) {
        let blend = Morph.smoothstep(0.5, 0.9, s)
        var centreT = frame.centre, headingT = pose.heading
        var centreR = frame.centre, headingR = pose.heading
        if blend < 1 { (centreT, headingT) = translationSolution(anchor: anchor, at: ndc, first: here) }
        if blend > 0 { (centreR, headingR) = rotationSolution(anchor: anchor, picked: here.direction) }
        if blend <= 0 { return (centreT, headingT) }
        if blend >= 1 { return (centreR, headingR) }
        let mixed = centreT * (1 - blend) + centreR * blend
        return (simd_length(mixed) > 1e-6 ? simd_normalize(mixed) : centreR,
                WorldPose.wrap(headingT + Torus.wrapDelta(headingR - headingT, size: 2 * .pi) * blend))
    }

    /// The pose with the centre at direction `centre` and `heading`: `(x, y)` its readout, the direction kept in `cap` inside a pole cap.
    package mutating func setOrientation(centre c: SIMD3<Float>, heading: Float) {
        var centre = simd_normalize(c)
        if s < Self.capFrom {
            // before the globe proper the mesh has no cap under the centre: the centre stays on the Mercator rows (the pole rows at most)
            let limit = GlobeMap.phiMax - 1e-3
            let latitude = asin(min(max(centre.z, -1), 1))
            if abs(latitude) > limit {
                centre = GlobeMap.direction(longitude: atan2(centre.y, centre.x), latitude: limit * (latitude < 0 ? -1 : 1))
            }
        }
        let f = Morph.Frame(centre: centre, heading: heading, L: size)
        var next = pose
        next.x = f.x
        if abs(f.latitude) < GlobeMap.phiMax - 1e-4 {
            next.y = Torus.wrap(f.ybar, size: size)
            next.cap = nil
        } else {
            next.y = size / 2
            next.cap = f.centre
        }
        next.heading = WorldPose.wrap(heading)
        pose = next
    }

    /// The exact rotation taking the picked direction `b` to the anchor's `a` about `b x a`, applied to the camera's basis: the whole
    /// rig turns rigidly about the sphere's centre, so the point that was at `b` is at `a`, under the same pointer; the heading is
    /// whatever the turned basis reads (parallel transport).
    private func rotationSolution(anchor: Anchor, picked b: SIMD3<Float>) -> (SIMD3<Float>, Float) {
        let a = anchor.direction
        let axisRaw = simd_cross(b, a)
        let sine = simd_length(axisRaw), cosine = simd_dot(b, a)
        var q = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        if sine > 1e-7 {
            q = simd_quatf(angle: atan2(sine, cosine), axis: axisRaw / sine)
        } else if cosine < 0 {
            // antipodal: any axis perpendicular to b
            let helper: SIMD3<Float> = abs(b.z) < 0.9 ? SIMD3(0, 0, 1) : SIMD3(1, 0, 0)
            q = simd_quatf(angle: .pi, axis: simd_normalize(simd_cross(b, helper)))
        }
        let centre = simd_normalize(q.act(frame.centre))
        let right = q.act(frame.right)
        return (centre, Morph.Frame.heading(right: right, centre: centre))
    }

    /// The translation of the plane's coordinates `(x, ybar)` that puts the anchor under the pointer, from the first-order step
    /// (the offset between what is under the pointer and the anchor) refined by a damped Gauss-Newton: at most 6 iterations, the
    /// Jacobian of the picked place with respect to the centre by finite differences, the heading fixed.
    private func translationSolution(anchor: Anchor, at ndc: SIMD2<Float>, first: Hit) -> (SIMD3<Float>, Float) {
        let L = size
        var cx = frame.x, cy = frame.ybar
        // The centre goes where `(cx, cy)` says. While the seam copies exist (`s` under 0.3) the row wraps, as on the flat map; after,
        // the centre stays on the Mercator rows (a centre in a cap belongs to the globe's regime: `setOrientation`).
        let wraps = s < SeamCopies.fadeEnd
        func direction(_ cx: Float, _ cy: Float) -> SIMD3<Float> {
            let row = wraps ? Torus.wrapDelta(cy, size: L) : min(max(cy, -L / 2), L / 2)
            return GlobeMap.direction(longitude: GlobeMap.longitude(x: cx, L: L), latitude: GlobeMap.latitude(ybar: row, L: L))
        }
        func under(_ cx: Float, _ cy: Float) -> SIMD2<Float>? {
            var v = self
            v.setOrientation(centre: direction(cx, cy), heading: pose.heading)
            guard let hit = v.pick(at: ndc, height: anchor.z) else { return nil }
            return SIMD2(Torus.wrapDelta(hit.x - anchor.x, size: L), Torus.wrapDelta(hit.y - anchor.y, size: L))
        }
        var residual = SIMD2(Torus.wrapDelta(first.x - anchor.x, size: L), Torus.wrapDelta(first.y - anchor.y, size: L))
        let hStep: Float = 0.25
        for iteration in 0..<6 {
            if simd_length(residual) < 2e-3 { break }
            var step = -residual  // the unit Jacobian: the translation of the plane
            if iteration > 0, let ex = under(cx + hStep, cy), let ey = under(cx, cy + hStep) {
                let jx = (ex - residual) / hStep, jy = (ey - residual) / hStep
                let det = jx.x * jy.y - jy.x * jx.y
                if abs(det) > 1e-6 {
                    step = SIMD2((-residual.x * jy.y + residual.y * jy.x) / det, (-jx.x * residual.y + jx.y * residual.x) / det)
                }
            }
            var factor: Float = 1
            var next: SIMD2<Float>?
            for _ in 0..<4 {
                next = under(cx + step.x * factor, cy + step.y * factor)
                if let next, simd_length(next) < simd_length(residual) { break }
                factor /= 2
                next = nil
            }
            guard let accepted = next else { break }
            cx += step.x * factor
            cy += step.y * factor
            residual = accepted
        }
        return (direction(cx, cy), pose.heading)
    }

    /// Multiplies the distance by `factor` (clamped to the limit) and moves the centre so that the surface point that was under the
    /// pointer is under it again (a grab with the new `s`). Without a pointer on the surface the zoom is about the centre.
    package mutating func zoom(by factor: Float, at ndc: SIMD2<Float>?) {
        guard factor > 0, factor.isFinite else { return }
        let anchor = ndc.flatMap { self.anchor(at: $0) }
        let limit = pose.distanceLimit(size: size, aspect: aspect, globe: globe)
        var next = pose
        next.distance = min(max(pose.distance * factor, WorldLevels.minDistance), max(limit, WorldLevels.minDistance))
        pose = next
        if let anchor, let ndc { grab(anchor: anchor, at: ndc) }
    }

    /// A pan by `dx`, `dy` points of a view `view` across (+y down): the map follows the fingers (a grab of the point under the middle
    /// of the picture carried by the displacement). A view that looks nearly along the surface (from `gripTilt`) moves along it by the
    /// displacement at the distance of the point looked at, in the plane's coordinates, scaled by `1/lambda` so that the speed on the
    /// screen is the same at every latitude.
    package mutating func pan(dx: Float, dy: Float, view: CGSize) {
        guard view.width >= 1, view.height >= 1 else { return }
        if s <= 0 {
            var moved = pose
            moved.pan(dx: dx, dy: dy, view: view, aspect: aspect, size: size)
            pose = moved
            return
        }
        if pose.tilt < WorldPose.gripTilt, let middle = anchor(at: .zero) {
            grab(anchor: middle, at: SIMD2(2 * dx / Float(view.width), -2 * dy / Float(view.height)))
            return
        }
        let lam = max(Morph.lambda(s: s, latitude: frame.latitude), 0.05)
        let perPoint = 2 * pose.distance * tan(pose.fovY / 2) / Float(view.height) / lam
        let right = SIMD2(cos(pose.heading), sin(pose.heading)), ahead = SIMD2(-sin(pose.heading), cos(pose.heading))
        let moved = (-dx * right + dy * ahead) * perPoint
        let row = Torus.wrapDelta(frame.ybar + moved.y, size: size)
        let centre = GlobeMap.direction(longitude: GlobeMap.longitude(x: frame.x + moved.x, L: size),
                                        latitude: GlobeMap.latitude(ybar: row, L: size))
        setOrientation(centre: centre, heading: pose.heading)
    }

    /// The clamps of every edit, with the globe's distance limit.
    package mutating func clamp() {
        var next = pose
        next.clamp(aspect: aspect, size: size, height: height, globe: globe)
        pose = next
    }
}
