import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeCore

/// Seeded random numbers for the tests of the globe's view.
struct VantageRng {
    var state: UInt64 = 7
    mutating func next() -> Float {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return Float((z ^ (z >> 31)) >> 40) / Float(1 << 24)
    }
    mutating func range(_ lo: Float, _ hi: Float) -> Float { lo + (hi - lo) * next() }
}

/// The vantage at a given `s`, centre row and angles, on the 1100 x 800 window of the strip tests.
enum VantageFixture {
    static let L: Float = 320
    static let H: Float = 80
    static let view = CGSize(width: 1100, height: 800)
    static let fov = Camera.defaultFovY
    static var aspect: Float { Float(view.width / view.height) }

    static func distance(s: Float) -> Float {
        if s <= 0 { return Morph.flatLimit * L * 0.5 }
        if s >= 1 { return Morph.globeDistance(L: L, fovY: fov, aspect: aspect) }
        return Morph.distance(forS: s, L: L, fovY: fov, aspect: aspect)
    }

    static func make(s: Float, ybar: Float = 0, x: Float = 118, heading: Float = 0, tilt: Float = 0, z: Float = 0,
                     distance d: Float? = nil, globe: Bool = true) -> WorldVantage {
        let pose = WorldPose(x: x, y: Torus.wrap(ybar, size: L), z: z, heading: heading, tilt: tilt, distance: d ?? distance(s: s), fovY: fov)
        return WorldVantage(pose: pose, size: L, height: H, aspect: aspect, globe: globe)
    }

    static func ndc(_ p: CGPoint) -> SIMD2<Float> { WorldPose.ndc(of: p, in: view) }
    static func point(_ n: SIMD2<Float>) -> CGPoint { CGPoint(x: CGFloat(n.x * 0.5 + 0.5) * view.width, y: CGFloat(0.5 - n.y * 0.5) * view.height) }
}

/// The view of the world through the morph (DESIGN §36.10; spec 2.6 and 7.1 `WorldViewPickTests`): the pick on the plane, the sphere and
/// in the morph against a ray-cast ground truth, and the projection that is its inverse.
@Suite struct WorldVantageTests {
    private typealias F = VantageFixture
    private let L = F.L

    // MARK: The close-up is the old close-up

    @Test func atSZeroTheVantageProjectsAndPicksAsTheFlatCameraDoes() {
        var rng = VantageRng()
        for tilt in [Float(0), 0.7, 1.2] {
            let pose = WorldPose(x: 130, y: 40, z: 0, heading: 0.4, tilt: tilt, distance: 20)
            let projector = WorldProjector(pose: pose, size: L, height: F.H, view: F.view, globe: true)
            let flat = WorldProjector(camera: pose.camera(size: L, height: F.H), size: L, view: F.view)
            let v = WorldVantage(pose: pose, size: L, height: F.H, aspect: F.aspect, globe: true)
            #expect(v.s == 0 && !projector.isMorphing)
            for _ in 0..<60 {
                let w = SIMD3(130 + rng.range(-12, 12), 40 + rng.range(-9, 9), rng.range(0, 40))
                guard let a = flat.screen(w), let b = v.screen(w, in: F.view), abs(a.x - 550) < 1100, abs(a.y - 400) < 800 else { continue }
                #expect(hypot(a.x - b.x, a.y - b.y) < 0.05, "tilt \(tilt): \(a) \(b)")
            }
            for _ in 0..<60 {
                let p = CGPoint(x: Double(rng.range(100, 1000)), y: Double(rng.range(250, 760)))
                guard let a = flat.world(at: p, height: 0) else { continue }
                let b = v.pick(at: F.ndc(p), height: 0)
                #expect(b != nil)
                if let b { #expect(abs(b.x - a.x) < 5e-3 && abs(b.y - a.y) < 5e-3, "tilt \(tilt)") }
            }
        }
    }

    @Test func theProjectorOfAPoseWithoutTheGlobeIsTheFlatCameraAtEveryDistance() {
        let pose = WorldPose(x: 130, y: 40, distance: 120)
        let off = WorldProjector(pose: pose, size: L, height: F.H, view: F.view, globe: false)
        let on = WorldProjector(pose: pose, size: L, height: F.H, view: F.view, globe: true)
        #expect(!off.isMorphing && off.vantage == nil)
        #expect(on.isMorphing && on.s > 0.3)
        #expect(on.level == .region || on.level == .planet)
        #expect(WorldProjector(pose: pose, size: L, height: F.H, view: F.view, mapShown: false, globe: true).vantage == nil)
    }

    // MARK: The sphere

    @Test func atSOneAPointOfTheNearSideRoundTripsThroughTheScreen() {
        var rng = VantageRng()
        for ybar in [Float(0), 38, 80] {
            let v = F.make(s: 1, ybar: ybar)
            var worst: Float = 0
            var checked = 0
            for _ in 0..<200 {
                let x = Torus.wrap(118 + rng.range(-70, 70), size: L)
                let y = Torus.wrap(ybar + rng.range(-60, 60), size: L)
                let zz = rng.range(0, F.H)
                guard let screen = v.screen(SIMD3(x, y, zz), in: F.view), let hit = v.pick(at: F.ndc(screen), height: zz) else { continue }
                checked += 1
                worst = max(worst, abs(Torus.wrapDelta(hit.x - x, size: L)), abs(Torus.wrapDelta(hit.y - y, size: L)))
            }
            #expect(checked > 100 && worst < 0.02, "ybar \(ybar): checked \(checked), worst \(worst)")
        }
    }

    @Test func theFarSideOfTheGlobeIsBehindTheLimbAndTheSkyHasNoPlace() {
        let v = F.make(s: 1, ybar: 0, x: 0)
        #expect(v.screen(SIMD3(L / 2, 0, 0), in: F.view) == nil, "the antipode")
        #expect(v.screen(SIMD3(0, 0, 0), in: F.view) != nil)
        #expect(v.pick(at: SIMD2(0.98, 0.98)) == nil, "outside the disc")
        #expect(v.pick(at: .zero) != nil)
        // The picked point is on the near side: its normal faces the eye.
        let hit = v.pick(at: SIMD2(0.3, 0.2))!
        #expect(simd_dot(simd_normalize(hit.point), v.camera.position - hit.point) > 0)
    }

    @Test func aPointerOnAPoleCapNamesTheSeam() {
        var v = F.make(s: 1, ybar: 0)
        v.setOrientation(centre: SIMD3(0, 0, 1), heading: 0)
        #expect(v.pose.cap != nil && v.pose.y == L / 2)
        let hit = v.pick(at: .zero)
        #expect(hit?.seam == true)
        #expect(abs(hit!.direction.z - 1) < 1e-4)
        // 20 degrees from the pole (well out of the 4.95 degree cap): a place.
        let away = v.pick(at: SIMD2(0.5, 0.0))
        #expect(away?.seam == false)
        // and the centre keeps its direction through the readout
        #expect(simd_length(v.centreDirection - SIMD3(0, 0, 1)) < 1e-6)
    }

    @Test func aCapDirectionLivesOnlyWhileXAndYAgreeWithIt() {
        var v = F.make(s: 1, ybar: 0)
        v.setOrientation(centre: simd_normalize(SIMD3(0.05, 0.02, 1)), heading: 0.3)
        #expect(v.validCap != nil)
        var edited = v.pose
        edited.y = 100  // a follow moved the centre
        let moved = WorldVantage(pose: edited, size: L, height: F.H, aspect: F.aspect, globe: true)
        #expect(moved.validCap == nil)
        var clamped = v.pose
        clamped.clamp(aspect: F.aspect, size: L, height: F.H, globe: false)
        #expect(clamped.cap == nil, "no cap on the flat map")
    }

    // MARK: The Newton pick against the ray-cast mesh

    /// The front-most front-facing triangle of the mesh at height `z` (6-unit cells, with the seam copies while they exist) a ray meets.
    private struct Mesh {
        var positions: [SIMD3<Float>] = []
        var triangles: [(Int, Int, Int)] = []

        init(_ v: WorldVantage, z: Float) {
            let count = 55, step = F.L / Float(count - 1)
            var copies = [0]
            if SeamCopies.alpha(s: v.s) > 0 { copies = [-1, 0, 1] }
            for k in copies {
                let base = positions.count
                for j in 0..<count {
                    for i in 0..<count {
                        let e = -F.L / 2 + Float(i) * step
                        let n = -F.L / 2 + Float(j) * step + Float(k) * F.L - v.frame.ybar
                        positions.append(v.position(e: e, n: n, z: z))
                    }
                }
                for j in 0..<(count - 1) {
                    for i in 0..<(count - 1) {
                        let a = base + j * count + i
                        triangles.append((a, a + 1, a + count))
                        triangles.append((a + 1, a + count + 1, a + count))
                    }
                }
            }
        }

        /// The nearest front-facing hit along the ray, as the point, with the cosine of the angle between the ray and the normal (the
        /// ray-cast is a chord of the curved surface: its error grows as that cosine falls, near the limb).
        func cast(_ o: SIMD3<Float>, _ r: SIMD3<Float>) -> (point: SIMD3<Float>, t: Float, grazing: Float)? {
            var best: (SIMD3<Float>, Float, Float)?
            for (a, b, c) in triangles {
                let p0 = positions[a], e1 = positions[b] - p0, e2 = positions[c] - p0
                let h = simd_cross(r, e2)
                let det = simd_dot(e1, h)
                guard abs(det) > 1e-9 else { continue }
                let inv = 1 / det
                let sv = o - p0
                let u = simd_dot(sv, h) * inv
                guard u >= 0, u <= 1 else { continue }
                let q = simd_cross(sv, e1)
                let w = simd_dot(r, q) * inv
                guard w >= 0, u + w <= 1 else { continue }
                let t = simd_dot(e2, q) * inv
                guard t > 0 else { continue }
                // front-facing: the outward normal (east x north) faces the eye
                let facing = simd_dot(simd_normalize(simd_cross(e1, e2)), -r)
                guard facing > 0 else { continue }
                if best == nil || t < best!.1 { best = (o + r * t, t, facing) }
            }
            return best
        }
    }

    @Test func theMorphPickFindsTheFrontMostSheetOfTheMesh() {
        var rng = VantageRng(state: 99)
        var total = 0, good = 0, none = 0, grazingSkipped = 0, online = 0
        var iterations = 0
        var worst: Float = 0
        for s in [Float(0.05), 0.2, 0.4, 0.5, 0.6, 0.8, 0.95] {
            for ybar in [Float(0), 38, 80] {
                for z in [Float(0), 60] {
                    let v = F.make(s: s, ybar: ybar, z: 0)
                    let mesh = Mesh(v, z: z)
                    for _ in 0..<40 {
                        let ndc = SIMD2(rng.range(-0.9, 0.9), rng.range(-0.9, 0.9))
                        let (o, r) = v.ray(at: ndc)
                        guard let truth = mesh.cast(o, r) else { none += 1; continue }
                        let counted = v.pickCounted(at: ndc, height: z)
                        iterations = max(iterations, counted.iterations)
                        guard let hit = counted.hit else { total += 1; continue }
                        // the pick is on the ray (exact), whatever the mesh says
                        let along = simd_dot(hit.point - o, r)
                        if simd_length(hit.point - (o + r * along)) < 0.05 { online += 1 }
                        // a chord of the mesh is a good ground truth only away from the limb
                        guard truth.grazing > 0.2 else { grazingSkipped += 1; continue }
                        total += 1
                        let error = simd_length(hit.point - truth.point)
                        worst = max(worst, error)
                        // the mesh is a chord of the convex surface: it lies behind it, by up to 0.6 / cos(grazing angle) at 6-unit cells
                        if error < 0.6 / truth.grazing, hit.depth <= truth.t + 0.05 { good += 1 }
                    }
                }
            }
        }
        print("PICK morph: rays \(total) (no mesh hit \(none), grazing \(grazingSkipped)), within 1 unit \(good), on the ray \(online), worst \(worst), most iterations \(iterations)")
        #expect(total > 500)
        #expect(Float(good) / Float(total) >= 0.995, "on the front-most sheet within the chord: \(good) of \(total), worst \(worst)")
        #expect(online >= good)
        #expect(iterations <= 10)
    }

    @Test func theMorphPickAlsoHoldsOnATiltedView() {
        var rng = VantageRng(state: 5)
        var total = 0, good = 0
        for tilt in [Float(0.5), 1.1] {
            for s in [Float(0.3), 0.7, 1.0] {
                let v = F.make(s: s, ybar: 38, tilt: tilt)
                let mesh = Mesh(v, z: 0)
                for _ in 0..<50 {
                    let ndc = SIMD2(rng.range(-0.9, 0.9), rng.range(-0.9, 0.9))
                    let (o, r) = v.ray(at: ndc)
                    guard let truth = mesh.cast(o, r), truth.grazing > 0.2 else { continue }
                    guard WorldFog.isReadable(offset: SIMD2(truth.point.x, truth.point.y) / Morph.M, size: L) else { continue }
                    total += 1
                    if let hit = v.pick(at: ndc), simd_length(hit.point - truth.point) < 0.6 / truth.grazing, hit.depth <= truth.t + 0.05 { good += 1 }
                }
            }
        }
        #expect(total > 100 && Float(good) / Float(total) >= 0.99, "\(good) of \(total)")
    }

    @Test func theFrontMostRootWinsWhereTheSheetsOverlap() {
        // Between s = 0.3 and 0.6 the vase's sheets overlap (the first draft's single start found the back one 25% of the time): the pick
        // must be on the sheet that faces the eye, nearer than the back sheet's point.
        var rng = VantageRng(state: 11)
        for s in [Float(0.35), 0.45, 0.55] {
            let v = F.make(s: s, ybar: 38)
            var facing = 0, n = 0
            for _ in 0..<100 {
                let ndc = SIMD2(rng.range(-0.8, 0.8), rng.range(-0.8, 0.8))
                guard let hit = v.pick(at: ndc) else { continue }
                n += 1
                let normal = Morph.normal(e: hit.e, n: hit.n, frame: v.frame, s: s, L: L)
                if simd_dot(normal, v.camera.position - hit.point) > 0 { facing += 1 }
            }
            #expect(n > 80 && facing == n, "s \(s): \(facing) of \(n) face the eye")
        }
    }

    @Test func theGridFallbackExistsForAViewWithNoAnalyticRoot() {
        // Not reachable by the starts alone? It is by the pick's own rule: here it only has to be deterministic and NaN-free.
        let v = F.make(s: 0.5, ybar: 80, tilt: 1.0)
        for ndc in [SIMD2<Float>(0, -0.99), SIMD2(0.97, 0.97), SIMD2(-0.97, 0.3)] {
            let a = v.pick(at: ndc), b = v.pick(at: ndc)
            #expect(a == b)
            if let a { #expect(a.x.isFinite && a.y.isFinite && a.depth.isFinite) }
        }
    }

    // MARK: Morph pieces this stage added

    @Test func theAnalyticNormalIsUpOnTheMapAndRadialOnTheSphere() {
        let flat = Morph.Frame(x: 100, ybar: 30, heading: 0.3, L: L)
        #expect(Morph.normal(e: 5, n: -3, frame: flat, s: 0, L: L) == SIMD3(0, 0, 1))
        let v = F.make(s: 1, ybar: 30)
        var rng = VantageRng(state: 3)
        for _ in 0..<50 {
            let e = rng.range(-40, 40), n = rng.range(-40, 40)
            let normal = Morph.normal(e: e, n: n, frame: v.frame, s: 1, L: L)
            let radial = simd_normalize(v.position(e: e, n: n, z: 0))
            #expect(simd_length(normal - radial) < 1e-4)
        }
        // mid-morph: unit length, facing the eye at the centre
        for s in [Float(0.1), 0.5, 0.9] {
            let normal = Morph.normal(e: 10, n: 4, frame: flat, s: s, L: L)
            #expect(abs(simd_length(normal) - 1) < 1e-5 && normal.z > 0)
        }
    }

    @Test func aFrameFromADirectionIsTheFrameOfItsPlaceAndKeepsAPoleCap() {
        let a = Morph.Frame(x: 250, ybar: -45, heading: 0.6, L: L)
        let b = Morph.Frame(centre: a.centre, heading: 0.6, L: L)
        #expect(abs(Torus.wrapDelta(b.x - a.x, size: L)) < 1e-2 && abs(b.ybar - a.ybar) < 0.02 && abs(b.latitude - a.latitude) < 1e-5)
        #expect(simd_length(b.right - a.right) < 1e-5 && simd_length(b.up - a.up) < 1e-5)
        #expect(abs(Morph.Frame.heading(right: a.right, centre: a.centre) - 0.6) < 1e-5)
        let pole = Morph.Frame(centre: SIMD3(0.02, 0, 1), heading: 0, L: L)
        #expect(pole.ybar == L / 2 && pole.latitude > GlobeMap.phiMax)
    }

    @Test func theRollKeepsItsPrecisionAtASmallS() {
        // 1 - cos(theta) with a radius of 1e5 would lose several tenths of a unit; the versine form keeps the plane's height.
        let f = Morph.Frame(x: 0, ybar: 0, L: L)
        let p = Morph.rollPosition(e: 100, n: 0, z: 0, frame: f, s: 1e-4, L: L, height: F.H)
        let R = GlobeMap.radius(L: L)
        let rho = R / 1e-4
        let exact = Double(R) - Double(rho) * 2 * pow(sin(Double(Morph.M) * 100 / Double(rho) / 2), 2)
        #expect(abs(Double(p.z) - exact) < 0.01, "z \(p.z) against \(exact)")
    }

    // MARK: Levels and limits with the globe

    @Test func theGlobeSetsTheDistanceLimitHomeAndTheLevel() throws {
        let far = WorldPose.maxDistance(size: L, fovY: F.fov, aspect: F.aspect, tilt: 1.2, globe: true)
        #expect(far == Morph.maxDistanceFraction * L)
        #expect(WorldPose.maxDistance(size: L, fovY: F.fov, aspect: F.aspect, tilt: 1.2) < far)
        let header = try #require(try PlayFixture(kind: "3d").events().compactMap { event -> StreamHeader? in
            if case .header(let h) = event { h } else { nil }
        }.first)
        let size = Float(header.size)
        let home = WorldPose.home(for: header, aspect: F.aspect, globe: true)
        #expect(abs(home.distance - Morph.globeDistance(L: size, fovY: F.fov, aspect: F.aspect)) < 1e-3)
        #expect(Morph.s(distance: home.distance, L: size, fovY: F.fov, aspect: F.aspect) == 1)
        let flatHome = WorldPose.home(for: header, aspect: F.aspect)
        #expect(flatHome.distance > home.distance, "the flat Home is the whole width of one world")
        var level = WorldProjector(pose: WorldPose(x: 0, y: 0, distance: 0.1 * L), size: L, height: F.H, view: F.view, globe: true)
        #expect(level.level == .organism)
        level = WorldProjector(pose: WorldPose(x: 0, y: 0, distance: Morph.globeDistance(L: L, fovY: F.fov, aspect: F.aspect)),
                               size: L, height: F.H, view: F.view, globe: true)
        #expect(level.level == .planet)
        level = WorldProjector(pose: WorldPose(x: 0, y: 0, distance: Morph.distance(forS: 0.3, L: L, fovY: F.fov, aspect: F.aspect)),
                               size: L, height: F.H, view: F.view, globe: true)
        #expect(level.level == .region)
    }

    @Test func theRingOfAStormIsTheConformalCircleOnTheGlobe() {
        // At Home a unit of the torus at latitude phi is M cos(phi) sphere units, foreshortened by the view: a circle at the centre is
        // round (flatten 1) and its radius in points is that of the sphere at the centre; far from the centre it is an ellipse.
        let v = F.make(s: 1, ybar: 0)
        let projector = WorldProjector(pose: v.pose, size: L, height: F.H, view: F.view, globe: true)
        let centre = SIMD3<Float>(118, 0, 0)
        let units = projector.points(ofUnits: 10, at: centre)!
        let expected = CGFloat(10 * Morph.M) * CGFloat(0.5 * Float(F.view.height) / tan(F.fov / 2)) / CGFloat(v.camera.distance)
        #expect(abs(units - expected) / expected < 0.01, "\(units) against \(expected)")
        #expect(abs(projector.flatten(at: centre) - 1) < 0.01)
        let edge = SIMD3<Float>(118 + 55, 0, 0)
        let ring = projector.flatten(at: edge)
        #expect(ring < 0.9 && ring >= 0.12, "foreshortened toward the limb: \(ring)")
        // at the same row the pole's side shrinks by cos(phi) in torus units: the unit is shorter in points
        let north = SIMD3<Float>(118, 100, 0)
        #expect(projector.points(ofUnits: 10, at: north)! < units)
    }

    @Test func aPickOfTheOrganismsOnTheGlobeFindsTheNearSideRowOnly() throws {
        let near = SIMD3<Float>(118, 0, 10), far = SIMD3<Float>(118 + L / 2, 0, 10)
        let store = try WorldFixtures.store(radius: 1, positions: [near, far])
        let v = F.make(s: 0.3, ybar: 0, distance: 0.12 * L * 1.2)
        let projector = WorldProjector(pose: v.pose, size: L, height: F.H, view: F.view, globe: true)
        let centre = try #require(projector.screen(near))
        let hit = WorldPicker.organism(at: centre, projector: projector, store: store)
        #expect(hit == .organism(uid: store.frame!.uid[0], index: 0))
        #expect(projector.screen(far) == nil || WorldPicker.organism(at: projector.screen(far)!, projector: projector, store: store) != .organism(uid: store.frame!.uid[1], index: 1))
    }
}
