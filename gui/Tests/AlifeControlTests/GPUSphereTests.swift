import AlifeCore
import Foundation
import Metal
import Testing
import simd

@testable import AlifeControl

/// Compiles the sphere's shaders and renders the sphere world offscreen on the GPU (DESIGN §46.8).
///
/// Opt in with `ALIFE_GPU_TESTS=1 swift test --filter GPUSphereTests`; skipped otherwise, so `swift test` never creates a Metal device. Images
/// are written under `$ALIFE_GPU_TESTS_OUT` or the temporary directory, for looking at. `ALIFE_SPHERE_STREAM` names a protocol 2 play
/// stream to draw instead of the committed fixture (the pictures of a real world).
@MainActor
@Suite(
    "GPU rendering of the sphere",
    .enabled(
        if: ProcessInfo.processInfo.environment["ALIFE_GPU_TESTS"] == "1",
        "set ALIFE_GPU_TESTS=1 to compile the shaders and render on the GPU")
)
struct GPUSphereTests {
    private func makeRenderer() throws -> SphereRenderer {
        let device = try #require(MTLCreateSystemDefaultDevice(), "no Metal device")
        return try SphereRenderer(device: device)
    }

    private func outputDirectory() throws -> URL {
        let base = ProcessInfo.processInfo.environment["ALIFE_GPU_TESTS_OUT"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.temporaryDirectory.appendingPathComponent("alife-gpu-tests", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    /// The store of the stream: the fixture, or `$ALIFE_SPHERE_STREAM`, every record applied.
    private func store() throws -> WorldStore {
        let url = ProcessInfo.processInfo.environment["ALIFE_SPHERE_STREAM"].map { URL(fileURLWithPath: $0) }
            ?? Repo.streamData.appendingPathComponent("play_sphere.bin")
        var decoder = FrameDecoder()
        let events = try decoder.feed(try Data(contentsOf: url))
        guard case .header(let header)? = events.first else { throw StreamError.badMagic }
        var store = WorldStore()
        store.reset(header: header)
        for case .record(let record) in events { store.apply(record) }
        return store
    }

    /// A renderer set up for `store` with the composed scene, looking from `pose`.
    private func configured(_ store: WorldStore, pose: SpherePose, display: WorldDisplay = WorldDisplay(),
                            unlit: Bool = false) throws -> (renderer: SphereRenderer, camera: SphereCamera) {
        let renderer = try makeRenderer()
        let sphere = try #require(store.header?.sphere)
        let camera = SphereCamera(pose: pose, planetRadius: Float(sphere.planetRadius), lidRadius: Float(sphere.lidRadius))
        renderer.configure(sphere: sphere, bodyRadius: Float(store.header?.radius ?? 1), camera: camera)
        var options = SphereScene.Options()
        options.display = display
        let scene = try #require(SphereScene.compose(store: store, options: options))
        renderer.setFloor(scene.floor)
        renderer.setCloud(scene.cloud)
        renderer.setTerritory(scene.territory)
        renderer.setOverlay(scene.overlay)
        renderer.setTop(scene.top)
        renderer.show(frame: store.frame)
        var draw = SphereDraw(camera: camera)
        draw.display = display
        draw.unlit = unlit
        renderer.setDraw(draw)
        return (renderer, camera)
    }

    private func write(_ image: OffscreenImage, _ name: String) throws {
        let url = try outputDirectory().appendingPathComponent("\(name).png")
        try PNG.write(image, to: url)
        print("GPUSphereTests: wrote \(url.path)")
    }

    private func isBackground(_ p: (r: UInt8, g: UInt8, b: UInt8)) -> Bool {
        abs(Int(p.r) - 4) <= 12 && abs(Int(p.g) - 5) <= 12 && abs(Int(p.b) - 8) <= 14
    }

    @Test func theShadersCompileAndTheLookupIsTheEnginesAndTheSwiftMirrors() throws {
        let renderer = try makeRenderer()
        let all = try #require(SphereFixture.expected["healpix"] as? [String: Any])
        for nside in [2, 4, 8, 21] {
            let t = try #require(all["\(nside)"] as? [String: Any])
            let points = try #require(t["points"] as? [Any]).map { v -> SIMD3<Float> in
                let d = SphereFixture.vector(v)
                return SIMD3(Float(d.x), Float(d.y), Float(d.z))
            }
            let want = try #require(t["pixels"] as? [Int])
            let gpu = try renderer.healpixOnGPU(points, nside: nside)
            var different = 0
            for i in points.indices {
                let swift = Healpix.pixelFloat(of: points[i], nside: nside)
                if Int(gpu[i]) != swift { different += 1 }
                if Int(gpu[i]) != want[i] {
                    // a border case: the neighbour pixel is acceptable, nothing farther
                    let a = Healpix.centre(of: Int(gpu[i]), nside: nside), b = Healpix.centre(of: want[i], nside: nside)
                    #expect(acos(min(1, simd_dot(a, b))) < 2.2 * (4 * Double.pi / Double(Healpix.npix(nside))).squareRoot())
                }
            }
            #expect(different <= 2, "Nside \(nside): the GPU and the Swift mirror differ on \(different) of \(points.count)")
        }
    }

    @Test func theHeadingOnTheGPUIsTheSwiftHeading() throws {
        let renderer = try makeRenderer()
        var positions: [SIMD3<Float>] = [SIMD3(0, 0, 21), SIMD3(0, 0, -23), SIMD3(15, 5, 12), SIMD3(-8, 1, -17), SIMD3(20, -20, 1)]
        var bearings: [SIMD2<Float>] = [SIMD2(0.3, 0.1), SIMD2(-2.5, -0.4), SIMD2(1, 0.5), SIMD2(3, -1), SIMD2(0, 1.4)]
        var rng = SystemRandomNumberGenerator()
        for _ in 0..<200 {
            positions.append(SIMD3(Float.random(in: -30...30, using: &rng), Float.random(in: -30...30, using: &rng), Float.random(in: -30...30, using: &rng)))
            bearings.append(SIMD2(Float.random(in: -3.1...3.1, using: &rng), Float.random(in: -1.4...1.4, using: &rng)))
        }
        let gpu = try renderer.headingsOnGPU(positions: positions, bearings: bearings)
        for i in positions.indices {
            let swift = SphereFrame.heading(at: positions[i], yaw: bearings[i].x, pitch: bearings[i].y)
            #expect(simd_length(gpu[i] - swift) < 2e-5, "agent \(i) at \(positions[i])")
        }
    }

    /// The strongest end-to-end check of the lookup: every cell of the floor gets its own colour (its number), the planet is drawn unlit from
    /// a few views, and the pixel at the projected centre of each visible cell must read back that number.
    /// The window is wide, so the views are 960 x 600 as well as square: a ray direction that ignored the aspect would read cells back
    /// at the wrong places there.
    @Test(arguments: [(20.0, 30.0, 0.0, 640, 640), (-60.0, 200.0, 0.0, 640, 640), (89.0, 0.0, 0.0, 640, 640), (0.0, -90.0, 0.9, 640, 640),
                      (20.0, 30.0, 0.0, 960, 600), (-60.0, 200.0, 0.4, 960, 600), (89.0, 0.0, 0.0, 960, 600), (0.0, -90.0, 0.9, 960, 600)])
    func everyVisibleCellReadsBackItsOwnColour(lat: Double, lon: Double, tilt: Double, width: Int, height: Int) throws {
        let store = try store()
        let sphere = try #require(store.header?.sphere)
        let pose = SpherePose(lat: lat, lon: lon, tilt: Float(tilt), distance: 60)
        let (renderer, camera) = try configured(store, pose: pose, unlit: true)
        renderer.setOverlay([])
        renderer.show(frame: nil)
        let n = sphere.fieldCells
        renderer.setFloor((0..<n).map { SIMD4(Float($0 & 255) / 255, Float(($0 >> 8) & 255) / 255, 0.5, 1) })
        var display = WorldDisplay()
        display.frame = false
        display.territory = .off
        display.map = false
        var draw = SphereDraw(camera: camera)
        draw.display = display
        draw.unlit = true
        draw.cellBorderHeight = 0
        renderer.setDraw(draw)
        let image = try renderer.renderOffscreen(width: width, height: height)
        let aspect = Float(width) / Float(height)
        try write(image, "lookup_\(Int(lat))_\(Int(lon))_\(Int(tilt * 10))_\(width)")
        let centres = SphereCentres.shared.centres(nside: sphere.nsideField)
        var checked = 0
        for (p, c) in centres.enumerated() {
            let world = c * Float(sphere.planetRadius)
            guard !camera.isOccluded(world), let ndc = camera.ndc(of: world, aspect: aspect), abs(ndc.x) < 0.95, abs(ndc.y) < 0.95 else { continue }
            // only cells that face the eye by more than 25 degrees (a grazing cell is a sliver the centre of which may land on its neighbour)
            guard simd_dot(c, simd_normalize(camera.eye - world)) > 0.42 else { continue }
            let x = Int((ndc.x * 0.5 + 0.5) * Float(width)), y = Int((0.5 - ndc.y * 0.5) * Float(height))
            let px = image.pixel(x, y)
            let got = Int(px.r) + 256 * Int(px.g)
            #expect(got == p && abs(Int(px.b) - 128) <= 1, "cell \(p) read back as \(got) at \(x), \(y)")
            checked += 1
        }
        #expect(checked >= 25, "only \(checked) cells were in view")
    }

    @Test func theWorldIsDrawnFromEveryDistanceAndTheHorizonCurves() throws {
        let store = try store()
        let sphere = try #require(store.header?.sphere)
        let rp = Float(sphere.planetRadius), lid = Float(sphere.lidRadius)
        let home = try #require(SphereScene.home(store: store, aspect: 1.6))
        let views: [(String, SpherePose)] = [
            ("home", home),
            ("region", SpherePose(up: home.up, heading: home.heading, tilt: 0.5, distance: 0.9 * (lid - rp) + 30)),
            ("hover", SpherePose(up: home.up, heading: home.heading, tilt: 1.15, distance: 22)),
            ("organism", SpherePose(up: home.up, heading: home.heading, tilt: 0.7, distance: 7)),
            ("pole", SpherePose(lat: 90, lon: 0, tilt: 0, distance: 120)),
            ("far-side", SpherePose(up: -home.up, heading: home.heading, tilt: 0, distance: 130)),
        ]
        for (name, pose) in views {
            let (renderer, _) = try configured(store, pose: pose)
            let image = try renderer.renderOffscreen(width: 960, height: 600)
            try write(image, "sphere_\(name)_v2")
            var lit = 0
            for y in stride(from: 0, to: image.height, by: 3) {
                for x in stride(from: 0, to: image.width, by: 3) where !isBackground(image.pixel(x, y)) { lit += 1 }
            }
            #expect(lit > 300, "\(name): the planet is not on screen")
            if name == "home" {
                // the planet is a disc: its centre is lit, the corners are sky
                #expect(!isBackground(image.pixel(480, 300)))
                #expect(isBackground(image.pixel(8, 8)) && isBackground(image.pixel(950, 590)))
            }
        }
    }

    // MARK: What the pixels must say (the shader's aspect, depth, radius and nose)

    /// A frame of organisms at `positions` with the given bearings and one colour.
    private func frame(_ positions: [SIMD3<Float>], bearings: [(Float, Float)]? = nil, rgb: UInt32 = 0x1030A0) -> PlayFrame {
        var f = TestFrames.frame(positions)
        f.rgb = [UInt32](repeating: rgb, count: positions.count)
        if let bearings {
            f.orient = bearings.flatMap { [Float16($0.0).bitPattern, Float16($0.1).bitPattern] }
        }
        return f
    }

    /// The floor's silhouette is at the analytic radius, along the row and the column through the centre, in a wide window: the shader's
    /// ray directions use the aspect, and the floor is the sphere of radius Rp.
    @Test func theFloorSilhouetteIsAtItsAnalyticRadius() throws {
        let store = try store()
        let sphere = try #require(store.header?.sphere)
        let pose = SpherePose(lat: 15, lon: 40, tilt: 0, distance: 60)
        let (renderer, camera) = try configured(store, pose: pose, unlit: true)
        renderer.show(frame: nil)
        renderer.setOverlay([])
        renderer.setFloor([SIMD4<Float>](repeating: SIMD4(0.9, 0.55, 0.2, 1), count: sphere.fieldCells))
        var display = WorldDisplay()
        (display.frame, display.territory, display.map) = (false, .off, false)
        var draw = SphereDraw(camera: camera)
        (draw.display, draw.unlit, draw.cellBorderHeight) = (display, true, 0)
        renderer.setDraw(draw)
        let (w, h) = (960, 600)
        let image = try renderer.renderOffscreen(width: w, height: h)
        func isFloor(_ x: Int, _ y: Int) -> Bool { let p = image.pixel(x, y); return p.r > 150 && p.g > 80 && p.b < 90 }
        let alpha = asin(Float(sphere.planetRadius) / simd_length(camera.eye))
        let tanHalf = tan(pose.fovY / 2)
        let halfWidth = tan(alpha) / (tanHalf * Float(w) / Float(h)) * Float(w) / 2
        let halfHeight = tan(alpha) / tanHalf * Float(h) / 2
        var right = w / 2
        while right < w - 1, isFloor(right + 1, h / 2) { right += 1 }
        var left = w / 2
        while left > 0, isFloor(left - 1, h / 2) { left -= 1 }
        var bottom = h / 2
        while bottom < h - 1, isFloor(w / 2, bottom + 1) { bottom += 1 }
        var top = h / 2
        while top > 0, isFloor(w / 2, top - 1) { top -= 1 }
        #expect(abs(Float(right - left + 1) / 2 - halfWidth) < 2, "the floor is \(right - left + 1) px wide, expected \(2 * halfWidth)")
        #expect(abs(Float(bottom - top + 1) / 2 - halfHeight) < 2, "the floor is \(bottom - top + 1) px high, expected \(2 * halfHeight)")
    }

    /// An organism just behind the limb must not show (the planet is in front of it), one just in front must: the depth of the impostor and
    /// of the floor are the same function.
    @Test func anOrganismBehindTheLimbIsHiddenAndOneInFrontIsNot() throws {
        let store = try store()
        let sphere = try #require(store.header?.sphere)
        let rp = Float(sphere.planetRadius)
        let pose = SpherePose(lat: 0, lon: 0, tilt: 0, distance: 40)
        let (renderer, camera) = try configured(store, pose: pose)
        let horizon = acos(rp / simd_length(camera.eye))
        func at(_ angle: Float, height: Float) -> SIMD3<Float> { SIMD3(cos(angle), sin(angle), 0) * (rp + height) }
        let front = at(horizon - 0.45, height: 0.3), behind = at(horizon + 0.35, height: 0.3)
        func pixel(_ frame: PlayFrame?, _ p: SIMD3<Float>) throws -> (r: UInt8, g: UInt8, b: UInt8) {
            renderer.show(frame: frame)
            let image = try renderer.renderOffscreen(width: 960, height: 600)
            let ndc = try #require(camera.ndc(of: p, aspect: 960.0 / 600.0))
            return image.pixel(Int((ndc.x * 0.5 + 0.5) * 960), Int((0.5 - ndc.y * 0.5) * 600))
        }
        func distance(_ a: (r: UInt8, g: UInt8, b: UInt8), _ b: (r: UInt8, g: UInt8, b: UInt8)) -> Int {
            abs(Int(a.r) - Int(b.r)) + abs(Int(a.g) - Int(b.g)) + abs(Int(a.b) - Int(b.b))
        }
        let none = try pixel(nil, behind), withBehind = try pixel(frame([behind]), behind)
        #expect(distance(none, withBehind) <= 3, "an organism behind the planet shows through it (\(distance(none, withBehind)))")
        let noneFront = try pixel(nil, front), withFront = try pixel(frame([front]), front)
        #expect(distance(noneFront, withFront) > 40, "an organism in front of the planet is not drawn (\(distance(noneFront, withFront)))")
    }

    /// The nose is a paler sphere on the heading. Seen from above the body hides most of it; the crescent that sticks out is on the side the
    /// transmitted bearing says (east is to the right when north is up): the pixel just beyond the body on the heading changes when the
    /// organism is drawn, the one on the opposite side does not.
    @Test func theNoseIsAlongTheHeading() throws {
        let store = try store()
        let sphere = try #require(store.header?.sphere)
        let rp = Float(sphere.planetRadius)
        let position = SIMD3<Float>(rp + 3, 0, 0)
        let pose = SpherePose(up: SIMD3(1, 0, 0), heading: SIMD3(0, 0, 1), tilt: 0, distance: 6)
        for (name, yaw) in [("east", Float.pi / 2), ("north", Float(0)), ("west", -Float.pi / 2), ("south", Float.pi)] {
            let (renderer, camera) = try configured(store, pose: pose)
            renderer.setOverlay([])
            var draw = SphereDraw(camera: camera)
            draw.display = WorldDisplay()
            draw.cappedSize = false
            renderer.setDraw(draw)
            let (w, h) = (960, 600)
            renderer.show(frame: nil)
            let empty = try renderer.renderOffscreen(width: w, height: h)
            renderer.show(frame: frame([position], bearings: [(yaw, 0)]))
            let image = try renderer.renderOffscreen(width: w, height: h)
            let heading = SphereFrame.heading(at: position, yaw: yaw, pitch: 0)
            func change(_ p: SIMD3<Float>) throws -> Int {
                let ndc = try #require(camera.ndc(of: p, aspect: Float(w) / Float(h)))
                let (x, y) = (Int((ndc.x * 0.5 + 0.5) * Float(w)), Int((0.5 - ndc.y * 0.5) * Float(h)))
                let a = image.pixel(x, y), b = empty.pixel(x, y)
                return abs(Int(a.r) - Int(b.r)) + abs(Int(a.g) - Int(b.g)) + abs(Int(a.b) - Int(b.b))
            }
            let along = try change(position + heading * 1.15), against = try change(position - heading * 1.15)
            #expect(along > 20, "the nose sticks out along the heading (\(name)): changed by \(along)")
            #expect(against < 6, "and not on the other side (\(name)): changed by \(against)")
            try write(image, "nose_\(name)")
        }
    }

    /// An overcast storm: the cloud is deep over the whole storm, and its ring and the organisms inside it must still show (the cloud is a veil
    /// of at most 45 % over them, and the ring is drawn after it).
    @Test func anOvercastStormStillShowsItsRingAndItsOrganisms() throws {
        var store = try store()
        let sphere = try #require(store.header?.sphere)
        let storm = try #require(store.storms?.storms.storms.first)
        let centre = try #require(SphereOverlay.direction(of: storm))
        var map = try #require(store.sphereMap)
        let centres = SphereCentres.shared.centres(nside: sphere.nsideField)
        for i in 0..<map.npix where SphereFrame.angle(centres[i], centre) < 0.9 { map.dc[i] = 255; map.p[i] = 200 }
        store.apply(.maph(map))
        let pose = SpherePose(up: centre, heading: SphereFrame.basis(at: centre).north, tilt: 0.5, distance: 34)
        let rp = Float(sphere.planetRadius)
        // organisms inside the ring, near the floor and aloft
        let basis = SphereFrame.basis(at: centre)
        let inside = Float(storm.r) / rp * 0.4  // well inside the ring, whatever the storm
        let around = [centre + basis.east * inside, centre - basis.east * inside * 0.7 + basis.north * inside * 0.5, centre + basis.north * -inside]
        let positions = [simd_normalize(around[0]) * (rp + 1.0), simd_normalize(around[1]) * (rp + 2.5), simd_normalize(around[2]) * (rp + 6)]
        let (renderer, camera) = try configured(store, pose: pose)
        renderer.show(frame: frame(positions, rgb: 0xF0E442))
        let (w, h) = (960, 600)
        let image = try renderer.renderOffscreen(width: w, height: h)
        try write(image, "sphere_overcast_v2")
        let ringPoint = SphereOverlay.circle(centre: centre, angle: Float(storm.r) / rp, radius: rp + 0.35, count: 48)[6]
        let ndc = try #require(camera.ndc(of: ringPoint, aspect: Float(w) / Float(h)))
        let (x, y) = (Int((ndc.x * 0.5 + 0.5) * Float(w)), Int((0.5 - ndc.y * 0.5) * Float(h)))
        renderer.setTop([])
        let without = try renderer.renderOffscreen(width: w, height: h)
        var best = 0
        for dy in -3...3 {
            for dx in -3...3 {
                let a = image.pixel(x + dx, y + dy), b = without.pixel(x + dx, y + dy)
                best = max(best, abs(Int(a.r) - Int(b.r)) + abs(Int(a.g) - Int(b.g)) + abs(Int(a.b) - Int(b.b)))
            }
        }
        #expect(best > 120, "the storm's ring does not show through the overcast (\(best))")
        // the yellow organism near the floor keeps most of its colour under the veil
        let ndcO = try #require(camera.ndc(of: positions[0], aspect: Float(w) / Float(h)))
        let o = image.pixel(Int((ndcO.x * 0.5 + 0.5) * Float(w)), Int((0.5 - ndcO.y * 0.5) * Float(h)))
        #expect(Int(o.r) > 130 && Int(o.g) > 110, "the organism is washed out by the cloud (\(o))")
    }

    @Test func anOrganismIsWhereItsPositionProjects() throws {
        let store = try store()
        let sphere = try #require(store.header?.sphere)
        let frame = try #require(store.frame)
        let target = frame.position(0)
        let up = simd_normalize(target)
        let pose = SpherePose(up: up, heading: SphereFrame.basis(at: up).north, tilt: 0, distance: 14)
        let (withAgents, camera) = try configured(store, pose: pose)
        let a = try withAgents.renderOffscreen(width: 480, height: 480)
        withAgents.show(frame: nil)
        let b = try withAgents.renderOffscreen(width: 480, height: 480)
        let ndc = try #require(camera.ndc(of: target, aspect: 1))
        let x = Int((ndc.x * 0.5 + 0.5) * 480), y = Int((0.5 - ndc.y * 0.5) * 480)
        let withPixel = a.pixel(x, y), withoutPixel = b.pixel(x, y)
        let d = abs(Int(withPixel.r) - Int(withoutPixel.r)) + abs(Int(withPixel.g) - Int(withoutPixel.g)) + abs(Int(withPixel.b) - Int(withoutPixel.b))
        #expect(d > 30, "no organism at the projection of its position (difference \(d))")
        _ = sphere
        try write(a, "organism_closeup")
    }
}

/// The engine's expected values of `tests/data/sphere_gui.json` (written by `tests/make_sphere_gui_fixture.py`).
enum SphereFixture {
    nonisolated(unsafe) static let expected: [String: Any] = {
        guard let data = try? Data(contentsOf: Repo.streamData.appendingPathComponent("sphere_gui.json")),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [:] }
        return object
    }()

    static func vector(_ any: Any?) -> SIMD3<Double> {
        let v = (any as? [Any])?.compactMap { ($0 as? NSNumber)?.doubleValue } ?? [0, 0, 0]
        return SIMD3(v[0], v[1], v[2])
    }
}
