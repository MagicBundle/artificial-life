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
    @Test(arguments: [(20.0, 30.0, 0.0), (-60.0, 200.0, 0.0), (89.0, 0.0, 0.0), (0.0, -90.0, 0.9)])
    func everyVisibleCellReadsBackItsOwnColour(lat: Double, lon: Double, tilt: Double) throws {
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
        let size = 640
        let image = try renderer.renderOffscreen(width: size, height: size)
        try write(image, "lookup_\(Int(lat))_\(Int(lon))_\(Int(tilt * 10))")
        let centres = SphereCentres.shared.centres(nside: sphere.nsideField)
        var checked = 0
        for (p, c) in centres.enumerated() {
            let world = c * Float(sphere.planetRadius)
            guard !camera.isOccluded(world), let ndc = camera.ndc(of: world, aspect: 1), abs(ndc.x) < 0.95, abs(ndc.y) < 0.95 else { continue }
            // only cells that face the eye by more than 25 degrees (a grazing cell is a sliver the centre of which may land on its neighbour)
            guard simd_dot(c, simd_normalize(camera.eye - world)) > 0.42 else { continue }
            let x = Int((ndc.x * 0.5 + 0.5) * Float(size)), y = Int((0.5 - ndc.y * 0.5) * Float(size))
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
            try write(image, "sphere_\(name)")
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
