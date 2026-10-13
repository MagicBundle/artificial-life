import AlifeCore
import Foundation
import Metal
import Testing
import simd

@testable import AlifeControl

/// Compiles the viewer's shaders and renders the committed stream fixtures offscreen on the GPU.
///
/// Opt in with `ALIFE_GPU_TESTS=1 swift test --filter GPURenderTests` (the tests are skipped otherwise, so `swift test`
/// never creates a Metal device). The images are written to a directory under the temporary directory (or
/// `$ALIFE_GPU_TESTS_OUT`), whose path is printed, for looking at. The shaders are plain MSL text compiled at run time
/// (nothing else in the package needs a GPU or the Metal Toolchain), so this is the only place a typo in them shows up
/// before the viewer is opened.
@MainActor
@Suite(
    "GPU rendering",
    .enabled(
        if: ProcessInfo.processInfo.environment["ALIFE_GPU_TESTS"] == "1",
        "set ALIFE_GPU_TESTS=1 to compile the shaders and render on the GPU")
)
struct GPURenderTests {
    private static let size = 512

    private func makeRenderer() throws -> Renderer {
        let device = try #require(MTLCreateSystemDefaultDevice(), "no Metal device")
        return try Renderer(device: device)
    }

    private func outputDirectory() throws -> URL {
        let base = ProcessInfo.processInfo.environment["ALIFE_GPU_TESTS_OUT"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.temporaryDirectory.appendingPathComponent("alife-gpu-tests", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    /// Renders the renderer's scene and keeps the image as `name.png`.
    private func render(_ renderer: Renderer, _ name: String, width: Int = size, height: Int = size) throws
        -> OffscreenImage {
        let image = try renderer.renderOffscreen(width: width, height: height)
        let url = try outputDirectory().appendingPathComponent("\(name).png")
        try PNG.write(image, to: url)
        print("GPURenderTests: wrote \(url.path)")
        return image
    }

    // The clear colour is (0.035, 0.04, 0.055) of 255: 9, 10, 14.
    private func isBackground(_ p: (r: UInt8, g: UInt8, b: UInt8)) -> Bool {
        abs(Int(p.r) - 9) <= 2 && abs(Int(p.g) - 10) <= 2 && abs(Int(p.b) - 14) <= 2
    }

    private func litCount(_ image: OffscreenImage) -> Int {
        var count = 0
        for y in 0..<image.height {
            for x in 0..<image.width where !isBackground(image.pixel(x, y)) { count += 1 }
        }
        return count
    }

    /// Whether a pixel within `reach` of (x, y) is not the background.
    private func litNear(_ image: OffscreenImage, _ x: Int, _ y: Int, reach: Int = 1) -> Bool {
        for dy in -reach...reach {
            for dx in -reach...reach {
                let (px, py) = (x + dx, y + dy)
                if px >= 0, py >= 0, px < image.width, py < image.height, !isBackground(image.pixel(px, py)) {
                    return true
                }
            }
        }
        return false
    }

    private func pixel(of world: SIMD3<Float>, camera: Camera, image: OffscreenImage) throws -> (x: Int, y: Int) {
        let ndc = try #require(camera.ndc(of: world, aspect: Float(image.width) / Float(image.height)))
        return (Int((ndc.x * 0.5 + 0.5) * Float(image.width)), Int((0.5 - ndc.y * 0.5) * Float(image.height)))
    }

    private func scene(_ kind: String, frame index: Int = 0, display: ViewerDisplay = ViewerDisplay())
        throws -> (Renderer, StreamFixtureData, StreamFrame) {
        let renderer = try makeRenderer()
        let fixture = try StreamFixtureData(kind: kind)
        renderer.configure(header: fixture.header, aspect: 1)
        let frame = fixture.frames[index]
        renderer.show(frame: frame)
        renderer.display = display
        return (renderer, fixture, frame)
    }

    // MARK: Tests

    @Test func theShadersCompileAndEveryPipelineBuilds() throws {
        _ = try makeRenderer()
    }

    @Test func theLayersAreIndependent() throws {
        let (renderer, _, frame) = try scene("3d")
        var none = ViewerDisplay()
        (none.agents, none.bonds, none.frame) = (false, false, false)
        renderer.display = none
        #expect(litCount(try render(renderer, "3d-nothing")) == 0)

        renderer.display = ViewerDisplay(agents: true, bonds: false, frame: false)
        let agents = litCount(try render(renderer, "3d-agents"))
        #expect(agents >= frame.nAlive)

        renderer.display = ViewerDisplay(agents: false, bonds: true, frame: false)
        let bonds = litCount(try render(renderer, "3d-bonds"))
        #expect(bonds > 0)

        renderer.display = ViewerDisplay(agents: false, bonds: false, frame: true)
        let box = litCount(try render(renderer, "3d-frame"))
        #expect(box > 0)

        renderer.display = ViewerDisplay()
        let everything = litCount(try render(renderer, "3d-everything"))
        #expect(everything > agents)
    }

    @Test(arguments: ["2d", "3d"])
    func everyAliveAgentIsDrawnWhereTheCameraSaysItIs(kind: String) throws {
        let (renderer, fixture, frame) = try scene(kind, display: ViewerDisplay(agents: true, bonds: false, frame: false))
        let image = try render(renderer, "\(kind)-agents")
        let camera = renderer.camera
        #expect(camera == Camera.home(for: fixture.header, aspect: 1))
        var checked = 0
        for row in 0..<frame.rows where frame.isAlive(row) {
            let at = try pixel(of: frame.position(row), camera: camera, image: image)
            #expect(litNear(image, at.x, at.y), "agent \(row) at \(frame.position(row)) -> \(at)")
            checked += 1
        }
        #expect(checked == frame.nAlive)
    }

    /// The picture of the agents, pixel by pixel around each of them and on a sparse grid elsewhere, against
    /// `ReferenceSphere`, a CPU port of the sphere shader: the colour channels (a red agent is red, not blue), the
    /// headlight, which half-precision value is the yaw and which the pitch, the forward axis, and the nose (on the side
    /// the agent faces, and only there). The pixels within a hair of an edge are left out (`steady`).
    @Test(arguments: ["2d", "3d"])
    func eachAgentShowsItsColourItsHeadingAndItsNose(kind: String) throws {
        let (renderer, fixture, frame) = try scene(kind, display: ViewerDisplay(agents: true, bonds: false, frame: false))
        let image = try render(renderer, "\(kind)-shading")
        let spheres = (0..<frame.rows).filter(frame.isAlive).map {
            ReferenceSphere(frame: frame, row: $0, radius: Float(fixture.header.radius))
        }
        let rays = PixelRays(camera: renderer.camera, width: image.width, height: image.height)
        var pixels = pixelsAround(frame, camera: renderer.camera, size: Self.size, half: shadingHalfWidth(kind))
        for y in stride(from: 5, to: Self.size, by: 16) {
            for x in stride(from: 5, to: Self.size, by: 16) { pixels.append((x, y)) }
        }
        var compared = 0, lit = 0
        var seenRows = Set<Int>()
        var mismatches: [String] = []
        for p in pixels {
            let expected = referencePixel(spheres, rays: rays, x: p.x, y: p.y)
            guard expected.steady else { continue }
            compared += 1
            let seen = image.pixel(p.x, p.y)
            let agree: Bool
            if let color = expected.color {
                lit += 1
                if let nearest = expected.nearest { seenRows.insert(nearest) }
                let want = color * 255
                agree = abs(Float(seen.r) - want.x) <= 8 && abs(Float(seen.g) - want.y) <= 8
                    && abs(Float(seen.b) - want.z) <= 8
            } else {
                agree = isBackground(seen)
            }
            if !agree, mismatches.count < 6 {
                mismatches.append("pixel (\(p.x), \(p.y)) is \(seen), expected \(String(describing: expected.color.map { $0 * 255 }))"
                                  + " of agent \(String(describing: expected.nearest))")
            }
        }
        #expect(compared > 1500 && lit > 300, "only \(compared) pixels (\(lit) lit) were comparable")
        #expect(seenRows.count == spheres.count, "agents \(seenRows) of \(spheres.count) were compared")
        #expect(mismatches.isEmpty, "\(mismatches.joined(separator: "; "))")
    }

    @Test func aBondAcrossThePeriodicSeamIsDrawnAsTwoStubsAndNotAcrossTheWorld() throws {
        let (renderer, fixture, frame) = try scene("2d", display: ViewerDisplay(agents: false, bonds: true, frame: false))
        let image = try render(renderer, "2d-bonds", width: 768, height: 768)
        let camera = renderer.camera
        let world = Float(fixture.header.size)

        var seam: (a: SIMD3<Float>, b: SIMD3<Float>)?
        for k in 0..<frame.nBonds {
            let bond = frame.bond(k)
            let (a, b) = (frame.position(bond.i), frame.position(bond.j))
            if abs(b.x - a.x) > world / 2 || abs(b.y - a.y) > world / 2 { seam = (a, b) }
        }
        let (a, b) = try #require(seam, "the fixture has a bond across the seam")
        var d = b - a
        d.x -= world * (d.x / world).rounded(.toNearestOrEven)
        d.y -= world * (d.y / world).rounded(.toNearestOrEven)

        func lit(from p: SIMD3<Float>, to q: SIMD3<Float>, _ range: ClosedRange<Float>) throws -> Double {
            var hits = 0
            let samples = 24
            for i in 0..<samples {
                let t = range.lowerBound + (range.upperBound - range.lowerBound) * Float(i) / Float(samples - 1)
                let at = try pixel(of: p + (q - p) * t, camera: camera, image: image)
                if litNear(image, at.x, at.y) { hits += 1 }
            }
            return Double(hits) / Double(samples)
        }
        // The two half segments: from each end to the midpoint of the short way round.
        #expect(try lit(from: a, to: a + d / 2, 0.05...0.95) > 0.8)
        #expect(try lit(from: b, to: b - d / 2, 0.05...0.95) > 0.8)
        // The long way round, across the middle of the world, is not a bond.
        #expect(try lit(from: a, to: b, 0.3...0.7) < 0.3)
    }

    @Test func the2DFieldIsTheBackgroundAndKeepsItsOrientation() throws {
        let (renderer, fixture, frame) = try scene("2d", frame: 1, display: ViewerDisplay(agents: false, bonds: false, frame: false))
        let field = try #require(frame.field, "the fixture's frame 1 has a field section")
        let image = try render(renderer, "2d-field")
        let camera = renderer.camera
        let cells = fixture.header.fieldGrid
        let cell = Float(fixture.header.fieldCell)
        // The cells with the most colour, so that a swapped axis cannot hide.
        let brightest = field.indices.sorted { brightness(field[$0]) > brightness(field[$1]) }.prefix(12)
        var checked = 0
        for index in brightest {
            let (cx, cy) = (index % cells, index / cells)
            let at = try pixel(of: SIMD3((Float(cx) + 0.5) * cell, (Float(cy) + 0.5) * cell, 0), camera: camera, image: image)
            let seen = image.pixel(at.x, at.y)
            let want = field[index]
            #expect(abs(Int(seen.r) - Int((want >> 16) & 255)) <= 2, "cell (\(cx), \(cy)) red")
            #expect(abs(Int(seen.g) - Int((want >> 8) & 255)) <= 2, "cell (\(cx), \(cy)) green")
            #expect(abs(Int(seen.b) - Int(want & 255)) <= 2, "cell (\(cx), \(cy)) blue")
            checked += 1
        }
        #expect(checked == 12)
    }

    /// The fixture has a slice at layer 0 and one at layer 5 of 6 (z = 22): both must land at their own height.
    @Test(arguments: [true, false])
    func the3DSliceIsSeeThroughAtItsLayer(firstSlice: Bool) throws {
        let fixture = try StreamFixtureData(kind: "3d")
        let index = try #require(firstSlice ? fixture.frames.firstIndex { $0.hasField }
                                            : fixture.frames.lastIndex { $0.hasField })
        let (renderer, _, frame) = try scene("3d", frame: index, display: ViewerDisplay(agents: false, bonds: false, frame: false))
        let field = try #require(frame.field)
        let layer = try #require(frame.fieldLayer)
        let image = try render(renderer, firstSlice ? "3d-slice" : "3d-slice-upper")
        let cells = fixture.header.fieldGrid
        let cell = Float(fixture.header.fieldCell)
        var checked = 0
        for index in field.indices.sorted(by: { brightness(field[$0]) > brightness(field[$1]) }).prefix(12) {
            let (cx, cy) = (index % cells, index / cells)
            let centre = SIMD3((Float(cx) + 0.5) * cell, (Float(cy) + 0.5) * cell, (Float(layer) + 0.5) * cell)
            guard let ndc = renderer.camera.ndc(of: centre, aspect: 1), abs(ndc.x) < 0.9, abs(ndc.y) < 0.9 else { continue }
            let at = try pixel(of: centre, camera: renderer.camera, image: image)
            let seen = image.pixel(at.x, at.y)
            let want = field[index]
            // 0.6 of the colour over the background, blended in 8 bits.
            func blend(_ channel: UInt32, _ bg: Double) -> Int { Int((0.6 * Double(channel) + 0.4 * bg).rounded()) }
            #expect(abs(Int(seen.r) - blend((want >> 16) & 255, 9)) <= 3, "cell (\(cx), \(cy)) red")
            #expect(abs(Int(seen.g) - blend((want >> 8) & 255, 10)) <= 3, "cell (\(cx), \(cy)) green")
            #expect(abs(Int(seen.b) - blend(want & 255, 14)) <= 3, "cell (\(cx), \(cy)) blue")
            checked += 1
        }
        #expect(checked > 0)
    }

    @Test func aSnapshotIsAPNGOfTheViewersSize() throws {
        let (renderer, _, _) = try scene("3d")
        let image = try render(renderer, "3d-snapshot", width: 640, height: 400)
        #expect(image.width == 640 && image.height == 400)
        let url = try outputDirectory().appendingPathComponent("3d-snapshot.png")
        let bytes = try Data(contentsOf: url)
        #expect(Array(bytes.prefix(8)) == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
    }

    /// How far around an agent's centre `eachAgentShowsItsColourItsHeadingAndItsNose` looks: its body and nose reach 1.25
    /// radii, which are about 10 pixels in the face-on picture of the 2D fixture and 6 in the overview of the 3D one.
    private func shadingHalfWidth(_ kind: String) -> Int { kind == "2d" ? 16 : 11 }

    private func brightness(_ color: UInt32) -> UInt32 {
        ((color >> 16) & 255) + ((color >> 8) & 255) + (color & 255)
    }
}
