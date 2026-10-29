import AlifeCore
import Foundation
import Metal
import Testing
import simd

@testable import AlifeControl

/// Renders the feature layers offscreen (DESIGN §45): the material as tiles (2D) and cubes (3D), the soil, and the bond flow. Opt in with
/// `ALIFE_GPU_TESTS=1 swift test --filter GPUFeatureLayerTests`; skipped otherwise, so `swift test` never creates a Metal device.
@MainActor
@Suite(
    "GPU rendering of the feature layers",
    .enabled(
        if: ProcessInfo.processInfo.environment["ALIFE_GPU_TESTS"] == "1",
        "set ALIFE_GPU_TESTS=1 to compile the shaders and render on the GPU")
)
struct GPUFeatureLayerTests {
    private func scene(_ kind: String) throws -> (renderer: Renderer, header: StreamHeader) {
        let device = try #require(MTLCreateSystemDefaultDevice(), "no Metal device")
        let renderer = try Renderer(device: device)
        var header = try #require(try PlayFixtureStore.store(kind).header)
        // The fixture is a world without the layers: give it the grids of the field lattice.
        let grid = PlayHeader.CellGrid(enabled: true, farm: nil, cap: 4096, cells: [header.fieldLayers, header.fieldGrid, header.fieldGrid],
                                       cell: header.fieldCell)
        header.play?.structures = grid
        header.play?.niche = PlayHeader.CellGrid(enabled: true, farm: true, cap: 4095, cells: grid.cells, cell: grid.cell)
        renderer.configureWorld(header: header)
        renderer.camera = Camera.world(for: header, aspect: 1)
        renderer.worldDraw.level = .region
        var display = WorldDisplay()
        (display.frame, display.map, display.agents, display.bonds) = (false, false, false, false)
        display.territory = .off
        renderer.worldDraw.display = display
        return (renderer, header)
    }

    /// Renders and writes the picture under `$ALIFE_GPU_TESTS_OUT` (or the temporary directory), for looking at.
    private func render(_ renderer: Renderer, _ name: String) throws -> OffscreenImage {
        let image = try renderer.renderOffscreen(width: 512, height: 512)
        let base = ProcessInfo.processInfo.environment["ALIFE_GPU_TESTS_OUT"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.temporaryDirectory.appendingPathComponent("alife-gpu-tests", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        try PNG.write(image, to: base.appendingPathComponent("\(name).png"))
        return image
    }

    private func isBackground(_ p: (r: UInt8, g: UInt8, b: UInt8)) -> Bool {
        abs(Int(p.r) - 9) <= 2 && abs(Int(p.g) - 10) <= 2 && abs(Int(p.b) - 14) <= 2
    }

    private func pixel(of world: SIMD3<Float>, camera: Camera, image: OffscreenImage) throws -> (x: Int, y: Int) {
        let ndc = try #require(camera.ndc(of: world, aspect: 1))
        return (Int((ndc.x * 0.5 + 0.5) * Float(image.width)), Int((0.5 - ndc.y * 0.5) * Float(image.height)))
    }

    private func cells(_ header: StreamHeader, _ at: [(Int, Int, Int)], level: UInt8) -> PlayCells {
        let n = header.fieldGrid
        let indices = at.map { UInt32(($0.2 * n + $0.1) * n + $0.0) }.sorted()
        return PlayCells(head: RecordHead(step: 1, seq: 0, version: 1, flags: 0), nx: n, ny: n, nz: header.fieldLayers, indices: indices,
                         levels: [UInt8](repeating: level, count: indices.count))
    }

    private func centre(_ header: StreamHeader, _ c: (Int, Int, Int)) -> SIMD3<Float> {
        let s = Float(header.fieldCell)
        return SIMD3((Float(c.0) + 0.5) * s, (Float(c.1) + 0.5) * s, header.dims == 3 ? (Float(c.2) + 0.5) * s : 0)
    }

    @Test func aMaterialTileIsWarmGreyOnTheFloorOf2D() throws {
        let (renderer, header) = try scene("2d")
        renderer.showWorld(material: cells(header, [(5, 5, 0)], level: 255), serial: 1)
        let image = try render(renderer, "feature-2d-material")
        let here = try pixel(of: centre(header, (5, 5, 0)), camera: renderer.camera, image: image)
        let p = image.pixel(here.x, here.y)
        #expect(!isBackground(p), "the cell is drawn: \(p)")
        #expect(p.r > 100 && p.r > p.b, "warm: \(p)")
        let elsewhere = try pixel(of: centre(header, (9, 9, 0)), camera: renderer.camera, image: image)
        #expect(isBackground(image.pixel(elsewhere.x, elsewhere.y)), "a cell that is not listed is not drawn")
    }

    @Test func aThinnerCellIsFainterAndTheToggleHidesIt() throws {
        let (renderer, header) = try scene("2d")
        renderer.showWorld(material: cells(header, [(5, 5, 0)], level: 255), serial: 1)
        renderer.showWorld(soil: cells(header, [(8, 3, 0)], level: 255), serial: 1)
        let full = try render(renderer, "feature-2d-layers")
        let at = try pixel(of: centre(header, (5, 5, 0)), camera: renderer.camera, image: full)
        let soilAt = try pixel(of: centre(header, (8, 3, 0)), camera: renderer.camera, image: full)
        let soil = full.pixel(soilAt.x, soilAt.y)
        #expect(soil.g > soil.r && soil.g > soil.b, "the soil is green: \(soil)")
        renderer.showWorld(material: cells(header, [(5, 5, 0)], level: 20), serial: 2)
        let thin = try render(renderer, "feature-2d-material-thin")
        #expect(thin.pixel(at.x, at.y).r < full.pixel(at.x, at.y).r, "level 20 is fainter than 255")
        #expect(!isBackground(thin.pixel(at.x, at.y)))
        var display = renderer.worldDraw.display
        display.structures = false
        display.soil = false
        renderer.worldDraw.display = display
        let off = try render(renderer, "feature-2d-layers-off")
        #expect(isBackground(off.pixel(at.x, at.y)) && isBackground(off.pixel(soilAt.x, soilAt.y)), "both layers are off")
    }

    @Test func aMaterialCubeStandsInA3DWorld() throws {
        let (renderer, header) = try scene("3d")
        renderer.showWorld(material: cells(header, [(5, 5, 2)], level: 255), serial: 1)
        let image = try render(renderer, "feature-3d-cube")
        let here = try pixel(of: centre(header, (5, 5, 2)), camera: renderer.camera, image: image)
        let p = image.pixel(here.x, here.y)
        #expect(!isBackground(p), "the cube is drawn: \(p)")
        // (a cube is translucent: the colour is a warm tint of the background, not the full grey)
        #expect(p.r > p.b, "warm: \(p)")
    }

    @Test func theHalvesOfABondWithFlowTakeTheDonorAndReceiverColours() throws {
        let (renderer, header) = try scene("3d")
        var frame = try #require(try PlayFixtureStore.decode("3d").records.compactMap { if case .frma(let f) = $0 { f } else { nil } }.first)
        let L = Float(header.size)
        for i in 0..<frame.n { (frame.pos[3 * i], frame.pos[3 * i + 1], frame.pos[3 * i + 2]) = (24, 4, 20) }
        let (a, b) = frame.bond(0)
        (frame.pos[3 * a], frame.pos[3 * a + 1], frame.pos[3 * a + 2]) = (14, 24, 6)
        (frame.pos[3 * b], frame.pos[3 * b + 1], frame.pos[3 * b + 2]) = (34, 24, 6)
        frame.bonds = [UInt32(a), UInt32(b)]
        frame.flow = [127]  // from the lower row (a, on the left) to the higher (b, on the right)
        renderer.showWorld(frame: frame)
        var display = renderer.worldDraw.display
        display.bonds = true
        renderer.worldDraw.display = display
        var camera = Camera(target: SIMD3(24, 24, 6), distance: 34, yaw: .pi / 2, pitch: 0)
        camera.scene = Camera.patchScene(center: SIMD2(camera.target.x, camera.target.y), size: L, height: Float(header.height))
        renderer.camera = camera
        let image = try render(renderer, "feature-bond-flow")
        func lineColour(near world: SIMD3<Float>) throws -> (r: UInt8, g: UInt8, b: UInt8)? {
            let at = try pixel(of: world, camera: camera, image: image)
            for dy in -1...1 {
                for dx in -1...1 {
                    let p = image.pixel(at.x + dx, at.y + dy)
                    if !isBackground(p) { return p }
                }
            }
            return nil
        }
        let giver = try #require(try lineColour(near: SIMD3(19, 24, 6)), "the donor half is drawn")
        let receiver = try #require(try lineColour(near: SIMD3(29, 24, 6)), "the receiver half is drawn")
        #expect(giver.r > giver.b, "the half that gives is warm: \(giver)")
        #expect(receiver.b > receiver.r, "the half that receives is cool: \(receiver)")
        // and without the flag the halves are in the colours of their agents, alike on both sides of the middle
        frame.flow = nil
        renderer.showWorld(frame: frame)
        let plain = try render(renderer, "feature-bond-plain")
        let left = plain.pixel(try pixel(of: SIMD3(19, 24, 6), camera: camera, image: plain).x, try pixel(of: SIMD3(19, 24, 6), camera: camera, image: plain).y)
        #expect(!(left.r > 200 && left.b < 80), "no orange without a flow: \(left)")
    }

    // MARK: Wrap, restart and cost

    @Test func aCellAcrossTheSeamIsDrawnAtItsMinimumImage() throws {
        let (renderer, header) = try scene("2d")
        // the last cell of a row (x 44 to 48) with the camera target just past the seam: the cell is drawn left of the target, at x 44 - 48
        renderer.showWorld(material: cells(header, [(11, 5, 0)], level: 255), serial: 1)
        var camera = Camera.world(for: header, aspect: 1)
        camera.target = SIMD3(1, 22, 0)
        renderer.camera = camera
        let image = try render(renderer, "feature-2d-seam")
        let near = try pixel(of: SIMD3(-2, 22, 0), camera: camera, image: image)
        #expect(!isBackground(image.pixel(near.x, near.y)), "drawn at its minimum image")
        let far = try pixel(of: SIMD3(46, 22, 0), camera: camera, image: image)
        #expect(far.x >= image.width || far.x < 0 || isBackground(image.pixel(far.x, far.y)), "and not at the other image")
    }

    @Test func aNewWorldDoesNotShowTheOldOnesCells() throws {
        let (renderer, header) = try scene("2d")
        renderer.showWorld(material: cells(header, [(5, 5, 0)], level: 255), serial: 1)
        let first = try render(renderer, "feature-restart-old")
        let at = try pixel(of: centre(header, (5, 5, 0)), camera: renderer.camera, image: first)
        #expect(!isBackground(first.pixel(at.x, at.y)))
        // the next world numbers its deliveries from 1 again, and has not sent a layer yet
        renderer.configureWorld(header: header)
        renderer.camera = Camera.world(for: header, aspect: 1)
        var display = renderer.worldDraw.display
        display.agents = false
        renderer.worldDraw.display = display
        let second = try render(renderer, "feature-restart-new")
        #expect(isBackground(second.pixel(at.x, at.y)), "nothing is drawn until the new world sends its own")
        // and its own first delivery (serial 1 again) is drawn
        renderer.showWorld(material: cells(header, [(9, 9, 0)], level: 255), serial: 1)
        let third = try render(renderer, "feature-restart-third")
        let there = try pixel(of: centre(header, (9, 9, 0)), camera: renderer.camera, image: third)
        #expect(!isBackground(third.pixel(there.x, there.y)) && isBackground(third.pixel(at.x, at.y)))
    }

    /// The `64k3d` lattice (80 x 80 x 20 cells of 4, the world 320 x 320 x 80) and 65 536 cells in each of the two layers: a solid slab, and
    /// a checkerboard, which is the worst case for the surface (every cell shows all six faces). Prints the cost of a frame.
    @Test func aFullSizeRecordCostsLittle() throws {
        let (renderer, base) = try scene("3d")
        var header = base
        header.size = 320
        header.height = 80
        header.fieldGrid = 80
        header.fieldLayers = 20
        header.fieldCell = 4
        header.play?.structures = PlayHeader.CellGrid(enabled: true, cap: 4096, cells: [20, 80, 80], cell: 4)
        header.play?.niche = PlayHeader.CellGrid(enabled: true, farm: true, cap: 4095, cells: [20, 80, 80], cell: 4)
        renderer.configureWorld(header: header)
        var camera = Camera(target: SIMD3(160, 160, 20), distance: 330, yaw: .pi / 4, pitch: 0.7)
        camera.scene = Camera.patchScene(center: SIMD2(160, 160), size: 320, height: 80)
        renderer.camera = camera
        renderer.worldDraw.level = .region
        var display = WorldDisplay()
        (display.frame, display.map, display.agents, display.bonds, display.territory) = (false, false, false, false, .off)
        renderer.worldDraw.display = display

        func record(_ keep: (Int, Int, Int) -> Bool, limit: Int = 65536) -> PlayCells {
            var indices: [UInt32] = []
            outer: for z in 0..<20 { for y in 0..<80 { for x in 0..<80 where keep(x, y, z) {
                if indices.count == limit { break outer }
                indices.append(UInt32((z * 80 + y) * 80 + x))
            } } }
            return PlayCells(head: RecordHead(step: 1, seq: 0, version: 1, flags: 0), nx: 80, ny: 80, nz: 20, indices: indices,
                             levels: [UInt8](repeating: 200, count: indices.count))
        }
        let slab = record { _, _, z in z < 11 }
        let checker = record { x, y, z in (x + y + z) % 2 == 0 }
        #expect(slab.count == 65536 && checker.count == 64000)

        func frameMilliseconds(_ frames: Int = 60) throws -> Double {
            _ = try renderer.renderOffscreen(width: 1600, height: 1000)  // warm up
            let start = DispatchTime.now().uptimeNanoseconds
            for _ in 0..<frames { _ = try renderer.renderOffscreen(width: 1600, height: 1000) }
            return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 / Double(frames)
        }
        let empty = min(try frameMilliseconds(), try frameMilliseconds())
        var report = ["empty \(String(format: "%.2f", empty)) ms"]
        var worst = 0.0
        var serial = 0
        let one = record({ x, y, z in x == 5 && y == 5 && z == 5 }, limit: 1)
        for (name, material, soil) in [("one cell + one cell", one, one), ("slab + checker", slab, checker), ("checker + checker", checker, checker), ("slab + slab", slab, slab)] {
            serial += 1  // (a delivery is told by its serial: it must change)
            renderer.showWorld(material: material, serial: serial)
            renderer.showWorld(soil: soil, serial: serial)
            let built = DispatchTime.now().uptimeNanoseconds
            let faces = material.exposedFaces(flat: false).codes.count + soil.exposedFaces(flat: false).codes.count
            let buildMs = Double(DispatchTime.now().uptimeNanoseconds - built) / 1e6
            let ms = min(try frameMilliseconds(), try frameMilliseconds())
            report.append("\(name): \(faces) faces (\(6 * faces) vertices), faces built in \(String(format: "%.1f", buildMs)) ms, frame \(String(format: "%.2f", ms)) ms")
            worst = max(worst, ms - empty)
        }
        print("GPUFeatureLayerTests cost (1600 x 1000, offscreen with readback, M-series): " + report.joined(separator: "; "))
        #expect(worst < 8, "the marginal cost of two full layers is a fraction of a 60 Hz frame: \(worst) ms")
    }
}
