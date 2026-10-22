import AlifeCore
import Foundation
import Metal
import Testing
import simd

@testable import AlifeControl

/// Compiles the World window's shaders and renders the committed play fixtures offscreen on the GPU (DESIGN §33.4.6).
///
/// Opt in with `ALIFE_GPU_TESTS=1 swift test --filter GPUWorldTests`; skipped otherwise, so `swift test` never creates a
/// Metal device. Images are written under `$ALIFE_GPU_TESTS_OUT` or the temporary directory, for looking at.
@MainActor
@Suite(
    "GPU rendering of the World",
    .enabled(
        if: ProcessInfo.processInfo.environment["ALIFE_GPU_TESTS"] == "1",
        "set ALIFE_GPU_TESTS=1 to compile the shaders and render on the GPU")
)
struct GPUWorldTests {
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

    private func render(_ renderer: Renderer, _ name: String, width: Int = 512, height: Int = 512) throws -> OffscreenImage {
        let image = try renderer.renderOffscreen(width: width, height: height)
        let url = try outputDirectory().appendingPathComponent("\(name).png")
        try PNG.write(image, to: url)
        print("GPUWorldTests: wrote \(url.path)")
        return image
    }

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

    private func litNear(_ image: OffscreenImage, _ x: Int, _ y: Int, reach: Int = 1) -> Bool {
        for dy in -reach...reach {
            for dx in -reach...reach {
                let (px, py) = (x + dx, y + dy)
                if px >= 0, py >= 0, px < image.width, py < image.height, !isBackground(image.pixel(px, py)) { return true }
            }
        }
        return false
    }

    /// Whether a pixel is inside the picture with `margin` to spare (a row near the edge may be half out of view).
    private func inside(_ p: (x: Int, y: Int), _ image: OffscreenImage, margin: Int = 6) -> Bool {
        p.x >= margin && p.y >= margin && p.x < image.width - margin && p.y < image.height - margin
    }

    private func pixel(of world: SIMD3<Float>, camera: Camera, image: OffscreenImage) throws -> (x: Int, y: Int) {
        let ndc = try #require(camera.ndc(of: world, aspect: Float(image.width) / Float(image.height)))
        return (Int((ndc.x * 0.5 + 0.5) * Float(image.width)), Int((0.5 - ndc.y * 0.5) * Float(image.height)))
    }

    /// A renderer showing the first frame of a play fixture: no map, no territory, agents as instances at the region level.
    private func scene(_ kind: String = "3d") throws
        -> (renderer: Renderer, header: StreamHeader, store: WorldStore) {
        let renderer = try makeRenderer()
        let store = try PlayFixtureStore.store(kind)
        let header = try #require(store.header)
        renderer.configureWorld(header: header)
        renderer.camera = Camera.world(for: header, aspect: 1)
        renderer.worldDraw.level = .region
        var display = WorldDisplay()
        (display.frame, display.map, display.territory) = (false, false, false)
        renderer.worldDraw.display = display
        return (renderer, header, store)
    }

    /// The fixture's first frame (step 100): seven agents and three bonds.
    private func firstFrame(_ kind: String = "3d") throws -> PlayFrame {
        let records = try PlayFixtureStore.decode(kind).records
        return try #require(records.compactMap { if case .frma(let f) = $0 { f } else { nil } }.first)
    }

    // MARK: Pipelines

    @Test func theWorldShadersCompileAndEveryPipelineBuilds() throws {
        _ = try makeRenderer()
    }

    // MARK: Agents

    @Test(arguments: ["2d", "3d"])
    func everyListedRowIsDrawnWhereTheCameraSaysItIs(kind: String) throws {
        let (renderer, _, _) = try scene(kind)
        let frame = try firstFrame(kind)
        renderer.showWorld(frame: frame)
        let image = try render(renderer, "world-\(kind)-agents")
        #expect(frame.n > 0)
        var checked = 0
        for i in 0..<frame.n {
            let at = try pixel(of: frame.position(i), camera: renderer.camera, image: image)
            guard inside(at, image) else { continue }  // perspective: a row high above the floor near the edge is out of view
            #expect(litNear(image, at.x, at.y), "listed row \(i) at \(frame.position(i)) -> \(at)")
            checked += 1
        }
        #expect(checked >= frame.n - 2 && checked >= 4, "most rows are in view: \(checked) of \(frame.n)")
    }

    @Test func aRowAcrossTheSeamIsDrawnAtItsMinimumImage() throws {
        let (renderer, header, _) = try scene()
        var frame = try firstFrame()
        let L = Float(header.size)
        // One agent just inside the far edge, the camera target just inside the near one: the agent is 2 units to the left.
        // The other rows are parked at the middle of the world, 22 units from the target and out of its view.
        for i in 1..<frame.n { (frame.pos[3 * i], frame.pos[3 * i + 1], frame.pos[3 * i + 2]) = (24, 24, 12) }
        (frame.pos[0], frame.pos[1], frame.pos[2]) = (L - 0.5, 24, 6)
        renderer.showWorld(frame: frame)
        var display = renderer.worldDraw.display
        display.bonds = false  // a bond would reach the agent from its neighbour and light the pixel by itself
        renderer.worldDraw.display = display
        var camera = Camera(target: SIMD3(1.5, 24, 6), distance: 16, yaw: .pi / 2, pitch: 0)
        camera.scene = Camera.tiledScene(size: L, height: Float(header.height))
        renderer.camera = camera
        let image = try render(renderer, "world-seam-agent")
        let image3 = SIMD3<Float>(-0.5, 24, 6)  // L - 0.5 less one world
        let at = try pixel(of: image3, camera: camera, image: image)
        #expect(at.x >= 0 && at.x < image.width, "the image is in view: \(at)")
        #expect(litNear(image, at.x, at.y), "drawn at its minimum image \(at)")
        let far = try pixel(of: SIMD3(L - 0.5, 24, 6), camera: camera, image: image)
        #expect(!(far.x >= 0 && far.x < image.width) || !litNear(image, far.x, far.y), "and not at the other image")
    }

    @Test(arguments: [false, true])
    func aBondAcrossTheSeamIsOneShortSegment(reversed: Bool) throws {
        let (renderer, header, _) = try scene()
        var frame = try firstFrame()
        let L = Float(header.size)
        // Two bonded agents on either side of the seam; every other row is parked at the same point (no bond reaches it).
        for i in 0..<frame.n { (frame.pos[3 * i], frame.pos[3 * i + 1], frame.pos[3 * i + 2]) = (24, 4, 20) }
        let (a, b) = frame.bond(0)
        // Either end first: the second is brought to the image nearest the first, whichever side of the seam it is on.
        (frame.pos[3 * a], frame.pos[3 * a + 1], frame.pos[3 * a + 2]) = reversed ? (1.5, 24, 6) : (L - 1.5, 24, 6)
        (frame.pos[3 * b], frame.pos[3 * b + 1], frame.pos[3 * b + 2]) = reversed ? (L - 1.5, 24, 6) : (1.5, 24, 6)
        frame.bonds = [UInt32(a), UInt32(b)]
        renderer.showWorld(frame: frame)
        var display = renderer.worldDraw.display
        display.agents = false
        renderer.worldDraw.display = display
        var camera = Camera(target: SIMD3(0, 24, 6), distance: 14, yaw: .pi / 2, pitch: 0)
        camera.scene = Camera.tiledScene(size: L, height: Float(header.height))
        renderer.camera = camera
        let image = try render(renderer, "world-seam-bond-\(reversed ? "reversed" : "forward")")
        var minX = image.width, maxX = 0, count = 0
        for y in 0..<image.height {
            for x in 0..<image.width where !isBackground(image.pixel(x, y)) {
                minX = min(minX, x)
                maxX = max(maxX, x)
                count += 1
            }
        }
        #expect(count > 0, "the bond is drawn")
        // From x = -1.5 to 1.5 about the target: a segment of 3 units, a fifth or so of the view, not a line across it.
        #expect(maxX - minX < image.width / 2, "one short segment, not across the world: \(minX)...\(maxX)")
        #expect(abs((minX + maxX) / 2 - image.width / 2) < 12, "centred on the seam")
    }

    @Test func smallAgentsAreFlatDiscsAndStillThere() throws {
        let (renderer, _, _) = try scene()
        let frame = try firstFrame()
        renderer.showWorld(frame: frame)
        var camera = renderer.camera
        camera.distance *= 6  // far away in a small picture: the body projects to under half a pixel
        renderer.camera = camera
        let image = try render(renderer, "world-flat-discs", width: 128, height: 128)
        var checked = 0
        for i in 0..<frame.n {
            let at = try pixel(of: frame.position(i), camera: camera, image: image)
            guard inside(at, image, margin: 3) else { continue }
            // A ray test on a body that small would light a pixel or none; the disc is at least 1.2 pixels in radius.
            var lit = 0
            for dy in -2...2 { for dx in -2...2 where !isBackground(image.pixel(at.x + dx, at.y + dy)) { lit += 1 } }
            #expect(lit >= 3, "row \(i) at \(at): \(lit) pixels")
            checked += 1
        }
        #expect(checked >= 4)
    }

    @Test func thePlanetLevelDrawsNoInstancesUnlessThePointsAreAskedFor() throws {
        let (renderer, _, _) = try scene()
        let frame = try firstFrame()
        renderer.showWorld(frame: frame)
        renderer.worldDraw.level = .planet
        #expect(litCount(try render(renderer, "world-planet-none")) == 0, "organisms hide the map at this level")
        var display = renderer.worldDraw.display
        display.agentPoints = true
        renderer.worldDraw.display = display
        let image = try render(renderer, "world-planet-points")
        var checked = 0
        for i in 0..<frame.n {
            let at = try pixel(of: frame.position(i), camera: renderer.camera, image: image)
            guard inside(at, image, margin: 2) else { continue }
            #expect(litNear(image, at.x, at.y, reach: 2), "point \(i)")
            checked += 1
        }
        #expect(checked >= 4)
        #expect(litCount(image) < 200, "points, not discs")
    }

    @Test func theLayerTogglesAreIndependent() throws {
        let (renderer, _, _) = try scene()
        renderer.showWorld(frame: try firstFrame())
        var d = renderer.worldDraw.display
        (d.agents, d.bonds) = (false, false)
        renderer.worldDraw.display = d
        #expect(litCount(try render(renderer, "world-off")) == 0)
        d.agents = true
        renderer.worldDraw.display = d
        let agents = litCount(try render(renderer, "world-agents-only"))
        #expect(agents > 0)
        d.bonds = true
        renderer.worldDraw.display = d
        #expect(litCount(try render(renderer, "world-agents-bonds")) > agents)
        d.frame = true
        renderer.worldDraw.display = d
        #expect(litCount(try render(renderer, "world-all")) > agents)
    }

    // MARK: The map

    /// A map of `n` cells a side: all planes zero except rain, which is full in the cell (cx, cy).
    private func mapWithRain(at cx: Int, _ cy: Int, from base: PlayWeatherMap) -> PlayWeatherMap {
        var map = base
        let n = map.n
        map.p = [UInt8](repeating: 0, count: n * n)
        map.dc = [UInt8](repeating: 0, count: n * n)
        map.zeta = [Int8](repeating: 0, count: n * n)
        map.ts = [UInt8](repeating: 0, count: n * n)
        for dy in -2...2 { for dx in -2...2 { map.p[(cy + dy) * n + cx + dx] = 255 } }
        return map
    }

    private func mapScene() throws -> (Renderer, StreamHeader, PlayWeatherMap) {
        let (renderer, header, store) = try scene()
        var display = renderer.worldDraw.display
        display.map = true
        display.agents = false
        renderer.worldDraw.display = display
        renderer.worldDraw.level = .planet
        let map = try #require(store.map)
        return (renderer, header, map)
    }

    @Test func theMapIsOrientedRowThenColumnWithYUp() throws {
        let (renderer, header, base) = try mapScene()
        let n = base.n
        let (cx, cy) = (30, 110)
        renderer.deliverWorld(map: mapWithRain(at: cx, cy, from: base), serial: 1, start: .current)
        let L = Float(header.size)
        let image = try render(renderer, "world-map-orientation")
        func blue(_ world: SIMD3<Float>) throws -> Int {
            let at = try pixel(of: world, camera: renderer.camera, image: image)
            return Int(image.pixel(at.x, at.y).b)
        }
        let cell = L / Float(n)
        let on = try blue(SIMD3((Float(cx) + 0.5) * cell, (Float(cy) + 0.5) * cell, 0))
        let swapped = try blue(SIMD3((Float(cy) + 0.5) * cell, (Float(cx) + 0.5) * cell, 0))
        let mirrored = try blue(SIMD3((Float(cx) + 0.5) * cell, L - (Float(cy) + 0.5) * cell, 0))
        #expect(on > 110, "the rain is where [cy][cx] puts it: blue \(on)")
        #expect(swapped < 60 && mirrored < 60, "and not transposed (\(swapped)) or flipped (\(mirrored))")
    }

    @Test func theMapRepeatsAcrossTheSeamSoTheTorusIsTiled() throws {
        let (renderer, header, base) = try mapScene()
        let n = base.n
        let (cx, cy) = (60, 100)
        renderer.deliverWorld(map: mapWithRain(at: cx, cy, from: base), serial: 1, start: .current)
        let L = Float(header.size)
        var camera = renderer.camera
        camera.distance = 130  // the view reaches 60 units from the middle: well past the edge of the world
        renderer.camera = camera
        let image = try render(renderer, "world-map-tiled")
        let cell = L / Float(n)
        let p = SIMD3((Float(cx) + 0.5) * cell, (Float(cy) + 0.5) * cell, 0)
        for shift in [SIMD3<Float>(L, 0, 0), SIMD3(-L, 0, 0), SIMD3(0, L, 0), SIMD3(L, L, 0)] {
            let at = try pixel(of: p + shift, camera: camera, image: image)
            guard at.x >= 0, at.x < image.width, at.y >= 0, at.y < image.height else { continue }
            #expect(image.pixel(at.x, at.y).b > 110, "the rain is repeated at \(shift): \(at)")
        }
        let home = try pixel(of: p, camera: camera, image: image)
        #expect(image.pixel(home.x, home.y).b > 110)
    }

    @Test func theTwoEquatorsAreMarked() throws {
        let (renderer, header, base) = try mapScene()
        renderer.deliverWorld(map: mapWithRain(at: 40, 40, from: base), serial: 1, start: .current)
        let L = Float(header.size)
        let image = try render(renderer, "world-map-equators", width: 768, height: 768)
        func row(y: Float) throws -> Int { try pixel(of: SIMD3(L * 0.8, y, 0), camera: renderer.camera, image: image).y }
        func brightness(_ y: Int) -> Int {
            let p = image.pixel(image.width * 4 / 5, min(max(y, 0), image.height - 1))
            return Int(p.r) + Int(p.g) + Int(p.b)
        }
        let equator = try row(y: L / 2)
        let between = try row(y: L / 4)
        let line = (-2...2).map { brightness(equator + $0) }.max() ?? 0
        #expect(line > brightness(between) + 25, "a thin line at y = L/2: \(line) against \(brightness(between))")
    }

    @Test func theFadeBlendsThePreviousAndTheCurrentMap() throws {
        let (renderer, header, base) = try mapScene()
        let n = base.n
        let L = Float(header.size)
        renderer.deliverWorld(map: mapWithRain(at: 30, 110, from: base), serial: 1, start: .current)
        renderer.deliverWorld(map: mapWithRain(at: 110, 30, from: base), serial: 2, start: .current)
        let cell = L / Float(n)
        func blue(at cx: Int, _ cy: Int, blend: Float) throws -> Int {
            renderer.worldDraw.mapBlend = blend
            let image = try render(renderer, "world-map-fade-\(Int(blend * 10))")
            let at = try pixel(of: SIMD3((Float(cx) + 0.5) * cell, (Float(cy) + 0.5) * cell, 0), camera: renderer.camera, image: image)
            return Int(image.pixel(at.x, at.y).b)
        }
        let (old0, new0) = (try blue(at: 30, 110, blend: 0), try blue(at: 110, 30, blend: 0))
        let (old1, new1) = (try blue(at: 30, 110, blend: 1), try blue(at: 110, 30, blend: 1))
        let (oldHalf, newHalf) = (try blue(at: 30, 110, blend: 0.5), try blue(at: 110, 30, blend: 0.5))
        #expect(old0 > 110 && new0 < 60, "before the fade, the previous map: \(old0), \(new0)")
        #expect(old1 < 60 && new1 > 110, "after it, the current one: \(old1), \(new1)")
        #expect(oldHalf > 40 && oldHalf < old0 && newHalf > 40 && newHalf < new1, "halfway, both: \(oldHalf), \(newHalf)")
    }

    @Test func eachBufferSetHoldsItsOwnMapTexturesAndTheyAreWrittenUnderTheTicket() throws {
        let (renderer, header, base) = try mapScene()
        let L = Float(header.size), cell = L / Float(base.n)
        // The draws walk the ring of three sets, and a set is refreshed again after every delivery. Map k has its rain in
        // its own cell, so what a set shows at blend 0 (the previous map) and at blend 1 (the current one) says which
        // textures it holds.
        func rainCell(_ k: Int) -> (x: Int, y: Int) { (20 + 20 * k, 40 + 25 * k) }
        func blue(map k: Int, blend: Float) throws -> Int {
            renderer.worldDraw.mapBlend = blend
            let image = try render(renderer, "world-map-slot-\(blend)", width: 256, height: 256)
            let (cx, cy) = rainCell(k)
            let at = try pixel(of: SIMD3((Float(cx) + 0.5) * cell, (Float(cy) + 0.5) * cell, 0), camera: renderer.camera, image: image)
            return Int(image.pixel(at.x, at.y).b)
        }
        var before: [[ObjectIdentifier]] = []
        for serial in 1...3 {
            let (cx, cy) = rainCell(serial)
            renderer.deliverWorld(map: mapWithRain(at: cx, cy, from: base), serial: serial, start: .current)
            for _ in 0..<3 { _ = try render(renderer, "world-map-slot", width: 64, height: 64) }  // every set refreshed
            if serial == 2 { before = renderer.mapTextureIdentities }
        }
        #expect(renderer.mapSerialsBySlot.allSatisfy { $0.current == 3 && $0.previous == 2 }, "\(renderer.mapSerialsBySlot)")
        let all = renderer.mapTextureIdentities.flatMap { $0 }
        #expect(all.count == Set(all).count && !all.isEmpty, "no two sets share a texture")
        #expect(renderer.mapTextureIdentities.map(Set.init) == before.map(Set.init), "textures are written in place, never replaced under a draw")
        // Whatever set the next draw has, it shows map 2 at blend 0 and map 3 at blend 1 (a set that swapped must still agree).
        for _ in 0..<Renderer.slotCount {
            #expect(try blue(map: 2, blend: 0) > 110 && blue(map: 3, blend: 0) < 60, "previous map")
            #expect(try blue(map: 3, blend: 1) > 110 && blue(map: 2, blend: 1) < 60, "current map")
        }
    }

    // MARK: Territory

    @Test func theTerritoryTintsItsCellsWithTheKinPalette() throws {
        let (renderer, header, store) = try scene()
        var display = renderer.worldDraw.display
        display.territory = true
        renderer.worldDraw.display = display
        renderer.worldDraw.level = .planet
        let territory = try #require(store.territory)
        renderer.showWorld(territory: territory, serial: 1)
        let L = Float(header.size)
        let image = try render(renderer, "world-territory")
        // Row 3 of the fixture: cells 0 to 10 belong to slot 0 (orange), 11 to 21 to slot 1 (sky blue), then nobody.
        func colour(cell: Int, row: Int) throws -> (r: UInt8, g: UInt8, b: UInt8) {
            let at = try pixel(of: SIMD3((Float(cell) + 0.5) * L / Float(territory.nx), (Float(row) + 0.5) * L / Float(territory.ny), 0),
                               camera: renderer.camera, image: image)
            return image.pixel(at.x, at.y)
        }
        let orange = try colour(cell: 5, row: 3)
        let blue = try colour(cell: 16, row: 3)
        let none = try colour(cell: 28, row: 3)
        // No owner, no tint: the floor of a world without weather (0.045, 0.055, 0.075 of 255).
        #expect(abs(Int(none.r) - 11) <= 3 && abs(Int(none.g) - 14) <= 3 && abs(Int(none.b) - 19) <= 3, "no owner: \(none)")
        #expect(orange.r > orange.b + 8 && orange.r > 20, "slot 0 is orange: \(orange)")
        #expect(blue.b > blue.r + 8, "slot 1 is sky blue: \(blue)")
    }

    // MARK: Whole picture

    @Test func theWholeSceneDraws() throws {
        let (renderer, _, store) = try scene()
        var display = WorldDisplay()
        display.frame = true
        renderer.worldDraw.display = display
        renderer.worldDraw.level = .region
        renderer.showWorld(frame: try firstFrame())
        renderer.deliverWorld(map: try #require(store.map), serial: 1, start: .current)
        renderer.showWorld(territory: store.territory, serial: 1)
        let image = try render(renderer, "world-scene", width: 800, height: 600)
        #expect(litCount(image) > 100_000, "the map fills the picture")
    }
}
