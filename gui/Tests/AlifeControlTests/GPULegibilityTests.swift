import AlifeCore
import Foundation
import Metal
import Testing
import simd

@testable import AlifeControl

/// The legibility floors of the World viewer's flat views, measured on the GPU (DESIGN §36.1, §36.6, §36.7; spec 7.2): the map has
/// its own luminance range, the outline of an organism has at least 3:1 against the pixel next to it, the size policy keeps single
/// organisms separable, and no pixel comes out as the black of a NaN.
///
/// Opt in with `ALIFE_GPU_TESTS=1 swift test --filter GPULegibilityTests`; skipped otherwise. The recorded planet's numbers are
/// `DiagProbe`'s (`ALIFE_V2_STREAM`). Images go under `$ALIFE_GPU_TESTS_OUT` or the temporary directory.
@MainActor
@Suite(
    "GPU legibility of the flat World",
    .enabled(
        if: ProcessInfo.processInfo.environment["ALIFE_GPU_TESTS"] == "1",
        "set ALIFE_GPU_TESTS=1 to compile the shaders and render on the GPU")
)
struct GPULegibilityTests {
    // MARK: Scene

    /// A world of the 3D fixture (L = 48, height 24, agent radius 1) seen from straight above, x to the right and y up.
    @MainActor
    private final class Rig {
        let renderer: Renderer
        let header: StreamHeader
        let base: PlayWeatherMap
        let territory: PlayTerritory
        var serial = 0

        init(renderer: Renderer, header: StreamHeader, base: PlayWeatherMap, territory: PlayTerritory) {
            (self.renderer, self.header, self.base, self.territory) = (renderer, header, base, territory)
        }

        var size: Float { Float(header.size) }

        func show(map: PlayWeatherMap) {
            serial += 1
            renderer.deliverWorld(map: map, serial: serial, start: .current)
        }

        func look(at target: SIMD3<Float>, distance: Float) -> Camera {
            var camera = Camera(target: target, distance: distance, yaw: .pi / 2, pitch: -.pi / 2)
            camera.scene = Camera.tiledScene(size: size, height: Float(header.height))
            return camera
        }
    }

    private func rig() throws -> Rig {
        let device = try #require(MTLCreateSystemDefaultDevice(), "no Metal device")
        let renderer = try Renderer(device: device)
        let store = try PlayFixtureStore.store("3d")
        let header = try #require(store.header)
        renderer.configureWorld(header: header)
        var display = WorldDisplay()
        (display.frame, display.agents, display.bonds) = (false, false, false)
        display.territory = .off
        renderer.worldDraw.display = display
        renderer.worldDraw.level = .region
        let rig = Rig(renderer: renderer, header: header, base: try #require(store.map),
                      territory: try #require(store.territory))
        rig.show(map: Self.map(rig.base, sea: 0, rain: 0, cloud: 0))
        return rig
    }

    /// A map of one weather everywhere: `sea` is the warm-pool byte, `rain` and `cloud` the depth bytes, no vorticity.
    private static func map(_ base: PlayWeatherMap, sea: UInt8, rain: UInt8, cloud: UInt8) -> PlayWeatherMap {
        var map = base
        let cells = map.n * map.n
        map.p = [UInt8](repeating: rain, count: cells)
        map.dc = [UInt8](repeating: cloud, count: cells)
        map.zeta = [Int8](repeating: 0, count: cells)
        map.ts = [UInt8](repeating: sea, count: cells)
        return map
    }

    private func outputDirectory() throws -> URL {
        let base = ProcessInfo.processInfo.environment["ALIFE_GPU_TESTS_OUT"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.temporaryDirectory.appendingPathComponent("alife-gpu-tests", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    private func render(_ rig: Rig, _ name: String?, width: Int = 512, height: Int = 512) throws -> OffscreenImage {
        let image = try rig.renderer.renderOffscreen(width: width, height: height)
        if let name { try PNG.write(image, to: try outputDirectory().appendingPathComponent("\(name).png")) }
        return image
    }

    private func rgb(_ p: (r: UInt8, g: UInt8, b: UInt8)) -> SIMD3<Float> { SIMD3(Float(p.r), Float(p.g), Float(p.b)) / 255 }

    private func luma(_ image: OffscreenImage, _ x: Int, _ y: Int) -> Float { OutlineContrast.luma(rgb(image.pixel(x, y))) }

    /// The mean luma of the whole picture.
    private func meanLuma(_ image: OffscreenImage) -> Float {
        var sum: Float = 0
        for y in 0..<image.height { for x in 0..<image.width { sum += luma(image, x, y) } }
        return sum / Float(image.width * image.height)
    }

    // MARK: The map's brightness (F1, F2, F7)

    /// The cold sea at the three opacities of the map (1.0 at 0.296 L, 0.9 at 0.15 L, 0.7 at 0.08 L), seen from straight above over the
    /// northern half: the GPU's pixel is the Swift mirror's (`MapPalette`), the contrast against the clear colour is at least 1.4, and
    /// the map-only frame at the organism level has a luma of at least 0.14 (arithmetic 0.160; v1's 0.25 opacity gave 0.085).
    @Test(arguments: [(0.296, 1.0), (0.15, 0.9), (0.08, 0.7)] as [(Double, Double)])
    func coldSeaKeepsItsOwnLuminanceRangeAtEveryOpacity(fraction: Double, opacity: Double) throws {
        let rig = try rig()
        rig.show(map: Self.map(rig.base, sea: 0, rain: 0, cloud: 0))
        let L = rig.size
        let distance = Float(fraction) * L
        #expect(abs(WorldLevels.mapOpacity(distance: distance, size: L) - Float(opacity)) < 1e-4)
        rig.renderer.camera = rig.look(at: SIMD3(L * 0.7, L / 4, 0), distance: distance)
        let image = try render(rig, "legibility-cold-sea-\(Int(fraction * 1000))")
        let sea = rgb(image.pixel(image.width / 2, image.height / 2))
        let predicted = MapPalette.eightBit(MapPalette.shown(
            MapPalette.color(sea: 0, zeta: 0, rain: 0, cloud: 0, northness: 1, hemisphereWeight: MapPalette.hemisphereWeight(distance: Float(fraction))),
            opacity: Float(opacity)))
        #expect(simd_reduce_max(simd_abs(sea - predicted)) * 255 <= 2.01, "GPU \(sea * 255) against the mirror \(predicted * 255)")
        let contrast = OutlineContrast.contrast(sea, MapPalette.eightBit(MapPalette.background))
        #expect(contrast >= 1.4, "cold sea against the clear colour at \(opacity): \(contrast)")
        if fraction == 0.08 { #expect(meanLuma(image) >= 0.14, "map-only frame at the organism level: \(meanLuma(image))") }
        // The raised sea is far from where v1 had it (1.05:1 at full opacity).
        #expect(contrast > 1.3 * 1.05)
    }

    /// Warm sea, full rain and full cloud, over the map at its full opacity: the Swift mirror's colour (the order of the shader's mixes).
    @Test(arguments: [("warm sea", 255, 0, 0), ("rain", 0, 255, 0), ("cloud", 0, 0, 255)] as [(String, Int, Int, Int)])
    func warmSeaRainAndCloudAreTheMirrorsColours(name: String, sea: Int, rain: Int, cloud: Int) throws {
        let rig = try rig()
        rig.show(map: Self.map(rig.base, sea: UInt8(sea), rain: UInt8(rain), cloud: UInt8(cloud)))
        let L = rig.size
        rig.renderer.camera = rig.look(at: SIMD3(L * 0.7, L / 4, 0), distance: 0.296 * L)
        let image = try render(rig, "legibility-map-\(name.replacingOccurrences(of: " ", with: "-"))")
        let got = rgb(image.pixel(image.width / 2, image.height / 2))
        let predicted = MapPalette.eightBit(MapPalette.color(
            sea: sea > 0 ? 1 : 0, zeta: 0, rain: rain > 0 ? 1 : 0, cloud: cloud > 0 ? 1 : 0, northness: 1, hemisphereWeight: 1))
        #expect(simd_reduce_max(simd_abs(got - predicted)) * 255 <= 3.01, "\(name): GPU \(got * 255) against the mirror \(predicted * 255)")
    }

    @Test func brightnessScalesTheMapAndNotTheOrganisms() throws {
        let rig = try rig()
        let L = rig.size
        var display = rig.renderer.worldDraw.display
        display.agents = true
        rig.renderer.worldDraw.display = display
        let frame = Self.frame([SIMD3(L * 0.7, L / 4, 0)], colors: [0x3070C0])
        rig.renderer.showWorld(frame: frame)
        rig.renderer.camera = rig.look(at: SIMD3(L * 0.7 + 3, L / 4 + 3, 0), distance: 0.296 * L)  // the organism is off to a corner of the view
        var seaY: [Float] = []
        var organism: [SIMD3<Float>] = []
        for exposure in WorldDisplay.exposurePresets {
            display.exposure = exposure
            rig.renderer.worldDraw.display = display
            let image = try render(rig, "legibility-exposure-\(Int(exposure * 100))")
            seaY.append(luma(image, image.width / 2, image.height / 2))
            let at = try pixel(of: frame.position(0), camera: rig.renderer.camera, image: image)
            organism.append(rgb(image.pixel(at.x, at.y)))
        }
        #expect(zip(seaY, seaY.dropFirst()).allSatisfy { $0 < $1 }, "the sea gets brighter with the exposure: \(seaY)")
        #expect(abs(seaY[2] / seaY[1] - 1.25) < 0.05, "x1.25 is 1.25 times the luma: \(seaY)")
        #expect(organism.allSatisfy { simd_reduce_max(simd_abs($0 - organism[0])) * 255 <= 1.01 }, "the organism is not scaled: \(organism)")
    }

    // MARK: Territory (F3)

    @Test func territoryIsBordersAndAWeakTintOfTheOwnersColour() throws {
        let rig = try rig()
        let L = rig.size
        rig.renderer.showWorld(territory: rig.territory, serial: 1)
        // Cells 0 to 10 of row 3 belong to slot 0, cells 11 to 21 to slot 1: the border between them is at 11 cells from the left.
        let cell = L / Float(rig.territory.nx)
        rig.renderer.camera = rig.look(at: SIMD3(11 * cell, 3.5 * cell, 0), distance: 0.15 * L)
        var display = rig.renderer.worldDraw.display
        var images: [TerritoryMode: OffscreenImage] = [:]
        for mode in TerritoryMode.allCases {
            display.territory = mode
            rig.renderer.worldDraw.display = display
            images[mode] = try render(rig, "legibility-territory-\(mode)")
        }
        let (off, borders, tint) = (try #require(images[.off]), try #require(images[.borders]), try #require(images[.tint]))
        let mid = off.height / 2
        let inside = off.width / 2 - 60  // well inside cell 8 or so: 60 px is a few cells at this distance
        // Borders only: the interior of a cell is the map; the border is a bright line.
        #expect(Self.sameColour(off.pixel(inside, mid), borders.pixel(inside, mid), by: 1), "no tint in the 'borders' state")
        var brightest = 0
        for x in (off.width / 2 - 4)...(off.width / 2 + 4) {
            brightest = max(brightest, Int(borders.pixel(x, mid).g) - Int(off.pixel(x, mid).g))
        }
        #expect(brightest >= 60, "a border: white at 0.55 over the sea (brightened by \(brightest))")
        // The tint is 10% of the owner's colour: slot 0 is E69F00, so red rises and blue falls a little.
        let (a, b) = (off.pixel(inside, mid), tint.pixel(inside, mid))
        let shift = (Int(b.r) - Int(a.r), Int(b.b) - Int(a.b))
        #expect(shift.0 > 5 && shift.0 < 40, "a weak tint: red \(a.r) -> \(b.r)")
        #expect(shift.1 < 0 && shift.1 > -20, "a weak tint: blue \(a.b) -> \(b.b)")
    }

    private static func sameColour(_ a: (r: UInt8, g: UInt8, b: UInt8), _ b: (r: UInt8, g: UInt8, b: UInt8), by tolerance: Int) -> Bool {
        abs(Int(a.r) - Int(b.r)) <= tolerance && abs(Int(a.g) - Int(b.g)) <= tolerance && abs(Int(a.b) - Int(b.b)) <= tolerance
    }

    // MARK: Frames of hand-placed organisms

    /// A frame of rows at `positions` with `colors` (`0xRRGGBB`), forward axis `pitch` radians above the horizon (pi/2: towards a
    /// camera above, so the nose is in the middle of the body and away from its edge).
    private static func frame(_ positions: [SIMD3<Float>], colors: [UInt32], pitch: Float = .pi / 2) -> PlayFrame {
        var frame = TestFrames.frame(positions)
        frame.rgb = colors
        frame.orient = (0..<positions.count).flatMap { _ in [Float16(0).bitPattern, Float16(pitch).bitPattern] }
        return frame
    }

    private func pixel(of world: SIMD3<Float>, camera: Camera, image: OffscreenImage) throws -> (x: Int, y: Int) {
        let ndc = try #require(camera.ndc(of: world, aspect: Float(image.width) / Float(image.height)))
        return (Int((ndc.x * 0.5 + 0.5) * Float(image.width)), Int((0.5 - ndc.y * 0.5) * Float(image.height)))
    }

    // MARK: The outline (F6, spec 1.10)

    /// Every colour the viewer draws organisms in: the 16 kin slots and the 3 reserved values, the archetype and intent tables, hurt
    /// red, and 200 `tag`-mode colours (random, the way a recorded frame has them: none is a palette entry).
    private static let colours: [UInt32] = {
        var all = WorldPalette.slots + [WorldPalette.unknown, WorldPalette.other, WorldPalette.none]
        all += WorldPalette.archetypes.dropFirst() + WorldPalette.intents.dropFirst() + [0xFF0000]
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        for _ in 0..<200 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            all.append(UInt32(truncatingIfNeeded: state >> 40) & 0xFFFFFF)
        }
        return all
    }()

    /// The pixel next to the ring of every organism of `colours` (a grid of 7 x 7 per picture), over four backdrops, at 7 px and
    /// 9 px, with the depth cue at four heights from the floor to the top, in four directions: the outline is dark or light grey, and its contrast
    /// with the pixel inside it is at least 3:1.
    @Test(arguments: [("cold sea", 0, 0, 0), ("warm sea", 255, 0, 0), ("rain", 0, 255, 0), ("cloud", 0, 0, 255)] as [(String, Int, Int, Int)])
    func theOutlineHasAtLeastThreeToOneAgainstThePixelNextToIt(name: String, sea: Int, rain: Int, cloud: Int) throws {
        let rig = try rig()
        rig.show(map: Self.map(rig.base, sea: UInt8(sea), rain: UInt8(rain), cloud: UInt8(cloud)))
        let L = rig.size, height = Float(rig.header.height)
        var display = rig.renderer.worldDraw.display
        display.sizeMode = .capped
        rig.renderer.worldDraw.pixelsPerPoint = 2
        let camera = rig.look(at: SIMD3(L / 2, L / 2, 0), distance: 40)
        rig.renderer.camera = camera
        let dark = SIMD3<Float>(repeating: 0.06), light = SIMD3<Float>(repeating: 0.92)
        var measured = 0, worst: Float = .infinity
        var worstAt = ""
        let side = 7
        let batches = stride(from: 0, to: Self.colours.count, by: side * side).map { Array(Self.colours[$0..<min($0 + side * side, Self.colours.count)]) }
        for radius in [Float(7), 9] {
            rig.renderer.worldDraw.coverRadius = radius
            for z in [Float(0), height / 3, 2 * height / 3, height] {  // (one depth per picture: the grid's rows never overlap)
                for (b, batch) in batches.enumerated() {
                    // A grid centred on the target, all at the height `z`.
                    let positions = batch.indices.map { k in
                        SIMD3<Float>(L / 2 + (Float(k % side) - 3) * 2.2, L / 2 + (Float(k / side) - 3) * 2.2, z)
                    }
                    let frame = Self.frame(positions, colors: batch)
                    display.agents = false
                    rig.renderer.worldDraw.display = display
                    rig.renderer.showWorld(frame: frame)
                    let backdrop = try render(rig, nil)
                    display.agents = true
                    rig.renderer.worldDraw.display = display
                    let image = try render(rig, b == 0 && z == 0 ? "legibility-outline-\(name.replacingOccurrences(of: " ", with: "-"))-\(Int(radius))px" : nil)
                    for (k, p) in positions.enumerated() {
                        let c = try pixel(of: p, camera: camera, image: image)
                        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                            // From the centre outwards while the pixel differs from the backdrop: the last one is the ring.
                            var (x, y) = c
                            var length = 0
                            func organism(_ x: Int, _ y: Int) -> Bool {
                                x >= 0 && y >= 0 && x < image.width && y < image.height && !Self.sameColour(image.pixel(x, y), backdrop.pixel(x, y), by: 0)
                            }
                            while organism(x + dx, y + dy), length < 40 {
                                x += dx
                                y += dy
                                length += 1
                            }
                            #expect(length >= Int(radius) - 2 && length <= Int(radius) + 1, "\(name) colour \(String(batch[k], radix: 16)): radius \(radius) measured \(length)")
                            let ring = rgb(image.pixel(x, y))
                            let isOutline = simd_reduce_max(simd_abs(ring - dark)) * 255 <= 1.01 || simd_reduce_max(simd_abs(ring - light)) * 255 <= 1.01
                            #expect(isOutline, "the ring is an outline colour: \(ring * 255) of \(name) colour \(String(batch[k], radix: 16))")
                            // The pixel next to the ring: the first one inwards that is not the ring's own colour (the ring is one pixel
                            // wide where the pixel grid and the silhouette agree, and sometimes two: the pixels step a little less than
                            // a world pixel off the axis).
                            var inwards = 1
                            while inwards < 4, Self.sameColour(image.pixel(x - inwards * dx, y - inwards * dy), image.pixel(x, y), by: 0) { inwards += 1 }
                            let inside = rgb(image.pixel(x - inwards * dx, y - inwards * dy))
                            let contrast = OutlineContrast.contrast(ring, inside)
                            measured += 1
                            if contrast < worst {
                                worst = contrast
                                worstAt = "\(name) \(String(batch[k], radix: 16)) r=\(radius) z=\(p.z) dir=(\(dx),\(dy)) ring \(ring * 255) inside \(inside * 255)"
                            }
                            #expect(contrast >= 3.0, "edge contrast \(contrast) of \(name) colour \(String(batch[k], radix: 16)) r=\(radius) z=\(p.z) dir=(\(dx),\(dy)): ring \(ring * 255) inside \(inside * 255)")
                        }
                    }
                }
            }
        }
        print("GPULegibilityTests: outline over \(name): \(measured) edges, worst contrast \(worst) (\(worstAt))")
        #expect(measured >= 4 * 2 * 4 * Self.colours.count - 100)
    }

    /// The outline is the silhouette of the organism (the body and the nose together), not of the nose inside the body: a nose that
    /// faces the camera shows no ring inside the body (the "eye" look of the A0 review, finding 5), a nose that sticks out past the
    /// body is outlined where it does. Seen from above at 31 px, the nose being at the centre (pitch 90 degrees), off to +x (pitch 0) or between.
    @Test(arguments: [Float.pi / 2, Float.pi / 4, 0])
    func theOutlineIsThatOfTheWholeOrganismNotOfItsNose(pitch: Float) throws {
        let rig = try rig()
        let L = rig.size
        var display = rig.renderer.worldDraw.display
        display.agents = true
        display.sizeMode = .physical  // 31 px, not the 9 px cap of the distance
        rig.renderer.worldDraw.display = display
        let centre = SIMD3<Float>(L / 2, L / 4, 0)  // (off the equator's line)
        let camera = rig.look(at: centre, distance: 40)
        rig.renderer.camera = camera
        rig.renderer.showWorld(frame: Self.frame([centre], colors: [0x3070C0], pitch: pitch))
        let image = try render(rig, "legibility-nose-ring-pitch\(Int(pitch * 180 / .pi))", width: 1024, height: 1024)
        let radius = OrganismSizePolicy.pixelsPerUnit(depth: 40, viewHeight: 1024, fovY: camera.fovY) * Float(rig.header.radius)
        #expect(radius > 25 && radius < 40, "the body's radius in pixels: \(radius)")
        let (cx, cy) = (image.width / 2, image.height / 2)
        func isOutline(_ x: Int, _ y: Int) -> Bool {
            let c = rgb(image.pixel(x, y))
            return simd_reduce_max(simd_abs(c - SIMD3(repeating: 0.06))) * 255 <= 1.01 || simd_reduce_max(simd_abs(c - SIMD3(repeating: 0.92))) * 255 <= 1.01
        }
        // Inside the silhouette of the union: no pixel of the outline's colours (the body is blue, shaded 0.3 to 1.0).
        var inside = 0
        for y in 0..<image.height {
            for x in 0..<image.width {
                let (dx, dy) = (Float(x - cx), Float(y - cy))
                // The union's interior: the body's disc to 2 px inside its edge, and the nose's disc to 2 px inside as well.
                let body = (dx * dx + dy * dy).squareRoot() < radius - 2
                let nx = 0.75 * radius * cos(pitch)  // (the nose's centre, projected)
                let nose = ((dx - nx) * (dx - nx) + dy * dy).squareRoot() < 0.5 * radius - 2
                if (body || nose) && isOutline(x, y) { inside += 1 }
            }
        }
        #expect(inside == 0, "\(inside) outline-coloured pixels inside the organism")
        // The silhouette itself is outlined: the body's edge on -x (the nose is not there), and for pitch 0 the nose's tip on +x.
        func ringAlong(sign: Int, from: Float, to: Float) -> Bool {
            stride(from: from, through: to, by: 0.25).contains { isOutline(cx + sign * Int($0.rounded()), cy) }
        }
        #expect(ringAlong(sign: -1, from: radius - 1.5, to: radius + 0.5), "the body's edge has its ring")
        if pitch == 0 { #expect(ringAlong(sign: 1, from: 1.25 * radius - 1.5, to: 1.25 * radius + 0.5), "the nose's tip has its ring") }
    }

    // MARK: Single organisms (F6, spec 1.9)

    /// A dense synthetic frame in the fixture's world, seen from above at the region level: the organisms at the size policy keep at
    /// least 80% of their disc visible for at least 80% of them, where the same rows at their physical size keep 50% for far fewer;
    /// the occlusion raster agrees with what the GPU draws (the union of the discs is the picture's organism pixels, within
    /// the outline and the noses).
    @Test func capsKeepSingleOrganismsSeparableOnTheGPU() throws {
        let rig = try rig()
        let L = rig.size, height = Float(rig.header.height)
        let side = 1024
        let rows = 1500
        var state: UInt64 = 12345
        func unit() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(state >> 40) / Float(1 << 24)
        }
        let positions = (0..<rows).map { _ in SIMD3<Float>(unit() * L, unit() * L, unit() * height) }
        let colours = (0..<rows).map { Self.colours[$0 % 19] }
        let frame = Self.frame(positions, colors: colours)
        let target = SIMD3<Float>(L / 2, L / 2, 0)
        let camera = rig.look(at: target, distance: 0.9 * L)
        rig.renderer.camera = camera
        rig.renderer.showWorld(frame: frame)
        var display = rig.renderer.worldDraw.display
        display.map = true
        display.bonds = false
        let aspect: Float = 1
        let area = Float(side * side)
        let inView = OrganismSizePolicy.rowsInView(frame: frame, camera: camera, aspect: aspect, size: L)
        #expect(inView > 500, "rows in view: \(inView)")

        func discs(limits: (cap: Float, minimum: Float)) -> [OrganismDisc] {
            let tanHalf = tan(camera.fovY / 2)
            return positions.compactMap { p in
                let depth = simd_dot(p - camera.position, camera.forward)
                guard let ndc = camera.ndc(of: p, aspect: aspect), abs(ndc.x) <= 1, abs(ndc.y) <= 1, depth > 0 else { return nil }
                let physical = Float(rig.header.radius) * 0.5 * Float(side) / (depth * tanHalf)
                return OrganismDisc(x: (ndc.x * 0.5 + 0.5) * Float(side), y: (0.5 - ndc.y * 0.5) * Float(side),
                                    r: OrganismSizePolicy.radius(physical: physical, cap: limits.cap, minimum: limits.minimum), depth: depth)
            }
        }
        // Capped: the model's numbers (the cover radius of the rows in view, the absolute cap at this distance).
        let cover = OrganismSizePolicy.coverRadius(area: area, rows: inView)
        let capped = OrganismSizePolicy.limits(
            mode: .capped, cover: cover, distance: camera.distance, size: L, agentRadius: Float(rig.header.radius),
            viewHeight: Float(side), fovY: camera.fovY, pixelsPerPoint: 2)
        let cappedOcc = OcclusionRaster.measure(discs(limits: capped), width: side, height: side)
        let physicalOcc = OcclusionRaster.measure(discs(limits: OrganismSizePolicy.limits(
            mode: .physical, cover: cover, distance: camera.distance, size: L, agentRadius: Float(rig.header.radius),
            viewHeight: Float(side), fovY: camera.fovY, pixelsPerPoint: 2)), width: side, height: side)
        print("GPULegibilityTests: \(inView) rows in view, cover radius \(cover), cap \(capped.cap): at least half visible \(cappedOcc.halfVisible) (physical \(physicalOcc.halfVisible)), coverage \(cappedOcc.coverage) (physical \(physicalOcc.coverage))")
        #expect(cappedOcc.halfVisible >= 0.80, "capped: \(cappedOcc.halfVisible) of \(cappedOcc.counted)")
        #expect(physicalOcc.halfVisible < 0.60, "physical: \(physicalOcc.halfVisible)")
        #expect(cappedOcc.coverage <= 0.35 + 0.02, "the organisms cover at most 35%: \(cappedOcc.coverage)")

        // The GPU draws what the raster says: the pixels that differ from the same picture without organisms.
        for (mode, limits, expected) in [(OrganismSizeMode.capped, capped, cappedOcc.coverage), (.physical, (cap: Float.infinity, minimum: 1.2), physicalOcc.coverage)] {
            display.sizeMode = mode
            display.agents = true
            rig.renderer.worldDraw.display = display
            rig.renderer.worldDraw.coverRadius = cover
            rig.renderer.worldDraw.pixelsPerPoint = 2
            let withOrganisms = try render(rig, "legibility-separable-\(mode)", width: side, height: side)
            display.agents = false
            rig.renderer.worldDraw.display = display
            let without = try render(rig, nil, width: side, height: side)
            var differing = 0
            for y in 0..<side { for x in 0..<side where !Self.sameColour(withOrganisms.pixel(x, y), without.pixel(x, y), by: 0) { differing += 1 } }
            let drawn = Double(differing) / Double(side * side)
            print("GPULegibilityTests: \(mode) covers \(drawn) of the picture on the GPU, \(expected) by the raster (\(limits.cap))")
            #expect(abs(drawn - expected) < 0.04, "\(mode): GPU \(drawn) against the raster \(expected)")
        }
    }

    // MARK: No NaN (spec 7.2)

    /// Degenerate inputs must not turn into the black of a NaN: organisms at the eye's height and on the camera's axis, a radius cap of
    /// nothing and of everything, the nearest and farthest zoom, in both size modes and at both ends of the exposure. Black is
    /// never a legitimate pixel here (the darkest is the clear colour, then the dark outline, 15 of 255), and every frame has the map.
    @Test(arguments: [OrganismSizeMode.capped, .physical])
    func noPixelIsTheBlackOfANaN(mode: OrganismSizeMode) throws {
        let rig = try rig()
        let L = rig.size, height = Float(rig.header.height)
        var display = rig.renderer.worldDraw.display
        (display.agents, display.bonds, display.frame) = (true, true, false)
        display.sizeMode = mode
        display.territory = .tint
        rig.renderer.showWorld(territory: rig.territory, serial: 1)
        // Rows on a ring about the target, one just under the eye, one at the camera's height, one 1e-4 above the floor.
        let centre = SIMD3<Float>(L / 2, L / 2, 0)
        var positions: [SIMD3<Float>] = (0..<24).map { k in
            let a = Float(k) * .pi / 12
            return centre + SIMD3(cos(a) * 2, sin(a) * 2, Float(k % 4) * height / 3)
        }
        positions += [centre + SIMD3(0, 0, 1e-4), centre + SIMD3(0, 0, height), centre + SIMD3(0.001, 0, 5)]
        var frame = Self.frame(positions, colors: positions.indices.map { Self.colours[$0 % 19] }, pitch: 0.3)
        frame.bonds = [0, 1, 2, 3, 4, 5]
        rig.renderer.showWorld(frame: frame)
        var pictures = 0
        for distance in [Float(0.25), 1, 4, 12, 24, 48, 120, 480] {
            for cover in [Float(0), 1e-9, 2.5, 9, 1e9, .infinity] {
                for exposure in [Float(0.5), 1, 2] {
                    display.exposure = exposure
                    rig.renderer.worldDraw.display = display
                    rig.renderer.worldDraw.coverRadius = cover
                    rig.renderer.camera = rig.look(at: centre, distance: distance)
                    let image = try render(rig, nil, width: 96, height: 96)
                    var black = 0
                    for y in 0..<image.height { for x in 0..<image.width where image.pixel(x, y) == (0, 0, 0) { black += 1 } }
                    #expect(black == 0, "\(mode) d=\(distance) cover=\(cover) exposure=\(exposure): \(black) black pixels")
                    pictures += 1
                }
            }
        }
        #expect(pictures == 8 * 6 * 3)
    }

    // MARK: Flat views of the organisms

    /// The instances have no level at which they appear: the opacity is a ramp of the drawn radius (1.5 px to 3.5 px), so a row too small to
    /// see is not drawn and the picture has no step in the number of lit pixels as the camera moves away.
    @Test func organismsFadeInAsTheirDrawnRadiusGrows() throws {
        let rig = try rig()
        let L = rig.size
        var display = rig.renderer.worldDraw.display
        (display.agents, display.map) = (true, false)
        display.sizeMode = .physical
        rig.renderer.worldDraw.display = display
        rig.renderer.worldDraw.level = .planet
        let frame = Self.frame([SIMD3(L / 2, L / 2, 12)], colors: [0xE69F00])
        rig.renderer.showWorld(frame: frame)
        var lit: [(distance: Float, pixels: Int)] = []
        for distance in stride(from: Float(8), through: 400, by: 8) {
            rig.renderer.camera = rig.look(at: SIMD3(L / 2, L / 2, 12), distance: distance)
            let image = try render(rig, nil, width: 256, height: 256)
            var count = 0
            for y in 0..<image.height { for x in 0..<image.width where !Self.sameColour(image.pixel(x, y), (9, 10, 14), by: 2) { count += 1 } }
            lit.append((distance, count))
        }
        // A body of radius 1 is 1.5 px in a 256 px picture at a distance of 128 / (1.5 tan 25 deg) = 183: from there it is not drawn.
        #expect(lit.first!.pixels > 100, "drawn near: \(lit.first!)")
        #expect(lit.last!.pixels == 0, "and gone below 1.5 px: \(lit.last!)")
        #expect(zip(lit, lit.dropFirst()).allSatisfy { $1.pixels <= $0.pixels + 3 }, "never more lit pixels farther away: \(lit.map(\.pixels))")
    }
}
