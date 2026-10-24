import AlifeCore
import AppKit
import Foundation
import Metal
import SwiftUI
import Testing
import simd

@testable import AlifeControl

/// The orientation aids and the height colour, drawn by the renderer of the **hosted** `WorldView` (DESIGN §36.5; spec 7.2). Every
/// test goes through the whole `WorldView` in an `NSHostingView`, as the app does, and takes the renderer from the view that owns it
/// (the lesson of I61). Opt in with `ALIFE_GPU_TESTS=1 swift test --filter GPUHUDTests`; images go under `$ALIFE_GPU_TESTS_OUT`.
@MainActor
@Suite(
    "GPU minimap and height colour of the World (hosted)",
    .serialized,
    .enabled(
        if: ProcessInfo.processInfo.environment["ALIFE_GPU_TESTS"] == "1",
        "set ALIFE_GPU_TESTS=1 to compile the shaders and render on the GPU")
)
struct GPUHUDTests {
    // MARK: Helpers

    private func isBackground(_ p: (r: UInt8, g: UInt8, b: UInt8)) -> Bool {
        abs(Int(p.r) - 9) <= 2 && abs(Int(p.g) - 10) <= 2 && abs(Int(p.b) - 14) <= 2
    }

    /// A map of the fixture's size with rain where `rain(cx, cy)` says so, and nothing else.
    private func rainMap(from base: PlayWeatherMap, _ rain: (Int, Int) -> Bool) -> PlayWeatherMap {
        var map = base
        let n = map.n
        map.p = [UInt8](repeating: 0, count: n * n)
        map.dc = [UInt8](repeating: 0, count: n * n)
        map.zeta = [Int8](repeating: 0, count: n * n)
        map.ts = [UInt8](repeating: 0, count: n * n)
        for cy in 0..<n { for cx in 0..<n where rain(cx, cy) { map.p[cy * n + cx] = 255 } }
        return map
    }

    private let seaBlue = 125
    private let rainBlue = 150

    /// The pixel of an image drawn at one pixel per point at a point of the free rectangle.
    private func pixel(_ image: OffscreenImage, _ p: CGPoint) -> (r: UInt8, g: UInt8, b: UInt8) {
        image.pixel(min(max(Int(p.x), 0), image.width - 1), min(max(Int(p.y), 0), image.height - 1))
    }

    // MARK: The minimap

    @Test func theMinimapIsTheFlatMapOfTheSameTexturesNorthUpInTheRectangleTheModelHitTests() async throws {
        let world = try await HostedWorld()
        defer { world.finish() }
        let model = world.model
        model.display.minimap = true
        let base = try #require(world.play.store.map)
        let n = base.n
        let block = max(n / 8, 2)
        try world.show(map: rainMap(from: base) { cx, cy in (cx / block + cy / block) % 2 == 0 })
        let box = try #require(model.inset)
        let image = try world.render("minimap-checkerboard")
        let L = model.worldSize
        var compared = 0
        for by in 0..<(n / block) {
            for bx in 0..<(n / block) {
                // The middle of the block, in world units; its place in the box is the model's.
                let x = (Float(bx * block) + Float(block) / 2) / Float(n) * L, y = (Float(by * block) + Float(block) / 2) / Float(n) * L
                let at = box.point(forTorus: x, y: y)
                let blue = Int(pixel(image, at).b)
                guard blue > rainBlue || blue < seaBlue else { continue }
                #expect((blue > rainBlue) == ((bx + by) % 2 == 0), "block (\(bx), \(by)) at (\(at.x), \(at.y)): blue \(blue)")
                compared += 1
            }
        }
        #expect(compared >= (n / block) * (n / block) - 4, "every block was compared: \(compared)")
    }

    @Test func theMinimapHasEastToTheRightAndNorthUp() async throws {
        let world = try await HostedWorld()
        defer { world.finish() }
        let model = world.model
        model.display.minimap = true
        let base = try #require(world.play.store.map)
        let n = base.n
        let box = try #require(model.hud.minimap)
        // Rain in the first eighth of the columns (x from 0 to L/8), then in the first eighth of the rows (y from 0 to L/8, which is
        // north of the equator, just above the middle line of the box).
        try world.show(map: rainMap(from: base) { cx, _ in cx < n / 8 })
        var image = try world.render("minimap-west-strip")
        #expect(Int(pixel(image, CGPoint(x: box.minX + box.width * 0.06, y: box.midY + box.height * 0.3)).b) > rainBlue, "x near 0 is at the left")
        #expect(Int(pixel(image, CGPoint(x: box.maxX - box.width * 0.06, y: box.midY + box.height * 0.3)).b) < seaBlue, "and x near L is not at the right")
        #expect(Int(pixel(image, CGPoint(x: box.minX + box.width * 0.5, y: box.midY)).b) < seaBlue)
        try world.show(map: rainMap(from: base) { _, cy in cy < n / 8 })
        image = try world.render("minimap-north-strip")
        #expect(Int(pixel(image, CGPoint(x: box.midX, y: box.midY - box.height * 0.06)).b) > rainBlue, "y just above 0 is above the equator")
        #expect(Int(pixel(image, CGPoint(x: box.midX, y: box.midY + box.height * 0.06)).b) < seaBlue, "and not below it")
        // The rows just under the wrap, at the top edge (ybar near L/2), are the far end of the map: rain only in the last eighth of the rows.
        try world.show(map: rainMap(from: base) { _, cy in cy >= n / 2 - n / 16 && cy < n / 2 })
        image = try world.render("minimap-seam-strip")
        #expect(Int(pixel(image, CGPoint(x: box.midX, y: box.minY + box.height * 0.03)).b) > rainBlue, "the rows just before the wrap are at the top edge")
        #expect(Int(pixel(image, CGPoint(x: box.midX, y: box.maxY - box.height * 0.03)).b) < seaBlue)
    }

    @Test func theMinimapChangesNothingOutsideItsRectangleAndTheMinimapOffDrawsNone() async throws {
        let world = try await HostedWorld()
        defer { world.finish() }
        let model = world.model
        model.display.minimap = true
        let base = try #require(world.play.store.map)
        let n = base.n
        try world.show(map: rainMap(from: base) { cx, cy in (cx / 4 + cy / 4) % 2 == 0 })
        let box = try #require(model.hud.minimap)
        let on = try world.render("minimap-on")
        model.display.minimap = false
        let off = try world.render("minimap-off")
        #expect(model.hud.minimap == nil)
        var inside = 0, outside = 0
        for y in 0..<on.height {
            for x in 0..<on.width {
                let (a, b) = (on.pixel(x, y), off.pixel(x, y))
                guard a.r != b.r || a.g != b.g || a.b != b.b else { continue }
                if box.insetBy(dx: -1, dy: -1).contains(CGPoint(x: x, y: y)) { inside += 1 } else { outside += 1 }
            }
        }
        #expect(outside == 0, "\(outside) pixels outside the minimap's rectangle changed")
        #expect(inside > Int(box.width * box.height) / 3, "the minimap drew its map: \(inside) pixels differ")
        _ = n
    }

    @Test func theMinimapShowsTheOrganismsAsAHazeOnlyWhileTheyAreDrawn() async throws {
        let world = try await HostedWorld()
        defer { world.finish() }
        let model = world.model
        model.display.minimap = true
        let base = try #require(world.play.store.map)
        try world.show(map: rainMap(from: base) { _, _ in false })
        var display = model.display
        display.agents = true
        model.display = display
        let L = model.worldSize
        // A crowd of 12,000 in the north-west quarter of the world (x and y under L/4: just above the equator, at the left), none elsewhere.
        var generator = SystemRandomNumberGenerator()
        let crowd = TestFrames.frame((0..<12_000).map { _ in
            SIMD3<Float>(Float.random(in: 0..<(L / 4), using: &generator), Float.random(in: 0..<(L / 4), using: &generator), 3)
        })
        let renderer = try world.renderer()
        renderer.showWorld(frame: crowd)
        model.editPose { $0.x = L * 0.8; $0.y = L * 0.9; $0.distance = 1e6 }  // (the main view looks away from the crowd)
        let box = try #require(model.hud.minimap)
        let crowded = try world.render("minimap-haze")
        func mean(_ image: OffscreenImage, _ rect: CGRect) -> Double {
            var sum = 0.0, count = 0.0
            for y in Int(rect.minY)..<Int(rect.maxY) { for x in Int(rect.minX)..<Int(rect.maxX) { sum += Double(image.pixel(x, y).b); count += 1 } }
            return sum / count
        }
        let (w, h) = (box.width, box.height)
        let crowdBox = CGRect(x: box.minX + 0.01 * w, y: box.midY - 0.24 * h, width: 0.23 * w, height: 0.23 * h)  // x in (0, L/4), ybar in (0, L/4)
        let elsewhere = CGRect(x: box.midX + 0.1 * w, y: box.midY + 0.1 * h, width: 0.3 * w, height: 0.3 * h)
        display.agents = false
        model.display = display
        let empty = try world.render("minimap-no-haze")
        #expect(mean(crowded, crowdBox) > mean(empty, crowdBox) + 8, "the crowd is a lighter haze: \(mean(crowded, crowdBox)) against \(mean(empty, crowdBox)) without")
        #expect(abs(mean(crowded, elsewhere) - mean(empty, elsewhere)) < 0.5, "and nothing is added where there is no one")
    }

    // MARK: The height colour

    /// Three organisms along the middle of the view at heights `zs` (and one of them hurt, if asked), seen from straight above at their
    /// physical size, and the pixel at the middle of each.
    private func organismPixels(_ world: HostedWorld, zs: [Float], rgb: [UInt32], hurt: Int? = nil, name: String) throws -> [(r: UInt8, g: UInt8, b: UInt8)] {
        let model = world.model
        let renderer = try world.renderer()
        var display = model.display
        (display.agents, display.bonds, display.frame, display.map, display.territory) = (true, false, false, false, .off)
        display.sizeMode = .physical
        model.display = display
        var frame = try #require(world.play.store.frame)
        let centre = SIMD2<Float>(24, 24)
        for i in 0..<frame.n { (frame.pos[3 * i], frame.pos[3 * i + 1], frame.pos[3 * i + 2]) = (1, 1, 1) }  // (the rest, far from the view)
        for (i, z) in zs.enumerated() {
            (frame.pos[3 * i], frame.pos[3 * i + 1], frame.pos[3 * i + 2]) = (centre.x + 3 * Float(i - 1), centre.y, z)
            frame.rgb[i] = rgb[i]
        }
        if let hurt { frame.rgb[hurt] = 0xFF0000 }
        renderer.showWorld(frame: frame)
        model.editPose { $0.x = centre.x; $0.y = centre.y; $0.z = 8; $0.heading = 0; $0.tilt = 0; $0.distance = 30 }  // (the eye at 38, over all of them)
        let image = try world.render(name)
        let projector = model.projector(size: model.freeRect.size)
        return try zs.indices.map { i in
            let at = try #require(projector.screen(frame.position(i)))
            return pixel(image, at)
        }
    }

    @Test func anOrganismIsColouredByItsAltitudeOnTheViridisRampAndNothingElse() async throws {
        let world = try await HostedWorld()
        defer { world.finish() }
        let model = world.model
        let height = Float(try #require(world.play.store.header).height)
        let zs: [Float] = [1, height / 2, height - 1]
        model.setColourMode(HeightColour.modeName)
        let first = try organismPixels(world, zs: zs, rgb: [0x20A0E0, 0xE04020, 0x30E040], name: "height-colour-a")
        // (a different engine colour for each, the same heights)
        let second = try organismPixels(world, zs: zs, rgb: [0xE04020, 0x30E040, 0x20A0E0], name: "height-colour-b")
        for (i, z) in zs.enumerated() {
            let expected = HeightColour.color(HeightColour.fraction(z: z, height: height)) * 255
            for pixel in [first[i], second[i]] {
                #expect(abs(Float(pixel.r) - expected.x) <= 3 && abs(Float(pixel.g) - expected.y) <= 3 && abs(Float(pixel.b) - expected.z) <= 3,
                        "z = \(z): (\(pixel.r), \(pixel.g), \(pixel.b)) against \(expected)")
            }
            #expect(first[i] == second[i], "the stream's colour has no say")
        }
        // The ramp rises: each is brighter than the one below.
        let luma = first.map { 0.2126 * Double($0.r) + 0.7152 * Double($0.g) + 0.0722 * Double($0.b) }
        #expect(luma[0] < luma[1] && luma[1] < luma[2])
    }

    @Test func theHurtRedStaysRedAndTheEnginesColoursReturnWhenTheHeightIsOff() async throws {
        let world = try await HostedWorld()
        defer { world.finish() }
        let model = world.model
        let height = Float(try #require(world.play.store.header).height)
        let zs: [Float] = [height / 2, height / 2, height / 2]
        model.setColourMode(HeightColour.modeName)
        let hurt = try organismPixels(world, zs: zs, rgb: [0x20A0E0, 0x20A0E0, 0x20A0E0], hurt: 1, name: "height-colour-hurt")
        #expect(hurt[1].r > 240 && hurt[1].g < 15 && hurt[1].b < 15, "pure red in the height mode too: \(hurt[1])")
        #expect(hurt[0].r < 100, "its neighbours are the ramp's teal: \(hurt[0])")
        // The height off: the stream's colour with the depth cue of the engine's modes.
        model.display.heightColour = false
        let plain = try organismPixels(world, zs: [2, height - 2, height / 2], rgb: [0x20A0E0, 0x20A0E0, 0xE04020], name: "engine-colour")
        let cue = { (z: Float) in OutlineContrast.depthCueBase + OutlineContrast.depthCueSlope * z / height }
        for (i, z) in [Float(2), height - 2, height / 2].enumerated() {
            let rgb: [SIMD3<Float>] = [SIMD3(0x20, 0xA0, 0xE0), SIMD3(0x20, 0xA0, 0xE0), SIMD3(0xE0, 0x40, 0x20)]
            let expected = rgb[i] * cue(z)
            let p = plain[i]
            #expect(abs(Float(p.r) - expected.x) <= 4 && abs(Float(p.g) - expected.y) <= 4 && abs(Float(p.b) - expected.z) <= 4,
                    "\(i): (\(p.r), \(p.g), \(p.b)) against \(expected)")
        }
    }

    // MARK: The planet

    // MARK: Looking at the planet

    /// The recorded planet (`ALIFE_V2_STREAM`, a `.bin` of `PlayStream` records: the 64k3d world, 44,162 organisms in a world of 320),
    /// or a committed fixture. Renders home, a region at a storm and an organism, with the minimap, compass, scale bar and height colour.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ALIFE_V2_STREAM"] != nil, "set ALIFE_V2_STREAM=<a recorded planet stream>"))
    func rendersThePlanetAtHomeOnARegionAndOnAnOrganism() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["ALIFE_V2_STREAM"])
        let world = try await HostedWorld(stream: URL(fileURLWithPath: path), size: CGSize(width: 1500, height: 900), dockCollapsed: true,
                                          timeout: .seconds(90))
        defer { world.finish() }
        let model = world.model
        model.reduceMotion = { true }
        model.display.minimap = true
        let clock = Clock()
        model.now = { clock.t }
        func settle() {
            for _ in 0..<900 {
                clock.t += 1.0 / 60
                if !model.advance(), !model.isAnimating { break }
            }
            clock.t += 1
            world.settle()
        }
        model.drawableResized(CGSize(width: 2 * model.freeRect.width, height: 2 * model.freeRect.height))
        let header = try #require(world.play.store.header)
        #expect(header.dims == 3 && world.play.store.frame?.n ?? 0 > 1000)
        #expect(!model.display.heightColour, "a world opens in its own colours")
        model.goHome()
        settle()
        try world.composite("home-engine-colour")
        model.setColourMode(HeightColour.modeName)
        settle()
        let picture = try world.composite("home-height")
        // The picture is the window's: the legend's bar runs the ramp from the bottom (0) to the top (H), the minimap is in its corner.
        let image = try #require(world.lastComposite)
        let bar = WorldHUDView.legendBar(in: model.freeRect.size)
        let low = HostedWorld.rgb(of: image, Int(bar.midX * 2), Int((bar.maxY - 3) * 2)), high = HostedWorld.rgb(of: image, Int(bar.midX * 2), Int((bar.minY + 3) * 2))
        let expectedLow = HeightColour.color(0.03) * 255, expectedHigh = HeightColour.color(0.97) * 255
        #expect(simd_distance(low, expectedLow) < 16 && simd_distance(high, expectedHigh) < 16, "legend \(low) \(high) against \(expectedLow) \(expectedHigh)")
        let box = try #require(model.hud.minimap)
        #expect(!isBackground(picture.pixel(Int(box.midX * 2), Int(box.midY * 2))) && !isBackground(picture.pixel(Int(box.minX * 2) + 6, Int(box.minY * 2) + 6)))
        // A region at a storm.
        let storm = try #require(world.play.store.storms?.storms.storms.first)
        model.flyTo(SIMD3(Float(storm.x), Float(storm.y), 0), distance: model.flyDistance(.region))
        settle()
        try world.composite("region-height")
        model.setColourMode(try #require(header.colorModes.first))
        settle()
        try world.composite("region-engine-colour")
        model.setColourMode(HeightColour.modeName)
        // An organism, seen from a little to the side so that the altitude shows.
        let frame = try #require(world.play.store.frame)
        let i = frame.n / 3
        model.selectOrganism(frame.uid[i], follow: false, camera: false, fly: false)
        let p = frame.position(i)
        model.editPose { $0.x = p.x; $0.y = p.y; $0.z = p.z; $0.distance = 0.08 * model.worldSize; $0.tilt = 0.7; $0.heading = 0.4 }
        settle()
        try world.composite("organism-height")
    }

    private final class Clock { var t: Double = 1000 }
}
