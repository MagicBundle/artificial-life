import AlifeCore
import AppKit
import Foundation
import Metal
import SwiftUI
import Testing
import simd

@testable import AlifeControl

/// The one-world patch of the weather map and its copies across the wrap in y, drawn by the renderer of the **hosted** `WorldView` (DESIGN
/// §36.5; spec 7.2 GPU tests 3 and 4). Every test here goes through the whole `WorldView` in an `NSHostingView`, as the app does, and takes
/// the renderer from the view that owns it: the lesson of I61, where the window's renderer was freed at once and no test noticed because the
/// tests held theirs in a local.
///
/// Opt in with `ALIFE_GPU_TESTS=1 swift test --filter GPUPatchTests`; skipped otherwise. Images go under `$ALIFE_GPU_TESTS_OUT` or the
/// temporary directory.
@MainActor
@Suite(
    "GPU patch of the flat World (hosted)",
    .serialized,
    .enabled(
        if: ProcessInfo.processInfo.environment["ALIFE_GPU_TESTS"] == "1",
        "set ALIFE_GPU_TESTS=1 to compile the shaders and render on the GPU")
)
struct GPUPatchTests {
    /// The whole `WorldView` hosted in an invisible window far off every screen, over a played fixture.
    @MainActor
    private final class Hosted {
        let play = PlayController()
        let host: NSHostingView<WorldView>
        let window: NSWindow
        let scratch: Scratch
        var serial = 0

        init(kind: String = "3d") async throws {
            scratch = try Scratch()
            let script = try scratch.script("alife-live", "cat \"$1\"\nwhile IFS= read -r line; do :; done")
            play.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url,
                       arguments: [Repo.playFixtureURL(kind).path])
            let size = CGSize(width: 1100, height: 700)
            host = NSHostingView(rootView: WorldView(play: play))
            window = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless], backing: .buffered, defer: false)
            window.contentView = host
            host.frame = NSRect(origin: .zero, size: size)
            window.alphaValue = 0
            window.ignoresMouseEvents = true
            window.setFrameOrigin(NSPoint(x: -20000, y: -20000))
            window.orderFrontRegardless()
            #expect(await eventually { settle(); return play.store.header != nil && play.store.frame != nil && play.store.map != nil && metal != nil })
            settle()
        }

        func finish() {
            window.orderOut(nil)
            play.shutdown()
            scratch.cleanup()
        }

        func settle() {
            for _ in 0..<8 {
                host.layoutSubtreeIfNeeded()
                window.displayIfNeeded()
                RunLoop.current.run(until: Date().addingTimeInterval(0.02))
            }
        }

        var metal: WorldMTKView? { find(host) }

        private func find(_ view: NSView) -> WorldMTKView? {
            if let hit = view as? WorldMTKView { return hit }
            for sub in view.subviews { if let hit = find(sub) { return hit } }
            return nil
        }

        var model: WorldModel { metal!.model! }

        /// The renderer of the hosted view: the one the app draws with.
        func renderer() throws -> Renderer {
            #expect(metal?.delegate is Renderer, "the Metal view has no renderer")
            return try #require(model.renderer)
        }

        /// Only the weather map: no organisms, bonds, territory or frame; and `map` as the one delivered.
        func show(map: PlayWeatherMap) throws {
            var display = model.display
            (display.agents, display.bonds, display.frame, display.territory) = (false, false, false, .off)
            model.display = display
            serial += 1000
            let renderer = try renderer()
            renderer.deliverWorld(map: map, serial: serial, start: .current)
            renderer.worldDraw.mapBlend = 1
        }

        /// The picture at the size of the free rectangle (the aspect the model fitted the view to).
        func render(_ name: String) throws -> OffscreenImage {
            let free = model.freeRect
            let image = try renderer().renderOffscreen(width: Int(free.width), height: Int(free.height))
            let base = ProcessInfo.processInfo.environment["ALIFE_GPU_TESTS_OUT"].map { URL(fileURLWithPath: $0) }
                ?? FileManager.default.temporaryDirectory.appendingPathComponent("alife-gpu-tests", isDirectory: true)
            try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
            try PNG.write(image, to: base.appendingPathComponent("\(name).png"))
            return image
        }
    }

    private func isBackground(_ p: (r: UInt8, g: UInt8, b: UInt8)) -> Bool {
        abs(Int(p.r) - 9) <= 2 && abs(Int(p.g) - 10) <= 2 && abs(Int(p.b) - 14) <= 2
    }

    private func backgroundCount(_ image: OffscreenImage) -> Int {
        var count = 0
        for y in 0..<image.height { for x in 0..<image.width where isBackground(image.pixel(x, y)) { count += 1 } }
        return count
    }

    /// A map of the fixture's size with rain in the blocks of a checkerboard, `block` cells a side, and nothing else.
    private func checkerboard(from base: PlayWeatherMap, block: Int) -> PlayWeatherMap {
        var map = base
        let n = map.n
        map.p = [UInt8](repeating: 0, count: n * n)
        map.dc = [UInt8](repeating: 0, count: n * n)
        map.zeta = [Int8](repeating: 0, count: n * n)
        map.ts = [UInt8](repeating: 0, count: n * n)
        for cy in 0..<n { for cx in 0..<n where (cx / block + cy / block) % 2 == 0 { map.p[cy * n + cx] = 255 } }
        return map
    }

    private let seaBlue = 125
    private let rainBlue = 150

    // MARK: No edge

    @Test func theMapFillsTheHostedViewAtTheFarthestDistanceAtEveryHeadingAndNearTheSeams() async throws {
        let world = try await Hosted()
        defer { world.finish() }
        let model = world.model
        let map = try #require(world.play.store.map)
        try world.show(map: checkerboard(from: map, block: 16))
        let L = model.worldSize
        // (x, y, heading): the middle, across the wrap in y, across x = 0, a corner of both, and turned every which way.
        let views: [(Float, Float, Float)] = [(L / 2, 0, 0), (L / 2, L / 2, 0), (0.2, 10, 0.4), (L - 0.2, L / 2 + 0.2, .pi / 4), (3, L / 2 - 0.3, 1.2),
                                              (40, 3, 2.9), (L / 2, L - 1, -1.0), (1, 1, .pi / 2)]
        for (x, y, heading) in views {
            model.editPose { $0.x = x; $0.y = y; $0.heading = heading; $0.tilt = 0; $0.distance = 1e6 }  // (clamped to the farthest)
            #expect(model.pose.distance == model.distanceLimit, "the farthest the view may be at this heading")
            let image = try world.render("patch-far-\(Int(x))-\(Int(y))-\(Int(heading * 10))")
            #expect(backgroundCount(image) == 0, "an edge of the patch shows at (\(x), \(y)) heading \(heading): \(backgroundCount(image)) pixels")
        }
    }

    @Test func theMapFillsTheHostedViewTiltedAsFarAsTheViewMayBeTiltedAndNeverShowsTheHorizon() async throws {
        let world = try await Hosted()
        defer { world.finish() }
        let model = world.model
        let map = try #require(world.play.store.map)
        try world.show(map: checkerboard(from: map, block: 16))
        let L = model.worldSize
        for (y, heading, fraction) in [(Float(0), Float(0), Float(0.5)), (L / 2, 0.7, 0.3), (L / 2 + 1, -2.0, 0.6), (5, 1.6, 0.2)] {
            model.editPose { $0.x = 20; $0.y = y; $0.heading = heading; $0.tilt = 0; $0.distance = 1e6 }
            let limit = model.distanceLimit
            model.editPose { $0.distance = fraction * limit }
            model.editPose { $0.tilt = 1.3 }  // as tilted as the drag can ask: eased back until the picture is within one world
            #expect(model.pose.tilt > 0 && model.pose.tilt < 1.3, "tilt \(model.pose.tilt)")
            let image = try world.render("patch-tilted-\(Int(y))-\(Int(heading * 10))")
            #expect(backgroundCount(image) == 0, "the horizon or an edge shows at y \(y): \(backgroundCount(image)) pixels")
        }
    }

    // MARK: The picture is the projection

    @Test func everyPixelOfTheHostedViewIsTheMapAtThePointTheModelSaysIsUnderIt() async throws {
        let world = try await Hosted()
        defer { world.finish() }
        let model = world.model
        let base = try #require(world.play.store.map)
        let block = 16
        try world.show(map: checkerboard(from: base, block: block))
        let L = model.worldSize
        let n = Float(base.n)
        let views: [(x: Float, y: Float, heading: Float, tilt: Float, fraction: Float)] = [
            (L / 2, 0, 0, 0, 1), (20, L / 2, 0, 0, 0.7), (0.4, L / 2 - 0.6, 0.6, 0.5, 0.4), (L - 1, 3, 1.2, 0, 0.5),
            (L / 2, L - 0.5, -0.8, 0.9, 0.35), (2, L / 2 + 0.5, 3.0, 0.4, 0.25),
        ]
        var compared = 0
        for (x, y, heading, tilt, fraction) in views {
            model.editPose { $0.x = x; $0.y = y; $0.heading = heading; $0.tilt = 0; $0.distance = 1e6 }
            let wanted = model.pose.distance * fraction
            model.editPose { $0.distance = wanted; $0.tilt = tilt }
            let image = try world.render("patch-projection-\(Int(x))-\(Int(y))")
            let pose = model.pose
            let aspect = Float(image.width) / Float(image.height)
            for i in 1..<24 {
                for j in 1..<18 {
                    let ndc = SIMD2(Float(i) / 12 - 1, Float(j) / 9 - 1)
                    guard let at = pose.point(at: ndc, aspect: aspect, size: L) else { continue }
                    let u = at.x / L * n, v = at.y / L * n  // the cell, fractional
                    let (fu, fv) = (u.truncatingRemainder(dividingBy: Float(block)), v.truncatingRemainder(dividingBy: Float(block)))
                    let (cu, cv) = (u - 1.5, v - 1.5)  // clear of the edge of a block by a cell and a half (the sampler's blend)
                    _ = (cu, cv)
                    guard fu > 1.5, fu < Float(block) - 1.5, fv > 1.5, fv < Float(block) - 1.5 else { continue }
                    let rain = (Int(u) / block + Int(v) / block) % 2 == 0
                    let px = min(Int((ndc.x * 0.5 + 0.5) * Float(image.width)), image.width - 1)
                    let py = min(Int((0.5 - ndc.y * 0.5) * Float(image.height)), image.height - 1)
                    let blue = Int(image.pixel(px, py).b)
                    guard blue > rainBlue || blue < seaBlue else { continue }  // (a pixel the territory or a hemisphere tint could shade)
                    #expect((blue > rainBlue) == rain, "(\(px), \(py)) at the world point (\(at.x), \(at.y)) heading \(heading): blue \(blue), the map has \(rain ? "rain" : "sea")")
                    compared += 1
                }
            }
        }
        #expect(compared > 600, "enough pixels were compared: \(compared)")
    }

    @Test func aStripeOfRainAcrossTheWrapInYAndAcrossXZeroIsOneUnbrokenStripe() async throws {
        let world = try await Hosted()
        defer { world.finish() }
        let model = world.model
        var map = try #require(world.play.store.map)
        let n = map.n
        let L = model.worldSize
        // Rain in the rows and the columns around the wrap in y (row n/2) and around x = 0 (column 0), 6 cells wide each.
        map.p = [UInt8](repeating: 0, count: n * n)
        map.dc = [UInt8](repeating: 0, count: n * n)
        map.zeta = [Int8](repeating: 0, count: n * n)
        map.ts = [UInt8](repeating: 0, count: n * n)
        for cy in (n / 2 - 3)..<(n / 2 + 3) { for cx in 0..<n { map.p[cy * n + cx] = 255 } }
        for cx in Array(0..<3) + Array((n - 3)..<n) { for cy in 0..<n { map.p[cy * n + cx] = 255 } }
        try world.show(map: map)
        // Centred on the wrap: the horizontal stripe is a band across the middle of the picture, and the vertical one across x = 0 too.
        model.editPose { $0.x = 0.5; $0.y = L / 2; $0.heading = 0; $0.tilt = 0; $0.distance = 1e6 }
        let half = model.pose.distance * 0.5
        model.editPose { $0.distance = half }
        let image = try world.render("patch-stripes")
        func isRain(_ x: Int, _ y: Int) -> Bool { Int(image.pixel(x, y).b) > rainBlue }
        // A column through the picture, well clear of the vertical stripe: the rain is one run, with no gap at the wrap in y.
        let column = image.width * 3 / 4
        var runs = 0, last = false, length = 0
        for y in 0..<image.height {
            let now = isRain(column, y)
            if now { length += 1 }
            if now && !last { runs += 1 }
            last = now
        }
        #expect(runs == 1 && length > 20, "the stripe across the wrap in y is one band: \(runs) runs, \(length) rows")
        // A row through the picture's top, clear of the horizontal stripe: one run across x = 0.
        let row = image.height / 8
        runs = 0; last = false; length = 0
        for x in 0..<image.width {
            let now = isRain(x, row)
            if now { length += 1 }
            if now && !last { runs += 1 }
            last = now
        }
        #expect(runs == 1 && length > 20, "the stripe across x = 0 is one band: \(runs) runs, \(length) columns")
        // The pose goes on across the wrap: the centre at y = L/2 - 0.1 and L/2 + 0.1 draw the same stripe within a pixel of each other.
        var bands: [Int] = []
        for y in [L / 2 - 0.1, L / 2 + 0.1] {
            model.editPose { $0.y = y }
            let moved = try world.render("patch-stripes-\(Int(y * 10))")
            var top = -1, bottom = -1
            for r in 0..<moved.height where Int(moved.pixel(moved.width * 3 / 4, r).b) > rainBlue { if top < 0 { top = r }; bottom = r }
            bands.append(bottom - top)
        }
        #expect(abs(bands[0] - bands[1]) <= 2, "the stripe is as wide on either side of the wrap: \(bands)")
    }

    // MARK: A layer of a copy

    @Test func anOrganismJustPastTheWrapIsDrawnWhereTheMapPutsItsPlace() async throws {
        let world = try await Hosted()
        defer { world.finish() }
        let model = world.model
        let L = model.worldSize
        let renderer = try world.renderer()
        var display = model.display
        (display.agents, display.bonds, display.frame, display.territory, display.map) = (true, false, false, .off, false)
        model.display = display
        // One organism a hair past the wrap, one a hair before it, the others away in the middle: the view is centred just short of it.
        var frame = try #require(world.play.store.frame)
        for i in 0..<frame.n { (frame.pos[3 * i], frame.pos[3 * i + 1], frame.pos[3 * i + 2]) = (L / 2, 2, 12) }
        (frame.pos[0], frame.pos[1], frame.pos[2]) = (24, L / 2 + 0.6, 0.5)
        (frame.pos[3], frame.pos[4], frame.pos[5]) = (30, L / 2 - 0.6, 0.5)
        renderer.showWorld(frame: frame)
        model.editPose { $0.x = 27; $0.y = L / 2 - 2.2; $0.heading = 0; $0.tilt = 0; $0.distance = 14 }
        let image = try world.render("patch-organisms-at-the-wrap")
        let projector = model.projector(size: model.freeRect.size)
        let scale = Float(image.width) / Float(model.freeRect.width)
        for (index, name) in [(0, "past the wrap"), (1, "before the wrap")] {
            let at = try #require(projector.screen(frame.position(index)))
            let (px, py) = (Int(Float(at.x) * scale), Int(Float(at.y) * scale))
            var lit = false
            for dy in -1...1 { for dx in -1...1 where !isBackground(image.pixel(px + dx, py + dy)) { lit = true } }
            #expect(lit, "the organism \(name) is drawn at its image, (\(px), \(py))")
        }
    }
}
