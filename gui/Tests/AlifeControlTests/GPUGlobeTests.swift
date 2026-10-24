import AlifeCore
import AppKit
import Foundation
import Metal
import SwiftUI
import Testing
import simd

@testable import AlifeControl

/// The globe overview and its morph, drawn by the renderer of the **hosted** `WorldView` (DESIGN §36.10, the spec's 7.2). Every test goes
/// through the whole `WorldView` in an `NSHostingView`, as the app does (the lesson of I61). Opt in with
/// `ALIFE_GPU_TESTS=1 swift test --filter GPUGlobeTests`; images go under `$ALIFE_GPU_TESTS_OUT`. `ALIFE_V2_STREAM` names a recorded planet
/// (see `GPUCameraTests`) for the tests that want a real frame.
@MainActor
@Suite(
    "GPU globe: the mesh, the morph, the caps and the sky of the World (hosted)",
    .serialized,
    .enabled(
        if: ProcessInfo.processInfo.environment["ALIFE_GPU_TESTS"] == "1",
        "set ALIFE_GPU_TESTS=1 to compile the shaders and render on the GPU")
)
struct GPUGlobeTests {
    private final class Clock { var t: Double = 1000 }

    /// A hosted world with the globe on, the minimap and the hint off, over the 3D fixture.
    static func hosted(stream: URL? = nil, size: CGSize = CGSize(width: 900, height: 600)) async throws -> HostedWorld {
        let world = try await HostedWorld(stream: stream ?? Repo.playFixtureURL("3d"), size: size, dockCollapsed: true, timeout: .seconds(90))
        let model = world.model
        model.reduceMotion = { true }
        model.display.minimap = false
        model.dismissHint()
        model.display.globe = true
        model.pushCamera()
        world.settle()
        return world
    }

    /// The recorded planet when `ALIFE_V2_STREAM` names one, else the committed 3D fixture.
    static func planetURL() -> URL {
        ProcessInfo.processInfo.environment["ALIFE_V2_STREAM"].map { URL(fileURLWithPath: $0) } ?? Repo.playFixtureURL("3d")
    }

    /// Puts the view at the flat-equivalent distance that gives the morph scalar `s` (the centre at `x`, `y`; straight down), stopping any
    /// flight, and returns the image the renderer draws.
    @discardableResult
    static func render(_ world: HostedWorld, s: Float, x: Float? = nil, y: Float? = nil, tilt: Float = 0, name: String? = nil) throws
        -> OffscreenImage {
        let model = world.model
        let L = model.worldSize
        let d = s <= 0 ? 0.08 * L : Morph.distance(forS: s, L: L, fovY: model.pose.fovY, aspect: model.aspect)
        model.editPose {
            if let x { $0.x = x }
            if let y { $0.y = y }
            $0.distance = d
            $0.tilt = tilt
        }
        world.settle(2)
        let renderer = try world.renderer()
        let free = model.freeRect
        let image = try renderer.renderOffscreen(width: Int(free.width), height: Int(free.height))
        if let name { try PNG.write(image, to: HostedWorld.outputDirectory.appendingPathComponent("\(name).png")) }
        return image
    }

    @Test func rendersTheMorphAtEveryScalar() async throws {
        let world = try await Self.hosted(stream: Self.planetURL())
        defer { world.finish() }
        let L = world.model.worldSize
        for (label, y) in [("eq", Float(0)), ("belt", 0.12 * L)] {
            for s in [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1] as [Float] {
                try Self.render(world, s: s, x: L / 2, y: y, name: "globe-\(label)-s\(Int(s * 100))")
            }
        }
    }

    @Test func rendersThePolesAndTheTilt() async throws {
        let world = try await Self.hosted(stream: Self.planetURL())
        defer { world.finish() }
        let L = world.model.worldSize
        for s in [0.5, 0.7, 0.9, 1] as [Float] {
            try Self.render(world, s: s, x: L / 2, y: 0.46 * L, name: "pole-s\(Int(s * 100))")
            try Self.render(world, s: s, x: L / 2, y: 0.12 * L, tilt: 60 * .pi / 180, name: "tilt-s\(Int(s * 100))")
        }
    }

    // MARK: Measures

    /// Y' (0...1) of every pixel of the image, the Rec. 709 luma of the 8-bit values.
    static func luma(_ image: OffscreenImage) -> [Float] {
        var out = [Float](); out.reserveCapacity(image.width * image.height)
        for y in 0..<image.height { for x in 0..<image.width {
            let p = image.pixel(x, y)
            out.append((0.2126 * Float(p.r) + 0.7152 * Float(p.g) + 0.0722 * Float(p.b)) / 255)
        } }
        return out
    }

    static func black(_ image: OffscreenImage) -> Int {
        var n = 0
        for y in 0..<image.height { for x in 0..<image.width where image.pixel(x, y) == (0, 0, 0) { n += 1 } }
        return n
    }

    /// The mean absolute difference per channel between two images of one size, in 0...1.
    static func difference(_ a: OffscreenImage, _ b: OffscreenImage) -> Float {
        var sum = 0
        for i in 0..<a.bgra.count where i % 4 != 3 { sum += abs(Int(a.bgra[i]) - Int(b.bgra[i])) }
        return Float(sum) / Float(a.bgra.count / 4 * 3) / 255
    }

    /// Only the map: no organisms, bonds, territory or frame (and no equator lines), the fixture's map as it is.
    static func mapOnly(_ world: HostedWorld, uniform: Bool = false) throws {
        var map = try #require(world.play.store.map)
        if uniform {
            let n = map.n
            map.p = [UInt8](repeating: 0, count: n * n); map.dc = [UInt8](repeating: 0, count: n * n)
            map.zeta = [Int8](repeating: 0, count: n * n); map.ts = [UInt8](repeating: 0, count: n * n)
        }
        try world.show(map: map)
        try world.renderer().worldDraw.equatorLines = false
    }

    // MARK: Tests

    /// At s = 0 the globe path is the flat close-up: the same picture as the quad's, up to the lit and rim weights (0), the mesh's 1-unit
    /// facets and the world frame (off in both).
    @Test func theGlobePathAtZeroIsTheFlatPicture() async throws {
        let world = try await Self.hosted()
        defer { world.finish() }
        try Self.mapOnly(world)
        let L = world.model.worldSize
        for tilt: Float in [0, 40] {
            #expect(try world.renderer().worldDraw.vantage != nil, "the globe path")
            let globe = try Self.render(world, s: 0, x: L / 3, y: 0.1 * L, tilt: tilt * .pi / 180, name: "equiv-globe-\(Int(tilt))")
            world.model.display.globe = false
            world.model.editPose { $0.x = L / 3; $0.y = 0.1 * L; $0.distance = 0.08 * L; $0.tilt = tilt * .pi / 180 }
            world.settle(2)
            #expect(try world.renderer().worldDraw.vantage == nil && !world.model.globeEnabled, "the flat path")
            let free = world.model.freeRect
            let flat = try world.renderer().renderOffscreen(width: Int(free.width), height: Int(free.height))
            try PNG.write(flat, to: HostedWorld.outputDirectory.appendingPathComponent("equiv-flat-\(Int(tilt)).png"))
            let d = Self.difference(globe, flat)
            print("equivalence tilt \(tilt): mean abs difference \(d)")
            #expect(d < 0.01, "tilt \(tilt): \(d)")
            world.model.display.globe = true
            world.settle(2)
        }
    }

    /// The Home frame and a polar cold-sea globe: the luminance of the disc and of the frame (the floors of the spec's 7.2, set from these).
    @Test func theLuminanceOfTheGlobeAtItsPoses() async throws {
        let world = try await Self.hosted(stream: Self.planetURL(), size: CGSize(width: 1100, height: 700))
        defer { world.finish() }
        let L = world.model.worldSize
        func stats(_ image: OffscreenImage) -> (mean: Float, discMean: Float, p5: Float, under: Float) {
            let y = Self.luma(image)
            let mean = y.reduce(0, +) / Float(y.count)
            // The disc: the pixels inside the circle of the sphere's silhouette at its centre (the globe is centred in the frame).
            var disc: [Float] = []
            let cx = Float(image.width) / 2, cy = Float(image.height) / 2
            let r = 0.8 * Float(min(image.width, image.height)) / 2 * 0.9
            for j in 0..<image.height { for i in 0..<image.width where hypot(Float(i) - cx, Float(j) - cy) < r { disc.append(y[j * image.width + i]) } }
            disc.sort()
            return (mean, disc.reduce(0, +) / Float(disc.count), disc[disc.count / 20], Float(disc.filter { $0 < 0.10 }.count) / Float(disc.count))
        }
        try Self.mapOnly(world)
        let home = try Self.render(world, s: 1, x: L / 2, y: 0, name: "lum-home-map")
        let h = stats(home)
        print("luminance home (map only):", h)
        #expect(h.mean >= 0.14 && h.p5 >= 0.15 && h.under < 0.01, "Home: \(h)")
        try Self.mapOnly(world, uniform: true)
        let cold = try Self.render(world, s: 1, x: L / 2, y: 0.38 * L, name: "lum-cold")
        let c = stats(cold)
        print("luminance cold sea, polar:", c)
        #expect(c.discMean >= 0.19 && c.p5 >= 0.13, "polar cold sea: \(c)")
        #expect(Self.black(home) == 0 && Self.black(cold) == 0, "no NaN")
        let flat = try Self.render(world, s: 0, x: L / 2, y: 0.1 * L, name: "lum-flat-cold")
        let f = stats(flat)
        print("luminance flat region, cold:", f)
        #expect(f.mean >= 0.14 && f.under < 0.03, "flat region over cold sea: \(f)")
    }

    /// The zoom from 0.10 L to 0.35 L in equal log steps at a fixed centre: no pair of consecutive frames differs by more than
    /// max(0.02, 3 x the median difference), the picture never goes black or NaN, and no step at s = 0.5 where the derived level changes.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ALIFE_V2_STREAM"] != nil, "needs the recorded planet (ALIFE_V2_STREAM): the fixture has too few organisms")) func theTransitionHasNoDiscontinuity() async throws {
        let world = try await Self.hosted(stream: Self.planetURL(), size: CGSize(width: 700, height: 460))
        defer { world.finish() }
        let L = world.model.worldSize
        for ybar in [Float(0), 38] {
            var frames: [OffscreenImage] = []
            for i in 0..<90 {
                let d = 0.10 * L * pow(3.5, Float(i) / 89)
                let model = world.model
                model.editPose { $0.x = L / 2; $0.y = ybar; $0.distance = d; $0.tilt = 0 }
                world.settle(1)
                let free = model.freeRect
                frames.append(try world.renderer().renderOffscreen(width: Int(free.width), height: Int(free.height)))
            }
            let steps = zip(frames, frames.dropFirst()).map { Self.difference($0, $1) }
            let median = steps.sorted()[steps.count / 2]
            let worst = steps.max() ?? 0
            print("transition ybar \(ybar): median \(median), worst \(worst) at \(steps.firstIndex(of: worst) ?? -1)")
            #expect(worst < max(0.02, 3 * median), "ybar \(ybar): worst \(worst) against median \(median)")
            #expect(frames.allSatisfy { Self.black($0) == 0 }, "no NaN")
        }
    }

    /// One world: the picture does not depend on which turn of the world the centre is in (x and x + L), at the flat close-up, mid-morph and
    /// the globe; and with a uniform map no column is a seam: the step across the centre column (x = 0) is at most twice the median step.
    @Test func thereIsNoRepeatAndNoSeamInX() async throws {
        let world = try await Self.hosted(stream: Self.planetURL(), size: CGSize(width: 700, height: 460))
        defer { world.finish() }
        let L = world.model.worldSize
        try Self.mapOnly(world)
        for s in [0.05, 0.2, 0.5, 1] as [Float] {
            let a = try Self.render(world, s: s, x: 10, y: 0.1 * L)
            let b = try Self.render(world, s: s, x: 10 + L, y: 0.1 * L)
            #expect(Self.difference(a, b) < 0.002, "s \(s): x and x + L differ by \(Self.difference(a, b))")
        }
        try Self.mapOnly(world, uniform: true)
        for s in [0.05, 0.25] as [Float] {
            let image = try Self.render(world, s: s, x: 0, y: 0.1 * L, name: "seam-x-s\(Int(s * 100))")
            let row = image.height / 2
            var steps: [Int] = []
            for x in 1..<image.width { steps.append(abs(Int(image.pixel(x, row).b) - Int(image.pixel(x - 1, row).b))) }
            let median = steps.sorted()[steps.count / 2]
            let across = steps[image.width / 2 - 1]
            #expect(across <= max(2 * median, 2), "s \(s): the step across x = 0 is \(across), the median \(median)")
        }
    }

    /// The caps: the frame centred on the pole has no hole at the globe (the pixel at the pole is the cap's colour, not the space's), and the
    /// pole row's ring is closed (no pixel of the space colour inside the disc).
    @Test func thePolesAreClosedByTheCaps() async throws {
        let world = try await Self.hosted(stream: Self.planetURL(), size: CGSize(width: 700, height: 500))
        defer { world.finish() }
        let L = world.model.worldSize
        try Self.mapOnly(world)
        for y in [0.5 * L, 0.0] {
            let image = try Self.render(world, s: 1, x: 0.25 * L, y: y == 0 ? 0.5 * L : y, name: "cap-\(Int(y))")
            let p = image.pixel(image.width / 2, image.height / 2)
            let y709 = (0.2126 * Float(p.r) + 0.7152 * Float(p.g) + 0.0722 * Float(p.b)) / 255
            #expect(y709 > 0.15, "the pole's pixel is \(p)")
            // The whole disc around the centre (half its radius) is lit: no space showing through.
            var holes = 0
            let r = Int(0.15 * Float(min(image.width, image.height)))
            for j in -r...r { for i in -r...r where i * i + j * j < r * r {
                let q = image.pixel(image.width / 2 + i, image.height / 2 + j)
                if Int(q.r) + Int(q.g) + Int(q.b) < 45 { holes += 1 }
            } }
            #expect(holes == 0, "\(holes) dark pixels around the pole")
        }
    }

    /// Life at every s: the organisms (as instances) change the picture at every s, and the globe path draws none of the far side (the
    /// number of pixels the organisms touch beyond the limb of a globe is zero).
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ALIFE_V2_STREAM"] != nil, "needs the recorded planet (ALIFE_V2_STREAM): the fixture has too few organisms")) func theLifeIsVisibleAtEverySAndNothingIsBeyondTheLimb() async throws {
        let world = try await Self.hosted(stream: Self.planetURL(), size: CGSize(width: 700, height: 460))
        defer { world.finish() }
        let L = world.model.worldSize
        for s in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1] as [Float] {
            var display = world.model.display
            display.agents = false; display.bonds = false
            world.model.display = display
            let bare = try Self.render(world, s: s, x: L / 2, y: 0.1 * L)
            display.agents = true
            world.model.display = display
            let full = try Self.render(world, s: s, x: L / 2, y: 0.1 * L, name: "life-s\(Int(s * 100))")
            var changed = 0
            for i in stride(from: 0, to: full.bgra.count, by: 4) where full.bgra[i...(i + 2)] != bare.bgra[i...(i + 2)] { changed += 1 }
            let share = Float(changed) / Float(full.width * full.height)
            print("life s \(s): \(share) of the frame touched by organisms")
            #expect(share > 0.01, "s \(s): the organisms touch \(share)")
            #expect(Self.black(full) == 0, "no NaN at s \(s)")
        }
    }

    /// Sky and space: stars only from the globe's s, the frame's corner is space at the globe and the haze sky is gone.
    @Test func theSpaceBehindTheGlobeIsSpace() async throws {
        let world = try await Self.hosted(size: CGSize(width: 700, height: 460))
        defer { world.finish() }
        let L = world.model.worldSize
        try Self.mapOnly(world)
        let globe = try Self.render(world, s: 1, x: L / 2, y: 0, name: "space")
        var corner = 0
        for y in 0..<12 { for x in 0..<12 { let p = globe.pixel(x, y); corner += Int(p.r) + Int(p.g) + Int(p.b) } }
        #expect(corner / 144 < 90, "the corner is space: \(corner / 144)")
        var stars = 0
        for y in 0..<globe.height { for x in 0..<globe.width where globe.pixel(x, y).b > 150 && abs(x - globe.width / 2) > globe.width / 3 { stars += 1 } }
        #expect(stars > 5, "stars outside the disc: \(stars)")
    }

    @Test func rendersTheTiltedGlobe() async throws {
        let world = try await Self.hosted(stream: Self.planetURL(), size: CGSize(width: 800, height: 500))
        defer { world.finish() }
        let L = world.model.worldSize
        for s in [0.05, 0.15, 0.3, 0.5, 1] as [Float] {
            for tilt in [60, 88] as [Float] {
                try Self.render(world, s: s, x: L / 2, y: 0.1 * L, tilt: tilt * .pi / 180, name: "tilted-s\(Int(s * 100))-t\(Int(tilt))")
            }
        }
    }

    /// The largest step between neighbouring rows of any column after a median of five (the detector of `GPUCameraTests`).
    static func worstStep(_ image: OffscreenImage) -> Int {
        var worst = 0
        for x in stride(from: 0, to: image.width, by: 3) {
            for channel in 0..<3 {
                var column: [Int] = []
                for y in 0..<image.height {
                    let p = image.pixel(x, y)
                    column.append(Int([p.r, p.g, p.b][channel]))
                }
                var filtered = column
                for y in 2..<(column.count - 2) { filtered[y] = column[(y - 2)...(y + 2)].sorted()[2] }
                for y in 3..<(column.count - 3) { worst = max(worst, abs(filtered[y + 1] - filtered[y])) }
            }
        }
        return worst
    }

    /// The haze of the flat close-up carries the edge of the mesh through the morph (the mesh and its tiles are hidden by it until s = 0.3):
    /// a uniform map tilted to the horizon at s up to 0.3, at every heading and at the wrap in y, has no hard edge.
    @Test func theHazeHidesTheEdgeOfTheMeshThroughTheMorph() async throws {
        let world = try await Self.hosted(stream: Self.planetURL(), size: CGSize(width: 700, height: 440))
        defer { world.finish() }
        let L = world.model.worldSize
        try Self.mapOnly(world, uniform: true)
        var worst = 0, where_ = ""
        for (x, y) in [(L / 2, Float(0)), (L / 4, L / 2), (L - 3, L / 2 - 1)] {
            for s in [0.02, 0.1, 0.2] as [Float] {
                for tilt: Float in [60, 80, 90] {
                    for heading: Float in [0, 45, 135] {
                        world.model.editPose { $0.heading = heading * .pi / 180 }
                        let image = try Self.render(world, s: s, x: x, y: y, tilt: tilt * .pi / 180,
                                                    name: x == L / 2 && heading == 0 ? "haze-s\(Int(s * 100))-t\(Int(tilt))" : nil)
                        let step = Self.worstStep(image)
                        #expect(Self.black(image) == 0, "NaN at s \(s) tilt \(tilt)")
                        if step > worst { worst = step; where_ = "s \(s) tilt \(tilt) heading \(heading) at (\(x), \(y))" }
                        #expect(step <= 14, "s \(s) tilt \(tilt) heading \(heading) at (\(x), \(y)): a step of \(step)")
                    }
                }
            }
        }
        print("globe haze: the largest step is \(worst) of 255 (\(where_))")
    }
}
