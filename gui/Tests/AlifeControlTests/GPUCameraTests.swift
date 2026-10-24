import AlifeCore
import AppKit
import Foundation
import Metal
import SwiftUI
import Testing
import simd

@testable import AlifeControl

/// The tilt to edge-on, the haze and the sky, drawn by the renderer of the **hosted** `WorldView` (DESIGN §36.5, camera controls). Every
/// test goes through the whole `WorldView` in an `NSHostingView`, as the app does, and takes the renderer from the view that owns it (the
/// lesson of I61). Opt in with `ALIFE_GPU_TESTS=1 swift test --filter GPUCameraTests`; images go under `$ALIFE_GPU_TESTS_OUT`.
///
/// The detector of a hard edge: a uniform map with the equator lines off (so every step in a column is the world's edge), a median of
/// five rows down each column, and the largest difference between two neighbouring rows of what is left. With the haze it is a few levels of 255 at most (a gradient over tens of rows); without it the
/// edge of the world is a step of tens (`theDetectorSeesTheEdgeWithoutTheHaze`).
@MainActor
@Suite(
    "GPU camera: tilt, haze and sky of the World (hosted)",
    .serialized,
    .enabled(
        if: ProcessInfo.processInfo.environment["ALIFE_GPU_TESTS"] == "1",
        "set ALIFE_GPU_TESTS=1 to compile the shaders and render on the GPU")
)
struct GPUCameraTests {
    private func hosted(size: CGSize = CGSize(width: 900, height: 560)) async throws -> HostedWorld {
        let world = try await HostedWorld(stream: Repo.playFixtureURL("3d"), size: size, dockCollapsed: true)
        world.model.reduceMotion = { true }
        world.model.display.minimap = false
        world.model.dismissHint()
        world.settle()
        return world
    }

    /// The fixture's map with nothing in it: the sea of a cold half, the same everywhere but for the 3% tint of the hemispheres.
    private func uniform(from base: PlayWeatherMap) -> PlayWeatherMap {
        var map = base
        let n = map.n
        map.p = [UInt8](repeating: 0, count: n * n)
        map.dc = [UInt8](repeating: 0, count: n * n)
        map.zeta = [Int8](repeating: 0, count: n * n)
        map.ts = [UInt8](repeating: 0, count: n * n)
        return map
    }

    /// The largest step between two neighbouring rows of any column, in levels of 255 per channel, after a median of five rows; and
    /// whether any pixel is exactly black (the clear colour is not: a black pixel is a NaN or a missing write).
    private func worstStep(_ image: OffscreenImage) -> (step: Int, black: Int, at: (x: Int, y: Int)) {
        var worst = 0, black = 0, at = (x: 0, y: 0)
        for x in stride(from: 0, to: image.width, by: 3) {
            var column: [[Int]] = [[], [], []]
            for y in 0..<image.height {
                let p = image.pixel(x, y)
                if p.r == 0 && p.g == 0 && p.b == 0 { black += 1 }
                column[0].append(Int(p.r)); column[1].append(Int(p.g)); column[2].append(Int(p.b))
            }
            for channel in column {
                var filtered = channel
                for y in 2..<(channel.count - 2) { filtered[y] = channel[(y - 2)...(y + 2)].sorted()[2] }
                for y in 3..<(channel.count - 3) {
                    let step = abs(filtered[y + 1] - filtered[y])
                    if step > worst { worst = step; at = (x, y) }
                }
            }
        }
        return (worst, black, at)
    }

    /// The pose at `tilt` degrees, `heading` degrees, at `x`, `y`, at the farthest distance it allows.
    private func place(_ model: WorldModel, tilt: Float, heading: Float, x: Float, y: Float, z: Float = 0) {
        model.editPose {
            $0.x = x; $0.y = y; $0.z = z; $0.tilt = tilt * .pi / 180; $0.heading = heading * .pi / 180; $0.distance = 1e6
        }
    }

    // MARK: No hard edge

    @Test func theWorldHasNoHardEdgeAtAnyTiltAndHeadingAtTheFarthestDistanceAndNothingIsNaN() async throws {
        let world = try await hosted()
        defer { world.finish() }
        let model = world.model
        let L = model.worldSize
        try world.show(map: uniform(from: try #require(world.play.store.map)))
        try world.renderer().worldDraw.equatorLines = false  // (a line is a step to the detector)
        var worst = 0, rendered = 0
        var where_ = ""
        // The point looked at at the middle of the world, on the equator and on the wrap in y (where the copies are).
        for (x, y) in [(L / 2, Float(0)), (L / 4, L / 2), (L - 3, L / 2 - 1)] {
            for tilt: Float in [0, 15, 30, 45, 60, 70, 80, 85, 90] {
                for heading: Float in [0, 22.5, 45, 90, 135, 180, 270, -60] {
                    place(model, tilt: tilt, heading: heading, x: x, y: y)
                    let free = model.freeRect
                    let image = try world.renderer().renderOffscreen(width: Int(free.width), height: Int(free.height))
                    let found = worstStep(image)
                    if found.step > 12 || (x == L / 2 && heading == 45 && [45, 70, 85, 90].contains(tilt)) {
                        _ = try world.render("camera-edge-\(Int(tilt))-\(Int(heading))-\(Int(y))")  // (a few for the eye, and any that fail)
                    }
                    rendered += 1
                    #expect(found.black == 0, "tilt \(tilt) heading \(heading) at (\(x), \(y)): \(found.black) black pixels (NaN?)")
                    #expect(found.step <= 12, "tilt \(tilt) heading \(heading) at (\(x), \(y)): a step of \(found.step) levels at \(found.at)")
                    if found.step > worst { worst = found.step; where_ = "tilt \(tilt) heading \(heading) (\(x), \(y))" }
                }
            }
        }
        #expect(rendered == 3 * 9 * 8)
        print("camera: the largest step over \(rendered) views is \(worst) of 255 (\(where_))")
    }

    @Test func theDetectorSeesTheEdgeWithoutTheHaze() async throws {
        let world = try await hosted()
        defer { world.finish() }
        let model = world.model
        let L = model.worldSize
        try world.show(map: uniform(from: try #require(world.play.store.map)))
        let renderer = try world.renderer()
        renderer.worldDraw.equatorLines = false
        renderer.worldDraw.fog = false
        var steps: [Int] = []
        for tilt: Float in [60, 80, 90] {
            place(model, tilt: tilt, heading: 0, x: L / 2, y: 0)
            steps.append(worstStep(try world.render("camera-nofog-\(Int(tilt))")).step)
        }
        #expect(steps.contains { $0 > 20 }, "without the haze the edge of the world is a step the detector finds: \(steps)")
        renderer.worldDraw.fog = true
        place(model, tilt: 80, heading: 0, x: L / 2, y: 0)
        #expect(worstStep(try world.render("camera-fog-80")).step <= 12, "and with it, not")
    }

    @Test func theHorizonIsTheHazeAndTheSkyAboveItGoesToTheClearColour() async throws {
        let world = try await hosted()
        defer { world.finish() }
        let model = world.model
        let L = model.worldSize
        try world.show(map: uniform(from: try #require(world.play.store.map)))
        place(model, tilt: 90, heading: 0, x: L / 2, y: L / 4, z: model.height / 2)
        let image = try world.render("camera-side-uniform")
        let x = image.width / 2
        let horizon = image.height / 2  // edge-on: the horizon is the middle of the picture
        let haze = WorldFog.colour(exposure: model.display.exposure, opacity: WorldLevels.mapOpacity(distance: model.pose.distance, size: L))
        let expected = haze * 255
        // Just below and above the horizon: the haze.
        for y in [horizon - 2, horizon + 2] {
            let p = image.pixel(x, y)
            let got = SIMD3(Float(p.r), Float(p.g), Float(p.b))
            #expect(simd_distance(got, expected) < 16, "row \(y): \(got) against the haze \(expected)")
        }
        // The top of the picture: the sky is darker, towards the clear colour; the bottom of the floor is the sea.
        let top = image.pixel(x, 2), floorNear = image.pixel(x, image.height - 3), mid = image.pixel(x, horizon - 2)
        func luma(_ p: (r: UInt8, g: UInt8, b: UInt8)) -> Int { Int(p.r) + 2 * Int(p.g) + Int(p.b) }
        #expect(luma(top) < luma(mid), "the sky falls away from the haze: \(luma(top)) over \(luma(mid))")
        #expect(luma(top) >= luma((9, 10, 14)) - 4, "and does not go darker than the clear colour")
        #expect(luma(floorNear) > luma(mid), "the near floor is the sea, brighter than the haze: \(luma(floorNear)) against \(luma(mid))")
        // The Top view is as it was: no sky, no haze, the sea all over.
        place(model, tilt: 0, heading: 0, x: L / 2, y: L / 4)
        let top0 = try world.render("camera-top-uniform")
        let corner = top0.pixel(2, 2)
        #expect(corner.b > 60 && simd_distance(SIMD3(Float(corner.r), Float(corner.g), Float(corner.b)), SIMD3(Float(top0.pixel(top0.width / 2, top0.height / 2).r),
                Float(top0.pixel(top0.width / 2, top0.height / 2).g), Float(top0.pixel(top0.width / 2, top0.height / 2).b))) < 10, "the corner of the Top view is the sea, as the middle is")
    }

    @Test func theBrightnessScalesTheHazeAndTheSkyWithTheMap() async throws {
        let world = try await hosted()
        defer { world.finish() }
        let model = world.model
        let L = model.worldSize
        try world.show(map: uniform(from: try #require(world.play.store.map)))
        place(model, tilt: 90, heading: 0, x: L / 2, y: L / 4, z: model.height / 2)
        var lumas: [Int] = []
        for exposure in [Float(0.8), 1.0, 1.5] {
            model.display.exposure = exposure
            let image = try world.render("camera-haze-\(Int(exposure * 100))")
            let p = image.pixel(image.width / 2, image.height / 2 - 2)
            lumas.append(Int(p.r) + 2 * Int(p.g) + Int(p.b))
        }
        model.display.exposure = 1
        #expect(lumas[0] < lumas[1] && lumas[1] < lumas[2], "the haze follows the Brightness setting like the sea does: \(lumas)")
    }

    // MARK: The seam at grazing angles

    @Test func theWrapInYIsContinuousAtGrazingAnglesAndEachCopyHoldsTheMapWhereItBelongs() async throws {
        let world = try await hosted()
        defer { world.finish() }
        let model = world.model
        let L = model.worldSize
        let base = try #require(world.play.store.map)
        let n = base.n
        let block = max(n / 8, 2)
        // Rain in bands of rows (the wrap in y cuts across them), nothing else.
        var map = uniform(from: base)
        for cy in 0..<n { for cx in 0..<n where (cy / block) % 2 == 0 { map.p[cy * n + cx] = 255 } }
        try world.show(map: map)
        try world.renderer().worldDraw.equatorLines = false
        var compared = 0
        for tilt: Float in [45, 70, 85] {
            for heading: Float in [0, 180] {
                place(model, tilt: tilt, heading: heading, x: L / 2, y: L / 2 + 1)
                let image = try world.render("camera-seam-\(Int(tilt))-\(Int(heading))")
                let pose = model.pose
                for column in [image.width / 3, image.width / 2, 2 * image.width / 3] {
                    for row in stride(from: image.height / 2, to: image.height - 4, by: 3) {
                        let ndc = SIMD2(Float(column) / Float(image.width) * 2 - 1, 1 - Float(row) / Float(image.height) * 2)
                        guard let hit = pose.offset(at: ndc, aspect: model.aspect, planeZ: 0),
                              WorldFog.amount(c: WorldFog.chebyshev(hit), size: L) < 0.05 else { continue }
                        // (a pixel covers many cells where the floor is foreshortened: only where it covers a fraction of one)
                        let below = SIMD2(ndc.x, 1 - Float(row + 1) / Float(image.height) * 2)
                        guard let next = pose.offset(at: below, aspect: model.aspect, planeZ: 0),
                              abs(next.y - hit.y) < 0.25 * L / Float(n), abs(next.x - hit.x) < 0.25 * L / Float(n) || true else { continue }
                        let y = Torus.wrap(pose.y + hit.y, size: L)
                        let cy = min(Int(y / L * Float(n)), n - 1)
                        // (stay off the borders of the bands: a pixel is not a point)
                        let fraction = y / L * Float(n) - Float(cy)
                        guard fraction > 0.25, fraction < 0.75 else { continue }
                        let rain = (cy / block) % 2 == 0
                        let blue = Int(image.pixel(column, row).b)
                        guard blue > 150 || blue < 125 else { continue }  // (the equator's line and the tint of the hemispheres are between)
                        #expect((blue > 150) == rain, "tilt \(tilt) heading \(heading): row \(row) is y \(y) cy \(cy), rain \(rain), blue \(blue)")
                        compared += 1
                    }
                }
            }
        }
        #expect(compared > 400, "the bands were compared: \(compared)")
    }

    // MARK: The recorded planet

    private final class Clock { var t: Double = 1000 }

    /// A real 3D planet frame (`ALIFE_V2_STREAM`, a `.bin` of `PlayStream` records: the 64k3d world, 44,162 organisms in a world of 320) at
    /// the four standard views, in the world's own colours and in the height colour, at Home's distance and at a storm's region: the
    /// picture of the window (the renderer of the hosted view with the overlay and the controls drawn over it), saved for looking at.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ALIFE_V2_STREAM"] != nil, "set ALIFE_V2_STREAM=<a recorded planet stream>"))
    func rendersARealPlanetAtTheFourStandardViews() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["ALIFE_V2_STREAM"])
        let world = try await HostedWorld(stream: URL(fileURLWithPath: path), size: CGSize(width: 1500, height: 900), dockCollapsed: true,
                                          timeout: .seconds(90))
        defer { world.finish() }
        let model = world.model
        model.reduceMotion = { true }
        model.display.minimap = true
        model.dismissHint()
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
        let storm = try #require(world.play.store.storms?.storms.storms.first)
        for (series, region) in [("home", false), ("region", true)] {
            for (colour, height) in [("engine", false), ("height", true)] {
                model.setColourMode(height ? HeightColour.modeName : try #require(header.colorModes.first))
                for preset in ViewPreset.allCases {
                    model.goHome()
                    if region { model.flyTo(SIMD3(Float(storm.x), Float(storm.y), 0), distance: model.flyDistance(.region)) }
                    settle()
                    model.viewPreset(preset)
                    settle()
                    let image = try world.composite("planet-\(series)-\(preset.rawValue)-\(colour)")
                    // Nothing is NaN, and the haze and the sky are there from the Low view on.
                    var black = 0
                    for y in 0..<image.height { for x in 0..<image.width where image.pixel(x, y) == (0, 0, 0) { black += 1 } }
                    #expect(black == 0, "\(series) \(preset) \(colour): \(black) black pixels")
                }
            }
        }
    }
}
