import AlifeCore
import AppKit
import Foundation
import Metal
import SwiftUI
import Testing
import simd

@testable import AlifeControl

/// The minimap's footprint at the globe, the glow and the pin (DESIGN §36.10, steps 8 and 9), through the hosted `WorldView` and the
/// overlay's own `Canvas` views. Opt in with `ALIFE_GPU_TESTS=1 swift test --filter GPUGlobeOverlayTests`.
@MainActor
@Suite(
    "GPU globe overlay: the minimap's cap, the glow, the pin (hosted)",
    .serialized,
    .enabled(
        if: ProcessInfo.processInfo.environment["ALIFE_GPU_TESTS"] == "1",
        "set ALIFE_GPU_TESTS=1 to compile the shaders and render on the GPU")
)
struct GPUGlobeOverlayTests {
    /// The RGBA bytes of a `CGImage`.
    private static func rgba(_ image: CGImage) -> (w: Int, h: Int, bytes: [UInt8]) {
        var bytes = [UInt8](repeating: 0, count: image.width * image.height * 4)
        let space = CGColorSpaceCreateDeviceRGB()
        bytes.withUnsafeMutableBytes { raw in
            let context = CGContext(data: raw.baseAddress, width: image.width, height: image.height, bitsPerComponent: 8,
                                    bytesPerRow: image.width * 4, space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        }
        return (image.width, image.height, bytes)
    }

    private static func light(_ a: (w: Int, h: Int, bytes: [UInt8]), at p: CGPoint, scale: Int) -> Float {
        var sum: Float = 0
        var n: Float = 0
        for dy in -2...2 { for dx in -2...2 {
            let x = Int(p.x) * scale + dx, y = Int(p.y) * scale + dy
            guard x >= 0, y >= 0, x < a.w, y < a.h else { continue }
            let o = 4 * (y * a.w + x)
            sum += 0.2126 * Float(a.bytes[o]) + 0.7152 * Float(a.bytes[o + 1]) + 0.0722 * Float(a.bytes[o + 2])
            n += 1
        } }
        return sum / max(n, 1) / 255
    }

    @Test func theMinimapShowsTheVisibleCapAndDimsTheRest() async throws {
        let world = try await GPUGlobeTests.hosted(stream: GPUGlobeTests.planetURL(), size: CGSize(width: 1100, height: 700))
        defer { world.finish() }
        let model = world.model
        model.display.minimap = true
        model.display.agents = false
        model.goHome()
        world.settle()
        let L = model.worldSize
        let box = try #require(model.hud.minimap)
        let drawing = try world.composite("globe-minimap-home")
        let composite = Self.rgba(try #require(world.lastComposite))
        let alone = Self.rgba(HostedWorld.cgImage(drawing))
        let inset = WorldInset(rect: box, size: L)
        // Inside the cap (east of the centre, a quarter of the way to the limb) the overlay changes the picture by little; the antipodal
        // meridian is dimmed by 35%.
        let x0 = model.pose.x
        let near = inset.point(forTorus: x0 + 0.1 * L, y: 0.02 * L)
        let far = inset.point(forTorus: x0 + 0.5 * L, y: Torus.wrap(-0.1 * L, size: L))
        let nearRatio = Self.light(composite, at: near, scale: 2) / max(Self.light(alone, at: near, scale: 2), 1e-3)
        let farRatio = Self.light(composite, at: far, scale: 2) / max(Self.light(alone, at: far, scale: 2), 1e-3)
        print("minimap cap: ratio inside \(nearRatio), outside \(farRatio)")
        #expect(abs(nearRatio - 1) < 0.08, "inside the cap: \(nearRatio)")
        #expect(abs(farRatio - 0.65) < 0.08, "outside the cap, dimmed by 35%: \(farRatio)")
        // The yellow outline is there: some pixels of the box are yellow (r, g high, b low).
        var yellow = 0
        for j in Int(box.minY) * 2..<Int(box.maxY) * 2 { for i in Int(box.minX) * 2..<Int(box.maxX) * 2 {
            let o = 4 * (j * composite.w + i)
            if composite.bytes[o] > 200, composite.bytes[o + 1] > 170, composite.bytes[o + 2] < 90 { yellow += 1 }
        } }
        #expect(yellow > 100, "the footprint's outline: \(yellow) yellow pixels")
        // The pole cap: centred at a high latitude the footprint is the band between the edge and the boundary; the picture stays.
        let polar = 0.42 * L
        model.editPose { $0.y = polar }
        world.settle()
        _ = try world.composite("globe-minimap-polar")
        #expect(model.inset?.footprintPieces(of: model.vantage()).map { $0.count == 1 } == true)
    }

    @Test func theGlowTakesWhatTheInstancesLeaveAndDrawsNothingWhereTheyAreDrawn() async throws {
        let world = try await GPUGlobeTests.hosted(stream: GPUGlobeTests.planetURL(), size: CGSize(width: 1100, height: 700))
        defer { world.finish() }
        let model = world.model
        let renderer = try world.renderer()
        model.goHome()
        world.settle()
        let home = renderer.worldDraw.glowAlpha
        print("glow alpha at Home:", home, "cover", model.coverRadius, "target", model.coverTarget, "display.agents", model.display.agents, "n", world.play.store.frame?.n ?? -1)
        #expect(home >= 0 && home <= 1)
        let with = try renderer.renderOffscreen(width: Int(model.freeRect.width), height: Int(model.freeRect.height))
        renderer.worldDraw.glow = false
        let without = try renderer.renderOffscreen(width: Int(model.freeRect.width), height: Int(model.freeRect.height))
        let lw = GPUGlobeTests.luma(with).reduce(0, +), lo = GPUGlobeTests.luma(without).reduce(0, +)
        print("glow: summed luminance with \(lw) without \(lo)")
        #expect(lw > lo * 1.0005, "the glow adds light")
        renderer.worldDraw.glow = true
        // close in on an organism: the instances are drawn at their size and the glow is gone
        let near = WorldLevels.flyDistance(.organism, size: model.worldSize, fovY: model.pose.fovY, aspect: model.aspect)
        model.editPose { $0.distance = near }
        model.pushCamera()
        #expect(renderer.worldDraw.glowAlpha == 0, "no glow at the organism level: \(renderer.worldDraw.glowAlpha)")
    }

    @Test func thePinStandsOnTheSelectedOrganismAtTheGlobe() async throws {
        let world = try await GPUGlobeTests.hosted(stream: GPUGlobeTests.planetURL(), size: CGSize(width: 1100, height: 700))
        defer { world.finish() }
        let model = world.model
        model.display.minimap = true
        let frame = try #require(world.play.store.frame)
        // the organism nearest the middle of the view with some height
        var best: (Int, Float)?
        for i in 0..<frame.n where frame.position(i).z > 8 {
            let p = frame.position(i)
            let d = hypot(Torus.wrapDelta(p.x - model.pose.x, size: model.worldSize), Torus.wrapDelta(p.y - model.pose.y, size: model.worldSize))
            if best == nil || d < best!.1 { best = (i, d) }
        }
        let pick = try #require(best, "an organism above 8 units")
        model.selectOrganism(frame.uid[pick.0], follow: false, camera: false)
        let target = frame.position(pick.0), distance = 0.2 * model.worldSize
        model.editPose { $0.distance = distance; $0.x = target.x - 0.04 * model.worldSize; $0.y = target.y }
        world.settle()
        let projector = model.projector(size: model.freeRect.size)
        #expect(projector.isMorphing)
        let p = frame.position(pick.0)
        let foot = try #require(projector.screen(SIMD3(p.x, p.y, 0)))
        let top = try #require(projector.screen(p))
        print("pin: foot \(foot) top \(top) z \(p.z)")
        #expect(hypot(top.x - foot.x, top.y - foot.y) >= 3, "a stalk long enough to see")
        _ = try world.composite("globe-pin")
    }

    /// The sheet of the step's report: Home, tilted and turned, four morph frames and the flat close-up after the morph, each with the
    /// engine's colour and the height colour, composited with the overlay. Images only (set `ALIFE_GPU_TESTS_OUT`).
    @Test func rendersTheSheetOfTheGlobeAndTheMorph() async throws {
        let world = try await GPUGlobeTests.hosted(stream: GPUGlobeTests.planetURL(), size: CGSize(width: 1100, height: 700))
        defer { world.finish() }
        let model = world.model
        model.display.minimap = true
        let L = model.worldSize
        let fov = model.pose.fovY, aspect = model.aspect
        let x0 = model.pose.x
        struct Pose { var name: String; var d: Float; var y: Float; var tilt: Float; var heading: Float }
        var poses = [Pose(name: "home", d: Morph.globeDistance(L: L, fovY: fov, aspect: aspect), y: 0.05 * L, tilt: 0, heading: 0),
                     Pose(name: "tilted-turned", d: Morph.globeDistance(L: L, fovY: fov, aspect: aspect), y: 0.1 * L, tilt: 1.0, heading: 0.9)]
        for s: Float in [0.1, 0.3, 0.6, 0.9] {
            poses.append(Pose(name: "morph-s\(Int(s * 10))", d: Morph.distance(forS: s, L: L, fovY: fov, aspect: aspect), y: 0.1 * L, tilt: 0, heading: 0))
        }
        let close = WorldLevels.flyDistance(.region, size: L, fovY: fov, aspect: aspect, globe: true)
        poses.append(Pose(name: "flat-close-up", d: close, y: 0.1 * L, tilt: 0, heading: 0))
        print("region distance \(close) = s \(Morph.s(distance: close, L: L, fovY: fov, aspect: aspect))")
        for colour in ["engine", "height"] {
            model.display.heightColour = colour == "height"
            for pose in poses {
                model.editPose { $0.x = x0; $0.y = pose.y; $0.distance = pose.d; $0.tilt = pose.tilt; $0.heading = pose.heading }
                world.settle(3)
                _ = try world.composite("sheet-\(pose.name)-\(colour)")
            }
        }
    }
}
