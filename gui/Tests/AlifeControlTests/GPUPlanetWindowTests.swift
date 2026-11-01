import AlifeCore
import AppKit
import Foundation
import ImageIO
import SwiftUI
import Testing
import UniformTypeIdentifiers
import simd

@testable import AlifeControl

/// The whole World window hosted over a played sphere stream, as the app has it (`HostedWorld`'s plan for a planet): the renderer is the one the
/// planet's Metal view owns, so the drawing, the overlay and the controls are what a window would show. Opt in with
/// `ALIFE_GPU_TESTS=1 swift test --filter GPUPlanetWindowTests`; `ALIFE_SPHERE_STREAM` names a protocol 2 play stream (default: the fixture),
/// `ALIFE_GPU_TESTS_OUT` the folder of the pictures.
@MainActor
final class HostedPlanet {
    let play = PlayController()
    let host: NSHostingView<WorldView>
    let window: NSWindow
    let scratch: Scratch
    private var savedState: [String: Any?] = [:]

    init(stream: URL, size: CGSize = CGSize(width: 1280, height: 800)) async throws {
        scratch = try Scratch()
        for key in HostedWorld.stateKeys + HostedWorld.keys {
            savedState[key] = UserDefaults.standard.object(forKey: key)
            UserDefaults.standard.removeObject(forKey: key)
        }
        UserDefaults.standard.set(DockSide.right.rawValue, forKey: "world.dockSide")
        UserDefaults.standard.set(true, forKey: WorldModel.hintKey)  // no first-run hint over the picture
        let script = try scratch.script("alife-live", "cat \"$1\"\nwhile IFS= read -r line; do :; done")
        play.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url, arguments: [stream.path])
        host = NSHostingView(rootView: WorldView(play: play))
        window = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless], backing: .buffered, defer: false)
        window.contentView = host
        host.frame = NSRect(origin: .zero, size: size)
        window.alphaValue = 0
        window.ignoresMouseEvents = true
        window.setFrameOrigin(NSPoint(x: -20000, y: -20000))
        window.orderFrontRegardless()
        let ready = await eventually(timeout: .seconds(30)) {
            settle()
            return play.store.header != nil && play.store.frame != nil && metal != nil && metal?.renderer != nil
        }
        #expect(ready, "the hosted planet did not come up")
        settle()
    }

    func finish() {
        window.orderOut(nil)
        play.shutdown()
        for key in HostedWorld.stateKeys + HostedWorld.keys {
            if let value = savedState[key] ?? nil { UserDefaults.standard.set(value, forKey: key) } else { UserDefaults.standard.removeObject(forKey: key) }
        }
        scratch.cleanup()
    }

    func settle(_ turns: Int = 8) {
        for _ in 0..<turns {
            host.layoutSubtreeIfNeeded()
            window.displayIfNeeded()
            RunLoop.current.run(until: Date().addingTimeInterval(0.02))
        }
    }

    var metal: SphereMTKView? { find(host) }
    private func find(_ view: NSView) -> SphereMTKView? {
        if let hit = view as? SphereMTKView { return hit }
        for sub in view.subviews { if let hit = find(sub) { return hit } }
        return nil
    }
    var model: WorldModel { metal!.model! }

    /// Runs the camera's own motion to its end (the real clock; a flight takes a second or two).
    func finishMotion(seconds: Double = 6) async {
        // (the display link of a window nobody sees does not tick: the camera's own motion is driven by hand)
        let deadline = Date().addingTimeInterval(seconds)
        while Date() < deadline {
            let moving = model.sphere.advance()
            settle(1)
            try? await Task.sleep(for: .milliseconds(16))
            if !moving { break }
        }
        settle()
    }

    /// The picture the window shows over the free rectangle: the renderer's drawing at twice its size with the overlay and the controls drawn
    /// over it (the same `Canvas` views, over the same model). Saved as a PNG.
    @discardableResult
    func composite(_ name: String) throws -> OffscreenImage {
        let free = model.freeRect
        let scale = 2
        let renderer = try #require(metal?.renderer)
        let image = try renderer.renderOffscreen(width: Int(free.width) * scale, height: Int(free.height) * scale)
        let overlays = ZStack {
            Image(decorative: HostedWorld.cgImage(image), scale: CGFloat(scale))
            SphereOverlayView(model: model, overlay: model.overlay, play: play)
        }
        .frame(width: free.width, height: free.height)
        let rendered = ImageRenderer(content: overlays)
        rendered.scale = CGFloat(scale)
        guard let out = rendered.cgImage else { throw SphereRenderer.Failure.encoding }
        let url = HostedWorld.outputDirectory.appendingPathComponent("\(name).png")
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw SphereRenderer.Failure.encoding
        }
        CGImageDestinationAddImage(destination, out, nil)
        guard CGImageDestinationFinalize(destination) else { throw SphereRenderer.Failure.encoding }
        print("GPUPlanetWindowTests: wrote \(url.path)")
        return image
    }
}

@MainActor
@Suite(
    "The planet's window",
    .enabled(
        if: ProcessInfo.processInfo.environment["ALIFE_GPU_TESTS"] == "1",
        "set ALIFE_GPU_TESTS=1 to host the window and render on the GPU")
)
struct GPUPlanetWindowTests {
    private func stream() -> URL {
        ProcessInfo.processInfo.environment["ALIFE_SPHERE_STREAM"].map { URL(fileURLWithPath: $0) }
            ?? Repo.streamData.appendingPathComponent("play_sphere.bin")
    }

    @Test func theWindowHostsThePlanetAndDrawsItWithItsControls() async throws {
        let world = try await HostedPlanet(stream: stream())
        defer { world.finish() }
        #expect(world.metal?.delegate is SphereRenderer, "the Metal view has a renderer")
        #expect(world.model.sphereActive && world.model.renderer == nil, "a planet draws with the planet's renderer alone")
        world.model.sphere.freeRectChanged()
        let free = world.model.freeRect
        #expect(free.width > 600 && free.height > 400)
        let home = try world.composite("planet_window_home_v2")
        #expect(home.width == Int(free.width) * 2)
        // the planet is on screen: the middle of the picture is not the sky
        let mid = home.pixel(home.width / 2, home.height / 2)
        #expect(Int(mid.r) + Int(mid.g) + Int(mid.b) > 60, "the planet is in the middle of the window")
    }

    @Test func theWindowShowsTheStormTheCellAndTheHover() async throws {
        let world = try await HostedPlanet(stream: stream())
        defer { world.finish() }
        let sphere = world.model.sphere
        if let storm = world.play.store.storms?.storms.storms.first {
            world.model.selectStorm(storm.id, fly: true, follow: false)
            await world.finishMotion()
            try world.composite("planet_window_storm_v2")
        }
        // a region view, tilted, with a clicked cell
        world.model.perform(.viewPreset(.oblique))
        await world.finishMotion()
        sphere.click(at: CGPoint(x: world.model.freeRect.midX + 40, y: world.model.freeRect.midY + 30), shift: false, clicks: 1)
        sphere.hover(at: CGPoint(x: world.model.freeRect.midX - 80, y: world.model.freeRect.midY))
        try world.composite("planet_window_cell_v2")
        // the horizon, low
        sphere.zoom(by: 0.35)
        world.model.perform(.viewPreset(.low))
        await world.finishMotion()
        try world.composite("planet_window_horizon_v2")
        // the whole planet again
        world.model.home()
        await world.finishMotion()
        try world.composite("planet_window_home_again_v2")
    }
}
