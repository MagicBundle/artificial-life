import AlifeCore
import AppKit
import Foundation
import ImageIO
import Metal
import SwiftUI
import Testing
import UniformTypeIdentifiers
import simd

@testable import AlifeControl

/// The whole `WorldView` hosted in an `NSHostingView` inside an invisible window far off every screen, over a played stream, as the
/// app has it: the renderer is the one the Metal view owns (`renderer()` fails if the view has none), so a test through it cannot pass
/// with a renderer that the app would have freed (I61, where every draw test held the `Renderer` in a local and none noticed that the
/// window drew nothing). The window is ordered in (SwiftUI's gestures and `Canvas` see nothing in one that is not) and ignores the mouse.
@MainActor
final class HostedWorld {
    let play = PlayController()
    let host: NSHostingView<WorldView>
    let window: NSWindow
    let scratch: Scratch
    var serial = 0
    /// The picture of the newest `composite`, at twice the free rectangle.
    private(set) var lastComposite: CGImage?
    private var saved: [String: Any?] = [:]
    static let keys = ["world.dockSide", "world.dockExtent", "world.dockExtentBottom", "world.dockCollapsed"]
    /// What a test may change in the process's defaults (the hosted window model reads and writes them): kept and put back, and cleared
    /// at the start so that a world opens as a first one does.
    static let stateKeys = [WorldModel.minimapKey, WorldModel.hintKey, WorldModel.heightColourKey, WorldModel.scrollZoomsKey, WorldModel.exposureKey]
    private var savedState: [String: Any?] = [:]

    /// Over the stream at `stream` (a recorded `PlayStream`), in a window of `size`; `dockCollapsed` hides the control center, so that
    /// the map has the whole window. The dock is on the right, open, unless said otherwise (nil keeps the player's setting).
    init(stream: URL, size: CGSize = CGSize(width: 1100, height: 700), dockCollapsed: Bool? = false, timeout: Duration = .seconds(20)) async throws {
        scratch = try Scratch()
        for key in Self.stateKeys {
            savedState[key] = UserDefaults.standard.object(forKey: key)
            UserDefaults.standard.removeObject(forKey: key)
        }
        if let dockCollapsed {
            for key in Self.keys { saved[key] = UserDefaults.standard.object(forKey: key) }
            UserDefaults.standard.set(DockSide.right.rawValue, forKey: "world.dockSide")
            UserDefaults.standard.set(dockCollapsed, forKey: "world.dockCollapsed")
            UserDefaults.standard.removeObject(forKey: "world.dockExtent")
            UserDefaults.standard.removeObject(forKey: "world.dockExtentBottom")
        }
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
        let ready = await eventually(timeout: timeout) {
            settle()
            return play.store.header != nil && play.store.frame != nil && play.store.map != nil && metal != nil
        }
        #expect(ready, "the hosted world did not come up")
        // The minimap is over a corner of the picture: a test of the map alone has it off, and one of the minimap turns it on.
        if let model = metal?.model { model.display.minimap = false }
        settle()
    }

    convenience init(kind: String = "3d") async throws {
        try await self.init(stream: Repo.playFixtureURL(kind))
    }

    func finish() {
        window.orderOut(nil)
        play.shutdown()
        for key in Self.keys where saved.keys.contains(key) {
            if let value = saved[key] ?? nil { UserDefaults.standard.set(value, forKey: key) } else { UserDefaults.standard.removeObject(forKey: key) }
        }
        for key in Self.stateKeys {
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

    static var outputDirectory: URL {
        let base = ProcessInfo.processInfo.environment["ALIFE_GPU_TESTS_OUT"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.temporaryDirectory.appendingPathComponent("alife-gpu-tests", isDirectory: true)
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    /// The picture at the size of the free rectangle (the aspect the model fitted the view to), by the hosted view's renderer.
    func render(_ name: String) throws -> OffscreenImage {
        let free = model.freeRect
        let image = try renderer().renderOffscreen(width: Int(free.width), height: Int(free.height))
        try PNG.write(image, to: Self.outputDirectory.appendingPathComponent("\(name).png"))
        return image
    }

    /// The picture the window shows: the hosted renderer's drawing at twice the free rectangle (a Retina drawable), with the overlay and
    /// the controls (the same `Canvas` views the window has, over the same model) drawn over it. Saved as a PNG; returns the drawing
    /// of the renderer alone.
    @discardableResult
    func composite(_ name: String) throws -> OffscreenImage {
        lastComposite = nil
        let free = model.freeRect
        let scale = 2
        let image = try renderer().renderOffscreen(width: Int(free.width) * scale, height: Int(free.height) * scale)
        let map = Self.cgImage(image)
        let overlays = ZStack {
            Image(decorative: map, scale: CGFloat(scale))
            WorldOverlayView(model: model, overlay: model.overlay, play: play)
            WorldHUDView(model: model, overlay: model.overlay, play: play)
        }
        .frame(width: free.width, height: free.height)
        let rendered = ImageRenderer(content: overlays)
        rendered.scale = CGFloat(scale)
        guard let out = rendered.cgImage else { throw Renderer.Failure.encoding }
        lastComposite = out
        let url = Self.outputDirectory.appendingPathComponent("\(name).png")
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw Renderer.Failure.encoding
        }
        CGImageDestinationAddImage(destination, out, nil)
        guard CGImageDestinationFinalize(destination) else { throw Renderer.Failure.encoding }
        return image
    }

    /// The colour of the pixel of `image` at (`x`, `y`) in its own pixels, as 0...255 values.
    static func rgb(of image: CGImage, _ x: Int, _ y: Int) -> SIMD3<Float> {
        var bytes = [UInt8](repeating: 0, count: 4)
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        bytes.withUnsafeMutableBytes { buffer in
            let context = CGContext(data: buffer.baseAddress, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4, space: space,
                                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            context.draw(image, in: CGRect(x: -x, y: -(image.height - 1 - y), width: image.width, height: image.height))
        }
        return SIMD3(Float(bytes[0]), Float(bytes[1]), Float(bytes[2]))
    }

    static func cgImage(_ image: OffscreenImage) -> CGImage {
        let data = Data(image.bgra)
        let provider = CGDataProvider(data: data as CFData)!
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        return CGImage(width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 4 * image.width,
                       space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: info, provider: provider, decode: nil,
                       shouldInterpolate: false, intent: .defaultIntent)!
    }
}
