@testable import AlifeCore
import AppKit
import Foundation
import Metal
import MetalKit
import SwiftUI
import Testing

@testable import AlifeControl

/// The Metal view of the World window owns its renderer. `MTKView.delegate` is a weak reference and so is `WorldModel.renderer`:
/// when `WorldMetalView.makeNSView` kept nothing, the renderer was freed as soon as it returned, nothing was ever drawn into the
/// window (no map, no organisms: the SwiftUI overlay and the clear colour were all of the picture) and no test noticed, because
/// every rendering test holds the `Renderer` in a local. This one goes through the hosted `WorldView`, as the app does.
@MainActor
@Suite(.serialized) struct WorldRendererOwnershipTests {
    @Test func theHostedWorldViewKeepsItsRendererAndDrawsTheOrganisms() async throws {
        let scratch = try Scratch()
        let globe = UserDefaults.standard.object(forKey: WorldModel.globeKey)
        UserDefaults.standard.set(false, forKey: WorldModel.globeKey)  // (a test of the flat map's draw; the app opens on the globe)
        defer {
            if let globe { UserDefaults.standard.set(globe, forKey: WorldModel.globeKey) } else { UserDefaults.standard.removeObject(forKey: WorldModel.globeKey) }
            scratch.cleanup()
        }
        let commands = scratch.url.appendingPathComponent("commands.txt")
        let script = try scratch.script("alife-live", "cat \"$1\"\nwhile IFS= read -r line; do echo \"$line\" >> \"$2\"; done")
        let play = PlayController()
        play.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url,
                   arguments: [Repo.playFixtureURL("3d").path, commands.path])
        let size = CGSize(width: 1100, height: 700)
        let host = NSHostingView(rootView: WorldView(play: play))
        let window = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless], backing: .buffered, defer: false)
        window.contentView = host
        host.frame = NSRect(origin: .zero, size: size)
        window.alphaValue = 0
        window.ignoresMouseEvents = true
        window.setFrameOrigin(NSPoint(x: -20000, y: -20000))
        window.orderFrontRegardless()
        defer { window.orderOut(nil); play.shutdown() }

        func find(_ view: NSView) -> WorldMTKView? {
            if let hit = view as? WorldMTKView { return hit }
            for sub in view.subviews { if let hit = find(sub) { return hit } }
            return nil
        }
        func settle() {
            for _ in 0..<8 {
                host.layoutSubtreeIfNeeded()
                window.displayIfNeeded()
                RunLoop.current.run(until: Date().addingTimeInterval(0.02))
            }
        }
        #expect(await eventually { settle(); return play.store.header != nil && play.store.frame != nil && find(host) != nil })
        settle()
        let metal = try #require(find(host))
        let model = try #require(metal.model)
        // (after a turn of the run loop: the view is made, and the renderer would have been freed by now)
        #expect(metal.delegate is Renderer, "the Metal view has no renderer: nothing is drawn into the window")
        #expect(model.renderer != nil, "the model's renderer was freed")
        guard let renderer = model.renderer else { return }
        let image = try renderer.renderOffscreen(width: 400, height: 300)
        let clear = Renderer.background
        let bgra: [UInt8] = [UInt8((clear.blue * 255).rounded()), UInt8((clear.green * 255).rounded()), UInt8((clear.red * 255).rounded())]
        var other = 0
        for i in stride(from: 0, to: image.bgra.count, by: 4)
        where abs(Int(image.bgra[i]) - Int(bgra[0])) > 3 || abs(Int(image.bgra[i + 1]) - Int(bgra[1])) > 3 || abs(Int(image.bgra[i + 2]) - Int(bgra[2])) > 3 {
            other += 1
        }
        #expect(other > 400, "the picture is the clear colour: \(other) other pixels")
    }
}
