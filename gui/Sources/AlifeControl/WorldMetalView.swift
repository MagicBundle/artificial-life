import AlifeCore
import AppKit
import MetalKit
import QuartzCore
import SwiftUI

/// The World's drawing surface: an `MTKView` that takes the pointer. Left drag pans (the map follows the pointer), a right
/// or Option drag turns the view about its target, the wheel, the trackpad and a pinch zoom; all go through the
/// `WorldModel`. The keys, picking and follow are added with the interaction (DESIGN §33.4.8).
@MainActor
final class WorldMTKView: MTKView {
    weak var model: WorldModel?
    private var link: CADisplayLink?

    override var acceptsFirstResponder: Bool { true }

    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        window?.makeFirstResponder(self)
        if window == nil {
            stopAnimating()
        } else if model?.isAnimating == true {
            startAnimating()  // a fade began before the view had a window to run a display link in
        }
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
    }

    override func mouseDragged(with event: NSEvent) {
        model?.drag(dx: Float(event.deltaX), dy: Float(event.deltaY), orbit: event.modifierFlags.contains(.option))
    }

    override func rightMouseDragged(with event: NSEvent) {
        model?.drag(dx: Float(event.deltaX), dy: Float(event.deltaY), orbit: true)
    }

    override func otherMouseDragged(with event: NSEvent) {
        model?.drag(dx: Float(event.deltaX), dy: Float(event.deltaY), orbit: false)
    }

    override func scrollWheel(with event: NSEvent) {
        model?.scroll(delta: Float(event.scrollingDeltaY), precise: event.hasPreciseScrollingDeltas)
    }

    override func magnify(with event: NSEvent) {
        model?.magnify(Float(event.magnification))
    }

    // MARK: Animation

    /// Runs the display link while something animates (the map fade) and stops it when nothing does: an idle scene costs
    /// nothing.
    func startAnimating() {
        guard link == nil, window != nil else { return }
        let link = displayLink(target: self, selector: #selector(animationTick(_:)))
        link.add(to: .main, forMode: .common)
        self.link = link
    }

    func stopAnimating() {
        link?.invalidate()
        link = nil
    }

    @objc private func animationTick(_ link: CADisplayLink) {
        guard let model, model.advance() else {
            stopAnimating()
            return
        }
    }
}

/// Hosts a `WorldMTKView` and a renderer for it. What the world sends reaches the renderer through the model
/// (`WorldModel.deliver`), not through SwiftUI state; SwiftUI only creates the view and keeps its options current.
struct WorldMetalView: NSViewRepresentable {
    @ObservedObject var model: WorldModel

    func makeNSView(context: Context) -> WorldMTKView {
        let view = WorldMTKView(frame: .zero, device: nil)
        view.model = model
        view.colorPixelFormat = Renderer.colorFormat
        view.depthStencilPixelFormat = Renderer.depthFormat
        view.clearColor = Renderer.background
        view.clearDepth = 1
        view.isPaused = true
        view.enableSetNeedsDisplay = true
        guard let device = MTLCreateSystemDefaultDevice() else {
            let message = "\(Renderer.Failure.noDevice)"
            Task { @MainActor in model.failDrawing(message) }  // not while SwiftUI is updating
            return view
        }
        view.device = device
        do {
            let renderer = try Renderer(device: device)
            renderer.failureHandler = { [weak model] message in model?.failDrawing(message) }
            view.delegate = renderer
            model.attach(renderer: renderer, view: view)
            Task { @MainActor in model.failDrawing(nil) }  // an earlier failure is over
        } catch {
            let message = "\(error)"
            Task { @MainActor in model.failDrawing(message) }
        }
        return view
    }

    func updateNSView(_ view: WorldMTKView, context: Context) {
        view.needsDisplay = true
    }
}
