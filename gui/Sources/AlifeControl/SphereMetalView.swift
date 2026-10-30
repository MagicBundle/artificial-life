import AlifeCore
import AppKit
import MetalKit
import QuartzCore
import SwiftUI

/// The planet's drawing surface: an `MTKView` that takes the pointer and the keys. Left drag grabs the planet (the point under the pointer
/// stays under it), a right or Option drag turns the heading and tilts the view, a middle drag pans, a pinch and the mouse wheel zoom about
/// the pointer and the trackpad's two-finger scroll pans (View > Scroll zooms restores zoom); a click picks (a double click flies in), the
/// pointer over something names it, and the keys are `WorldKeys`'s. All go through the `SphereController` (DESIGN §46.8).
@MainActor
final class SphereMTKView: MTKView {
    weak var model: WorldModel?
    var sphere: SphereController? { model?.sphere }
    /// The view owns its renderer: `MTKView.delegate` and `SphereController.renderer` are both weak.
    var renderer: SphereRenderer?
    private var link: CADisplayLink?
    private var downPoint: CGPoint?
    private var dragging = false
    private var pressed = HUDTarget.map
    private var tracking: NSTrackingArea?
    private var keyMonitor: Any?

    static let clickSlop: CGFloat = 4

    override var acceptsFirstResponder: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        window?.makeFirstResponder(self)
        model?.observeWindow(window)
        removeKeyMonitor()
        if window == nil {
            stopAnimating()
        } else {
            installKeyMonitor()
            if sphere?.isAnimating == true { startAnimating() }
        }
    }

    private func installKeyMonitor() {
        guard keyMonitor == nil else { return }
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            nonisolated(unsafe) let key = event
            let handled = MainActor.assumeIsolated {
                guard let self, key.window != nil, key.window === self.window else { return false }
                return self.handleKey(key)
            }
            return handled ? nil : event
        }
    }

    private func removeKeyMonitor() {
        if let keyMonitor { NSEvent.removeMonitor(keyMonitor) }
        keyMonitor = nil
    }

    /// Whether `event` was a World key, and has been acted on. Not while a text field is being edited.
    func handleKey(_ event: NSEvent) -> Bool {
        if window?.firstResponder is NSText { return false }
        let flags = event.modifierFlags
        guard let model,
              let action = WorldKeys.action(for: event.charactersIgnoringModifiers ?? "", command: flags.contains(.command),
                                            control: flags.contains(.control))
        else { return false }
        if action == .confirmPower, !model.returnConfirmsPower { return false }
        model.perform(action)
        return true
    }

    private func point(of event: NSEvent) -> CGPoint {
        let inWindow = event.window == nil ? (window?.convertPoint(fromScreen: event.locationInWindow) ?? event.locationInWindow) : event.locationInWindow
        let p = convert(inWindow, from: nil)
        return CGPoint(x: p.x, y: bounds.height - p.y)
    }

    override func keyDown(with event: NSEvent) {
        if !handleKey(event) { super.keyDown(with: event) }
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let tracking { removeTrackingArea(tracking) }
        let area = NSTrackingArea(rect: .zero, options: [.mouseMoved, .mouseEnteredAndExited, .activeInKeyWindow, .inVisibleRect],
                                  owner: self, userInfo: nil)
        addTrackingArea(area)
        tracking = area
    }

    override func mouseMoved(with event: NSEvent) { sphere?.hover(at: point(of: event)) }
    override func mouseExited(with event: NSEvent) { sphere?.hover(at: nil) }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        let here = point(of: event)
        downPoint = here
        dragging = false
        pressed = event.modifierFlags.contains(.option) ? .map : (sphere?.pressTarget(at: here) ?? .map)
        if pressed != .map {
            sphere?.hudPress(pressed, at: here)
            return
        }
        sphere?.hover(at: nil)
        sphere?.beginGrab(at: here)
    }

    override func mouseDragged(with event: NSEvent) {
        if !dragging {
            guard let start = downPoint else { return }
            let here = point(of: event)
            guard hypot(here.x - start.x, here.y - start.y) >= Self.clickSlop else { return }
            dragging = true
        }
        if pressed != .map {
            sphere?.hudDrag(pressed, to: point(of: event))
            return
        }
        if event.modifierFlags.contains(.option) {
            sphere?.drag(dx: Float(event.deltaX), dy: Float(event.deltaY), orbit: true)
        } else {
            sphere?.grab(to: point(of: event))
        }
    }

    override func mouseUp(with event: NSEvent) {
        let target = pressed
        defer { downPoint = nil; dragging = false; pressed = .map; sphere?.endGrab() }
        if target != .map {
            sphere?.hudRelease(target, at: point(of: event), clicks: max(event.clickCount, 1), dragged: dragging)
            return
        }
        guard !dragging, downPoint != nil, !event.modifierFlags.contains(.option) else { return }
        sphere?.click(at: point(of: event), shift: event.modifierFlags.contains(.shift), clicks: max(event.clickCount, 1))
    }

    override func rightMouseDragged(with event: NSEvent) { sphere?.drag(dx: Float(event.deltaX), dy: Float(event.deltaY), orbit: true) }
    override func otherMouseDragged(with event: NSEvent) { sphere?.drag(dx: Float(event.deltaX), dy: Float(event.deltaY), orbit: false) }

    override func scrollWheel(with event: NSEvent) {
        let law = ScrollEvent(dx: Float(event.scrollingDeltaX), dy: Float(event.scrollingDeltaY), precise: event.hasPreciseScrollingDeltas,
                              phase: WorldMTKView.phase(event.phase), momentum: WorldMTKView.phase(event.momentumPhase), time: event.timestamp)
        sphere?.scroll(law, at: point(of: event))
    }

    override func magnify(with event: NSEvent) { sphere?.magnify(Float(event.magnification), at: point(of: event)) }

    // MARK: Animation

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
        guard let sphere, sphere.advance() else {
            stopAnimating()
            return
        }
    }
}

/// Hosts a `SphereMTKView` and a renderer for it. What the world sends reaches the renderer through the controller
/// (`WorldModel.deliver`), not through SwiftUI state.
struct SphereMetalView: NSViewRepresentable {
    @ObservedObject var model: WorldModel

    func makeNSView(context: Context) -> SphereMTKView {
        let view = SphereMTKView(frame: .zero, device: nil)
        view.model = model
        view.colorPixelFormat = SphereRenderer.colorFormat
        view.depthStencilPixelFormat = SphereRenderer.depthFormat
        view.sampleCount = SphereRenderer.sampleCount
        view.clearColor = SphereRenderer.background
        view.clearDepth = 1
        view.isPaused = true
        view.enableSetNeedsDisplay = true
        guard let device = MTLCreateSystemDefaultDevice() else {
            let message = "\(SphereRenderer.Failure.noDevice)"
            Task { @MainActor in model.failDrawing(message) }
            return view
        }
        view.device = device
        do {
            let renderer = try SphereRenderer(device: device)
            renderer.failureHandler = { [weak model] message in model?.failDrawing(message) }
            view.delegate = renderer
            view.renderer = renderer
            model.sphere.attach(renderer: renderer, view: view)
            Task { @MainActor in model.failDrawing(nil) }
        } catch {
            let message = "\(error)"
            Task { @MainActor in model.failDrawing(message) }
        }
        return view
    }

    func updateNSView(_ view: SphereMTKView, context: Context) {
        view.needsDisplay = true
    }

    static func dismantleNSView(_ view: SphereMTKView, coordinator: ()) {
        view.stopAnimating()
        view.model?.sphere.detach()
    }
}
