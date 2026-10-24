import AlifeCore
import AppKit
import MetalKit
import QuartzCore
import SwiftUI

/// The World's drawing surface: an `MTKView` that takes the pointer and the keys. Left drag grabs the map (the point under the pointer
/// stays under it), a right or Option drag turns the heading and tilts the view, a middle drag pans, a pinch and the mouse wheel zoom
/// about the pointer and the trackpad's two-finger scroll pans (View > Scroll zooms restores zoom); a click picks (a double click
/// flies in), the pointer over something names it, and the keys are `WorldKeys`'s. All go through the `WorldModel` (DESIGN
/// §33.4.8, §36.5).
@MainActor
final class WorldMTKView: MTKView {
    weak var model: WorldModel?
    /// The view owns its renderer: `MTKView.delegate` and `WorldModel.renderer` are both weak, so without this the
    /// renderer was freed as soon as `makeNSView` returned and the world drew nothing (DESIGN §33.4.4, I61).
    var renderer: Renderer?
    private var link: CADisplayLink?
    /// Where the left button went down (view points, origin at the top left), and whether the pointer has moved far enough
    /// since for the gesture to be a drag and not a click.
    private var downPoint: CGPoint?
    private var dragging = false
    private var tracking: NSTrackingArea?
    private var keyMonitor: Any?

    /// A press that moves less than this many points is a click.
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
            if model?.isAnimating == true {
                startAnimating()  // a fade began before the view had a window to run a display link in
            }
        }
    }

    /// The window's keys reach the world whichever view has the focus. After a click in the Cast or the Chronicle the focus is
    /// in the list, and space or esc would do nothing: a local monitor sees the key first. Text being edited keeps its keys.
    private func installKeyMonitor() {
        guard keyMonitor == nil else { return }
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            // (AppKit calls the monitor on the main thread; NSEvent is not Sendable, so it is carried across by hand.)
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
        // Return confirms a previewed power while the Powers tab shows it; otherwise it is not ours (a focused button may take it)
        if action == .confirmPower, !model.returnConfirmsPower { return false }
        model.perform(action)
        return true
    }

    /// The event's location in the view's points with the origin at the top left (what the picker and the overlay use).
    private func point(of event: NSEvent) -> CGPoint {
        let p = convert(event.locationInWindow, from: nil)
        return CGPoint(x: p.x, y: bounds.height - p.y)
    }

    // MARK: Keys

    override func keyDown(with event: NSEvent) {
        if !handleKey(event) { super.keyDown(with: event) }
    }

    // MARK: Pointer

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let tracking { removeTrackingArea(tracking) }
        let area = NSTrackingArea(rect: .zero, options: [.mouseMoved, .mouseEnteredAndExited, .activeInKeyWindow, .inVisibleRect],
                                  owner: self, userInfo: nil)
        addTrackingArea(area)
        tracking = area
    }

    override func mouseMoved(with event: NSEvent) {
        model?.hover(at: point(of: event))
    }

    override func mouseExited(with event: NSEvent) {
        model?.hover(at: nil)
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        downPoint = point(of: event)
        dragging = false
        model?.hover(at: nil)
        model?.beginGrab(at: point(of: event))
    }

    override func mouseDragged(with event: NSEvent) {
        if !dragging {
            guard let start = downPoint else { return }
            let here = point(of: event)
            guard hypot(here.x - start.x, here.y - start.y) >= Self.clickSlop else { return }  // still a click
            dragging = true
        }
        if event.modifierFlags.contains(.option) {
            model?.drag(dx: Float(event.deltaX), dy: Float(event.deltaY), orbit: true)
        } else {
            model?.grab(to: point(of: event))
        }
    }

    override func mouseUp(with event: NSEvent) {
        defer { downPoint = nil; dragging = false; model?.endGrab() }
        guard !dragging, downPoint != nil, !event.modifierFlags.contains(.option) else { return }
        model?.click(at: point(of: event), shift: event.modifierFlags.contains(.shift), clicks: max(event.clickCount, 1))
    }

    override func rightMouseDragged(with event: NSEvent) {
        model?.drag(dx: Float(event.deltaX), dy: Float(event.deltaY), orbit: true)
    }

    override func otherMouseDragged(with event: NSEvent) {
        model?.drag(dx: Float(event.deltaX), dy: Float(event.deltaY), orbit: false)
    }

    override func scrollWheel(with event: NSEvent) {
        let law = ScrollEvent(dx: Float(event.scrollingDeltaX), dy: Float(event.scrollingDeltaY), precise: event.hasPreciseScrollingDeltas,
                              phase: Self.phase(event.phase), momentum: Self.phase(event.momentumPhase), time: event.timestamp)
        model?.scroll(law, at: point(of: event))
    }

    override func magnify(with event: NSEvent) {
        model?.magnify(Float(event.magnification), at: point(of: event))
    }

    /// `NSEvent.Phase` as the pure law sees it.
    static func phase(_ phase: NSEvent.Phase) -> ScrollGesturePhase {
        if phase.contains(.began) { return .began }
        if phase.contains(.changed) { return .changed }
        if phase.contains(.ended) { return .ended }
        if phase.contains(.cancelled) { return .cancelled }
        if phase.contains(.mayBegin) { return .mayBegin }
        if phase.contains(.stationary) { return .stationary }
        return .none
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
            view.renderer = renderer
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
