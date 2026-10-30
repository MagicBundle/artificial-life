import AlifeCore
import AppKit
import MetalKit
import SwiftUI
import simd

/// The Lab's 3D viewer for a planet (DESIGN §46.8): a protocol 2 `alife stream` shows the shell of a sphere world with the same renderer as the
/// World window, a camera that orbits the planet (drag: the planet turns under the pointer; scroll or pinch: zoom; right drag: tilt and turn;
/// double click or `r`: the whole planet) and the field layer the stream sends, as the floor's colours. The box world's viewer is untouched.
@MainActor
final class SphereLabModel {
    private(set) var camera: SphereCamera?
    weak var renderer: SphereRenderer?
    weak var view: MTKView?
    private var grabAnchor: SIMD3<Float>?
    private var sphere: SphereHeader?
    var display = WorldDisplay()

    var aspect: Float {
        guard let view, view.bounds.height > 0 else { return 1 }
        return Float(view.bounds.width / view.bounds.height)
    }

    func configure(header: StreamHeader?, display: ViewerDisplay) {
        guard let sphere = header?.sphere else {
            self.sphere = nil
            camera = nil
            renderer?.configure(sphere: nil, bodyRadius: 1, camera: nil)
            return
        }
        self.sphere = sphere
        var shown = WorldDisplay()
        shown.agents = display.agents
        shown.bonds = display.bonds
        shown.frame = display.frame
        shown.territory = .off
        self.display = shown
        let pose = SphereCamera.home(planetRadius: Float(sphere.planetRadius), lidRadius: Float(sphere.lidRadius), centre: SIMD3(1, 0, 0), aspect: aspect)
        let camera = SphereCamera(pose: pose, planetRadius: Float(sphere.planetRadius), lidRadius: Float(sphere.lidRadius))
        self.camera = camera
        renderer?.configure(sphere: sphere, bodyRadius: Float(header?.radius ?? 1), camera: camera)
        let centres = SphereCentres.shared.centres(nside: sphere.nsideField)
        renderer?.setFloor(SphereColours.floor(sphere: sphere, map: nil, material: nil, soil: nil, display: shown, layer: nil, centres: centres))
        renderer?.setCloud([SIMD4<Float>](repeating: .zero, count: sphere.fieldCells))
        renderer?.setTerritory([SIMD4<Float>](repeating: .zero, count: sphere.territoryCells))
        push()
    }

    /// The frame of the stream: the organisms, and (when the stream sends the field) the floor in the colours of the layer.
    func show(frame: StreamFrame?, header: StreamHeader?) {
        renderer?.show(streamFrame: frame)
        if let sphere, let field = frame?.field, field.count == sphere.fieldCells {
            let colours = field.map { c in SIMD4<Float>(Float((c >> 16) & 255) / 255, Float((c >> 8) & 255) / 255, Float(c & 255) / 255, 1) }
            renderer?.setFloor(colours)
        } else if let sphere, frame?.field == nil {
            let centres = SphereCentres.shared.centres(nside: sphere.nsideField)
            renderer?.setFloor(SphereColours.floor(sphere: sphere, map: nil, material: nil, soil: nil, display: display, layer: nil, centres: centres))
        }
        view?.needsDisplay = true
    }

    func push() {
        guard let camera, let renderer else { return }
        var draw = SphereDraw(camera: camera)
        draw.display = display
        draw.pixelsPerPoint = Float(view?.window?.backingScaleFactor ?? 2)
        renderer.setDraw(draw)
        view?.needsDisplay = true
    }

    func reset() {
        guard let sphere else { return }
        let pose = SphereCamera.home(planetRadius: Float(sphere.planetRadius), lidRadius: Float(sphere.lidRadius), centre: camera?.pose.up ?? SIMD3(1, 0, 0), aspect: aspect)
        camera?.pose = pose
        push()
    }

    private func edit(_ change: (inout SphereCamera) -> Void) {
        guard var next = camera else { return }
        change(&next)
        next.clamp(aspect: aspect)
        camera = next
        push()
    }

    private func ndc(_ point: CGPoint) -> SIMD2<Float> {
        let size = view?.bounds.size ?? CGSize(width: 1, height: 1)
        return SpherePicker.ndc(of: point, in: size)
    }

    func beginGrab(at point: CGPoint) {
        grabAnchor = camera?.grabDirection(ndc: ndc(point), aspect: aspect)
    }

    func grab(to point: CGPoint) {
        guard let anchor = grabAnchor else { return }
        let a = aspect, n = ndc(point)
        edit { $0.grab(anchor: anchor, at: n, aspect: a) }
    }

    func endGrab() { grabAnchor = nil }

    func turn(dx: Float, dy: Float) { edit { $0.turn(dx: dx, dy: dy) } }

    func zoom(by factor: Float, at point: CGPoint?) {
        let a = aspect, n = point.map(ndc)
        edit { $0.zoom(by: factor, at: n, aspect: a) }
    }

    func snapshot(width: Int, height: Int) throws -> OffscreenImage? {
        try renderer?.renderOffscreen(width: width, height: height)
    }
}

/// The planet's surface in the Lab: an `MTKView` that takes the mouse.
@MainActor
final class SphereLabMTKView: MTKView {
    weak var viewer: ViewerModel?
    var renderer: SphereRenderer?
    private var downPoint: CGPoint?
    private var dragging = false

    override var acceptsFirstResponder: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        window?.makeFirstResponder(self)
    }

    private func point(of event: NSEvent) -> CGPoint {
        let p = convert(event.locationInWindow, from: nil)
        return CGPoint(x: p.x, y: bounds.height - p.y)
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        downPoint = point(of: event)
        dragging = false
        if event.clickCount >= 2 { viewer?.resetCamera(); return }
        viewer?.sphereLab.beginGrab(at: point(of: event))
    }

    override func mouseDragged(with event: NSEvent) {
        if event.modifierFlags.contains(.option) {
            viewer?.sphereLab.turn(dx: Float(event.deltaX), dy: Float(event.deltaY))
        } else {
            viewer?.sphereLab.grab(to: point(of: event))
        }
    }

    override func mouseUp(with event: NSEvent) { viewer?.sphereLab.endGrab() }
    override func rightMouseDragged(with event: NSEvent) { viewer?.sphereLab.turn(dx: Float(event.deltaX), dy: Float(event.deltaY)) }
    override func otherMouseDragged(with event: NSEvent) { viewer?.sphereLab.turn(dx: Float(event.deltaX), dy: Float(event.deltaY)) }

    override func scrollWheel(with event: NSEvent) {
        let factor = ViewerInput.dollyFactor(scroll: Float(event.scrollingDeltaY), precise: event.hasPreciseScrollingDeltas)
        viewer?.sphereLab.zoom(by: factor, at: point(of: event))
    }

    override func magnify(with event: NSEvent) {
        viewer?.sphereLab.zoom(by: ViewerInput.dollyFactor(magnification: Float(event.magnification)), at: point(of: event))
    }

    override func keyDown(with event: NSEvent) {
        if event.modifierFlags.contains(.command) {
            super.keyDown(with: event)
            return
        }
        if let characters = event.charactersIgnoringModifiers, let action = ViewerKeys.action(for: characters) { viewer?.perform(action) }
    }
}

/// Hosts the planet's view and feeds it from the stream, like `ViewerMetalView` does for a box.
struct SphereLabMetalView: NSViewRepresentable {
    @ObservedObject var stream: StreamController
    @ObservedObject var model: ViewerModel

    final class Coordinator {
        var renderer: SphereRenderer?
        var header: StreamHeader?
        var serial = -1
        var display = ViewerDisplay()
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> SphereLabMTKView {
        let view = SphereLabMTKView(frame: .zero, device: nil)
        view.viewer = model
        view.colorPixelFormat = SphereRenderer.colorFormat
        view.depthStencilPixelFormat = SphereRenderer.depthFormat
        view.sampleCount = SphereRenderer.sampleCount
        view.clearColor = SphereRenderer.background
        view.clearDepth = 1
        view.isPaused = true
        view.enableSetNeedsDisplay = true
        model.sphereLab.view = view
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
            context.coordinator.renderer = renderer
            model.sphereLab.renderer = renderer
            Task { @MainActor in model.failDrawing(nil) }
        } catch {
            let message = "\(error)"
            Task { @MainActor in model.failDrawing(message) }
        }
        return view
    }

    func updateNSView(_ view: SphereLabMTKView, context: Context) {
        let c = context.coordinator
        guard c.renderer != nil else { return }
        var changed = false
        if stream.header != c.header || model.display != c.display {
            let headerChanged = stream.header != c.header
            c.header = stream.header
            c.display = model.display
            c.serial = -1
            if headerChanged {
                model.sphereLab.configure(header: stream.header, display: model.display)
            } else {
                model.sphereLab.display.agents = model.display.agents
                model.sphereLab.display.bonds = model.display.bonds
                model.sphereLab.display.frame = model.display.frame
                model.sphereLab.push()
            }
            changed = true
        }
        if stream.frameSerial != c.serial {
            c.serial = stream.frameSerial
            model.sphereLab.show(frame: stream.frame, header: stream.header)
            changed = true
        }
        if changed { view.needsDisplay = true }
    }
}
