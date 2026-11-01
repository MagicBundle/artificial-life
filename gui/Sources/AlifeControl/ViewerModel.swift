import AlifeCore
import AppKit
import Foundation
import MetalKit

/// What the 3D viewer window does that is not the stream's: the display options, the camera control, the snapshot, and
/// the mapping from keys to actions. The stream's own commands go to the `StreamController`.
@MainActor
final class ViewerModel: ObservableObject {
    let stream: StreamController
    @Published var display = ViewerDisplay()
    @Published var cameraMode = CameraMode.orbit
    /// The last thing worth telling the user: where a snapshot went, or why it failed.
    @Published private(set) var note: String?
    /// Why nothing can be drawn (no Metal device, shaders that do not compile, a draw that failed); nil while it works.
    @Published private(set) var drawingFailure: String?

    /// Set by the view that draws; weak because the view owns the renderer and the window owns the view.
    weak var renderer: Renderer?
    weak var view: MTKView?

    /// The planet's camera and renderer feed, used instead of the box's while the stream is a sphere (DESIGN §46.8).
    let sphereLab = SphereLabModel()
    var isSphere: Bool { stream.header?.isSphere == true }

    /// What the colour pop-up shows, apart from the stream, which publishes with every frame (see `DistinctState`).
    let colour: DistinctState<ViewerColourInputs>

    init(stream: StreamController) {
        self.stream = stream
        colour = DistinctState(initial: ViewerColourInputs(stream: stream), sources: [stream.objectWillChange]) { [weak stream] in
            stream.map { ViewerColourInputs(stream: $0) }
        }
    }

    /// The view's width over its height (1 before there is a view).
    var aspect: Float {
        guard let view, view.bounds.height > 0 else { return 1 }
        return Float(view.bounds.width / view.bounds.height)
    }

    func failDrawing(_ message: String?) { drawingFailure = message }

    // MARK: Actions

    func perform(_ action: ViewerAction) {
        switch action {
        case .togglePause: stream.togglePause()
        case .step: stream.step()
        case .faster: stream.faster()
        case .slower: stream.slower()
        case .cycleColor: stream.cycleColor()
        case .toggleFields: stream.setFields(!stream.control.fields)
        case .fieldLayerDown: stream.setFieldLayer(stream.control.fieldLayer - 1)
        case .fieldLayerUp: stream.setFieldLayer(stream.control.fieldLayer + 1)
        case .snapshot: snapshot()
        case .toggleAgents: display.agents.toggle()
        case .toggleBonds: display.bonds.toggle()
        case .toggleFrame: display.frame.toggle()
        case .toggleCameraMode: cameraMode = cameraMode == .orbit ? .fly : .orbit
        case .resetCamera: resetCamera()
        }
    }

    func resetCamera() {
        if isSphere {
            sphereLab.reset()
            return
        }
        renderer?.resetCamera(aspect: aspect)
        view?.needsDisplay = true
    }

    // MARK: Pointer and keys

    func drag(dx: Float, dy: Float, pan: Bool) {
        guard let renderer else { return }
        renderer.camera.drag(dx: dx, dy: dy, mode: cameraMode, pan: pan,
                             viewportHeight: Float(view?.bounds.height ?? 1))
        view?.needsDisplay = true
    }

    func scroll(delta: Float, precise: Bool) {
        zoom(by: ViewerInput.dollyFactor(scroll: delta, precise: precise))
    }

    func magnify(_ magnification: Float) {
        zoom(by: ViewerInput.dollyFactor(magnification: magnification))
    }

    private func zoom(by factor: Float) {
        guard let renderer else { return }
        renderer.camera.zoom(by: factor, mode: cameraMode, reach: worldSize / 2)
        view?.needsDisplay = true
    }

    func press(_ key: ArrowKey, fast: Bool) {
        guard let renderer else { return }
        renderer.camera.press(key, mode: cameraMode, step: ViewerInput.flyStep(size: worldSize, fast: fast))
        view?.needsDisplay = true
    }

    private var worldSize: Float { Float(stream.header?.size ?? 1) }

    // MARK: Snapshot

    /// Draws the current view into an image of the window's size and writes it to `runs/NAME/media/view3d_<step>.png`.
    func snapshot() {
        if isSphere, let view = sphereLab.view, let run = stream.source?.run, let directory = stream.directory, let frame = stream.frame {
            let size = view.drawableSize
            let url = ViewerSnapshot.url(directory: directory, run: run, step: frame.step)
            do {
                guard let image = try sphereLab.snapshot(width: Int(size.width), height: Int(size.height)) else { return }
                try PNG.write(image, to: url)
                note = "Saved runs/\(run)/media/\(url.lastPathComponent)"
            } catch {
                note = "Snapshot failed: \(error)"
            }
            return
        }
        guard let renderer, let view, let step = renderer.frameStep, let run = stream.source?.run,
              let directory = stream.directory else {
            note = "Nothing to save yet."
            return
        }
        let size = view.drawableSize
        let url = ViewerSnapshot.url(directory: directory, run: run, step: step)
        do {
            let image = try renderer.renderOffscreen(width: Int(size.width), height: Int(size.height))
            try PNG.write(image, to: url)
            note = "Saved runs/\(run)/media/\(url.lastPathComponent)"
        } catch {
            note = "Snapshot failed: \(error)"
        }
    }
}
