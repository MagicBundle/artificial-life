import AlifeCore
import AppKit
import MetalKit
import SwiftUI

/// What the viewer's colour pop-up shows.
struct ViewerColourInputs: Equatable {
    var modes: [String]
    var current: String
    var live: Bool

    @MainActor
    init(stream: StreamController) {
        modes = stream.header?.colorModes ?? [stream.control.color]
        current = stream.control.color
        live = stream.phase == .live
    }
}

/// What the controls of the research Watch show of the stream, held apart from it (it publishes with every frame; a `Menu` that observes it
/// is rebuilt under the pointer: `DistinctState`).
struct ViewerControlInputs: Equatable {
    var hasSource: Bool
    var follows: Bool
    var live: Bool
    var paused: Bool
    var hasFrame: Bool
    var fields: Bool
    var fieldLayers: Int
    var fieldLayer: Int
    var every: Int

    @MainActor
    init(stream: StreamController) {
        hasSource = stream.source != nil
        follows = stream.source?.follow == true
        live = stream.phase == .live
        paused = stream.control.paused
        hasFrame = stream.frame != nil
        fields = stream.control.fields
        fieldLayers = stream.header?.fieldLayers ?? 0
        fieldLayer = stream.control.fieldLayer
        every = stream.control.every
    }
}

/// What the research Watch's page body reads of the stream: which picture to make and what to say while there is none.
struct ViewerPageInputs: Equatable {
    var phase: StreamPhase
    var isSphere: Bool
    var hasSource: Bool
    var follows: Bool

    @MainActor
    init(stream: StreamController) {
        phase = stream.phase
        isSphere = stream.header?.isSphere == true
        hasSource = stream.source != nil
        follows = stream.source?.follow == true
    }
}

/// The colour pop-up of the viewer. It reads `state`, not the stream: the stream publishes with every frame, and a pop-up that is
/// evaluated again while its menu is open is rebuilt under the pointer.
struct ViewerColourPicker: View {
    @ObservedObject var state: DistinctState<ViewerColourInputs>
    let stream: StreamController

    var body: some View {
        let _ = ViewProbe.note("ViewerColourPicker", for: stream)
        let inputs = state.value
        Picker("Colour", selection: Binding(get: { inputs.current }, set: { stream.setColor($0) })) {
            ForEach(inputs.modes, id: \.self) { Text($0) }
        }
        .frame(width: 150)
        .disabled(!inputs.live)
        .help("Colour mode (c)")
    }
}

// MARK: - The Metal view

/// Hosts a `ViewerMTKView` and feeds it from the stream. SwiftUI calls `updateNSView` when the stream, the display
/// options or the window change, which is when something may need drawing.
struct ViewerMetalView: NSViewRepresentable {
    @ObservedObject var stream: StreamController
    @ObservedObject var model: ViewerModel
    /// The keys follow the window, not the focus (the `alife` window's Watch).
    var keysFollowWindow = false

    /// What the renderer has been given, so that an update copies a frame only once.
    final class Coordinator {
        var renderer: Renderer?
        var header: StreamHeader?
        var serial = -1
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> ViewerMTKView {
        let view = ViewerMTKView(frame: .zero, device: nil)
        view.keysFollowWindow = keysFollowWindow
        view.model = model
        view.colorPixelFormat = Renderer.colorFormat
        view.depthStencilPixelFormat = Renderer.depthFormat
        view.clearColor = Renderer.background
        view.clearDepth = 1
        view.isPaused = true
        view.enableSetNeedsDisplay = true
        model.view = view
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
            context.coordinator.renderer = renderer
            model.renderer = renderer
            Task { @MainActor in model.failDrawing(nil) }  // an earlier failure is over
        } catch {
            let message = "\(error)"
            Task { @MainActor in model.failDrawing(message) }
        }
        return view
    }

    func updateNSView(_ view: ViewerMTKView, context: Context) {
        let coordinator = context.coordinator
        guard let renderer = coordinator.renderer else { return }
        // SwiftUI calls this for every change of the stream's or the model's state (the speed readout, the status line),
        // most of which change nothing that is drawn: the view is redrawn only when its content did. The camera and the
        // window's size ask for their own redraws.
        var changed = false
        if renderer.display != model.display {
            renderer.display = model.display
            changed = true
        }
        if stream.header != coordinator.header {
            coordinator.header = stream.header
            coordinator.serial = -1
            renderer.configure(header: stream.header, aspect: model.aspect)
            changed = true
        }
        if stream.frameSerial != coordinator.serial {
            coordinator.serial = stream.frameSerial
            renderer.show(frame: stream.frame)
            changed = true
        }
        if changed { view.needsDisplay = true }
    }
}
