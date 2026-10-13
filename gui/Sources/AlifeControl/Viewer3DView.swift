import AlifeCore
import AppKit
import MetalKit
import SwiftUI

/// The content of the "3D Viewer" window: the controls, the Metal view, and a status line.
struct Viewer3DView: View {
    @ObservedObject var stream: StreamController
    @StateObject private var model: ViewerModel

    init(stream: StreamController) {
        self.stream = stream
        _model = StateObject(wrappedValue: ViewerModel(stream: stream))
    }

    var body: some View {
        VStack(spacing: 0) {
            streamControls
            Divider()
            viewControls
            Divider()
            ZStack {
                // No Metal device is created until a stream is open.
                if stream.phase == .idle {
                    Color(red: 0.035, green: 0.04, blue: 0.055)
                } else {
                    ViewerMetalView(stream: stream, model: model)
                }
                overlay
            }
            Divider()
            statusBar
        }
        .frame(minWidth: 760, minHeight: 560)
        .background(WindowCloseObserver(restorable: false) { stream.stop() })
        .onDisappear { stream.stop() }
    }

    // MARK: Controls

    private var live: Bool { stream.phase == .live }
    private var following: Bool { stream.source?.follow == true }

    private var streamControls: some View {
        HStack(spacing: 10) {
            Picker("Source", selection: Binding(
                get: { stream.source?.follow ?? true },
                set: { stream.restart(follow: $0) })
            ) {
                Text("Follow run").tag(true)
                Text("Re-simulate").tag(false)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(width: 200)
            .disabled(stream.source == nil)
            .help("Follow shows the run's newest checkpoints, so a director run is shown faithfully. Re-simulate steps the latest checkpoint again in a separate process, without the director.")

            Divider().frame(height: 18)

            Button(stream.control.paused ? "Run" : "Pause") { model.perform(.togglePause) }
                .disabled(!live)
                .help("Space")
            Button(following ? "Newest" : "Step") { model.perform(.step) }
                .disabled(!live)
                .help(following ? "Fetch the newest checkpoint once (.)" : "Run one step and pause (.)")

            HStack(spacing: 4) {
                Button("-") { model.perform(.slower) }
                Text(following ? "-" : "\(stream.control.every)")
                    .font(.callout.monospacedDigit())
                    .frame(minWidth: 34)
                Button("+") { model.perform(.faster) }
                Text("steps/frame").font(.caption).foregroundStyle(.secondary)
            }
            .disabled(!live || following)
            .help("Simulation steps per frame (- and +). A followed run has its own pace.")

            Spacer()

            Picker("Colour", selection: Binding(
                get: { stream.control.color },
                set: { stream.setColor($0) })
            ) {
                ForEach(stream.header?.colorModes ?? [stream.control.color], id: \.self) { Text($0) }
            }
            .frame(width: 150)
            .disabled(!live)
            .help("Colour mode (c)")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private var viewControls: some View {
        HStack(spacing: 12) {
            Toggle("Agents", isOn: $model.display.agents).help("1")
            Toggle("Bonds", isOn: $model.display.bonds).help("2")
            Toggle("Frame", isOn: $model.display.frame).help("Box edges, floor and surface planes (3)")
            Toggle("Field slice", isOn: Binding(
                get: { stream.control.fields },
                set: { stream.setFields($0) })
            )
            .disabled(!live)
            .help("The resource and chemical colours of one layer (f)")
            if let header = stream.header, header.fieldLayers > 1, stream.control.fields {
                Stepper("layer \(stream.control.fieldLayer)", value: Binding(
                    get: { stream.control.fieldLayer },
                    set: { stream.setFieldLayer($0) }
                ), in: 0...(header.fieldLayers - 1))
                .help("Field layer ([ and ])")
            }

            Spacer()

            Picker("Camera", selection: $model.cameraMode) {
                ForEach(CameraMode.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .frame(width: 130)
            .labelsHidden()
            .help("Orbit turns about the target; Fly turns about the eye and the arrow keys move (o)")
            Button("Reset view") { model.perform(.resetCamera) }
                .help("r")
            Button("Snapshot") { model.perform(.snapshot) }
                .disabled(stream.frame == nil)
                .help("Save a PNG to runs/NAME/media/view3d_STEP.png (s)")
        }
        .toggleStyle(.checkbox)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    // MARK: Overlay and status

    @ViewBuilder
    private var overlay: some View {
        if let failure = model.drawingFailure {
            banner("Cannot draw: \(failure)", error: true)
        } else {
            switch stream.phase {
            case .idle:
                banner("No stream is open. Choose a run in the main window and press Open 3D viewer.")
            case .starting:
                VStack(spacing: 8) {
                    ProgressView()
                    Text("Starting alife stream (it loads the engine first, which takes a few seconds)...")
                        .foregroundStyle(.secondary)
                }
                .padding(16)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
            case .failed(let message):
                banner(message, error: true)
            case .ended(let message):
                VStack {
                    Text(message).padding(6).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 6))
                    Spacer()
                }
                .padding(8)
            case .live:
                EmptyView()
            }
        }
    }

    private func banner(_ text: String, error: Bool = false) -> some View {
        Text(text)
            .foregroundStyle(error ? Color.red : Color.secondary)
            .multilineTextAlignment(.center)
            .padding(14)
            .frame(maxWidth: 520)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
            .textSelection(.enabled)
    }

    private var statusBar: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 14) {
                if let frame = stream.frame, let header = stream.header {
                    Text(stream.source?.follow == true ? "following" : "re-simulating")
                    Text("step \(frame.step)")
                    Text("alive \(frame.nAlive) of \(header.capacity)")
                    Text("bonds \(frame.nBonds)")
                    Text("colour \(header.colorMode(of: frame) ?? "?")")
                    if frame.hasDiagError {
                        Text("invariant violation").foregroundStyle(.red)
                    }
                    if let rate = stream.stepsPerSecond { Text(String(format: "%.1f steps/s", rate)) }
                    if stream.control.paused { Text("paused").foregroundStyle(.orange) }
                } else {
                    Text("no frame yet")
                }
                Spacer()
                if let note = model.note {
                    Text(note).foregroundStyle(.secondary)
                }
            }
            .font(.caption.monospacedDigit())
            if let source = stream.source {
                Text(sourceHint(source))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func sourceHint(_ source: StreamSource) -> String {
        if source.follow {
            return "Following runs/\(source.run): the newest checkpoint, as the run (and its director) made it."
        }
        return "Re-simulating runs/\(source.run) from its latest checkpoint in a separate process. "
            + "Re-simulation ignores the director, so a director run diverges from this view after its first "
            + "intervention; Follow shows it faithfully."
    }
}

// MARK: - The Metal view

/// Hosts a `ViewerMTKView` and feeds it from the stream. SwiftUI calls `updateNSView` when the stream, the display
/// options or the window change, which is when something may need drawing.
struct ViewerMetalView: NSViewRepresentable {
    @ObservedObject var stream: StreamController
    @ObservedObject var model: ViewerModel

    /// What the renderer has been given, so that an update copies a frame only once.
    final class Coordinator {
        var renderer: Renderer?
        var header: StreamHeader?
        var serial = -1
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> ViewerMTKView {
        let view = ViewerMTKView(frame: .zero, device: nil)
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
