import AlifeCore
import SwiftUI

// The pieces of the 3D viewer that the "3D Viewer" window of the two old apps and the research Watch page of the `alife` window both use
// (docs/APP.md §3.4): one set of controls for one stream, so the two cannot drift apart. They read the stream and the viewer model only.

/// Source (follow or re-simulate), pause and step, steps per frame, colour.
struct ViewerStreamControls: View {
    @ObservedObject var stream: StreamController
    @ObservedObject var model: ViewerModel
    /// Following needs a run that grows; for a world that does not, only re-simulating is offered.
    var canFollow = true
    var chooseSource: (Bool) -> Void

    private var live: Bool { stream.phase == .live }
    private var following: Bool { stream.source?.follow == true }

    var body: some View {
        HStack(spacing: 10) {
            Picker("Source", selection: Binding(
                get: { stream.source?.follow ?? true },
                set: { chooseSource($0) })
            ) {
                Text("Follow run").tag(true).disabled(!canFollow)
                Text("Re-simulate").tag(false)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(width: 200)
            .disabled(stream.source == nil)
            .help(canFollow
                  ? "Follow shows the run's newest save points, so a run with an autopilot is shown faithfully. Re-simulate steps the latest save point again in a separate process, without the autopilot."
                  : "This world is not growing, so there is nothing to follow. Re-simulate steps its latest save point in a separate process, without the autopilot.")

            Divider().frame(height: 18)

            Button(stream.control.paused ? "Run" : "Pause") { model.perform(.togglePause) }
                .disabled(!live)
                .help("Space")
            Button(following ? "Newest" : "Step") { model.perform(.step) }
                .disabled(!live)
                .help(following ? "Fetch the newest save point once (.)" : "Run one step and pause (.)")

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

            ViewerColourPicker(state: model.colour, stream: stream)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }
}

/// Agents, bonds, frame, field slice and layer, camera mode, reset, snapshot.
struct ViewerViewControls: View {
    @ObservedObject var stream: StreamController
    @ObservedObject var model: ViewerModel

    private var live: Bool { stream.phase == .live }

    var body: some View {
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
}

/// What the picture's place says while there is no picture: no stream, starting, failed, ended, or a renderer that cannot draw.
struct ViewerOverlay: View {
    @ObservedObject var stream: StreamController
    @ObservedObject var model: ViewerModel
    var idleText: String

    var body: some View {
        if let failure = model.drawingFailure {
            banner("Cannot draw: \(failure)", error: true)
        } else {
            switch stream.phase {
            case .idle:
                banner(idleText)
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
}

/// The step, the population, the colour and the speed of the picture, and where it comes from.
struct ViewerStatusBar: View {
    @ObservedObject var stream: StreamController
    @ObservedObject var model: ViewerModel
    /// The line under the numbers; nil for the sentence of the 3D Viewer window.
    var sourceLine: ((StreamSource) -> String)?

    var body: some View {
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
                Text(sourceLine?(source) ?? Self.sourceHint(source))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    static func sourceHint(_ source: StreamSource) -> String {
        if source.follow {
            return "Following runs/\(source.run): the newest checkpoint, as the run (and its director) made it."
        }
        return "Re-simulating runs/\(source.run) from its latest checkpoint in a separate process. "
            + "Re-simulation ignores the director, so a director run diverges from this view after its first "
            + "intervention; Follow shows it faithfully."
    }
}

/// The same controls in two rows that fit a narrow picture (the research Watch's free rectangle is about 525 points at the window's minimum
/// size, beside the sidebar and the Run panel): the source, pause, step and colour in the first; what to show (a menu), the steps per frame
/// while re-simulating, the camera, reset and snapshot in the second.
struct ViewerCompactControls: View {
    @ObservedObject var stream: StreamController
    @ObservedObject var model: ViewerModel
    var canFollow = true
    var chooseSource: (Bool) -> Void

    private var live: Bool { stream.phase == .live }
    private var following: Bool { stream.source?.follow == true }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Picker("Source", selection: Binding(get: { stream.source?.follow ?? true }, set: { chooseSource($0) })) {
                    Text("Follow").tag(true).disabled(!canFollow)
                    Text("Re-simulate").tag(false)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 180)
                .disabled(stream.source == nil)
                .help(canFollow
                      ? "Follow shows the run's newest save points, so a run with an autopilot is shown faithfully. Re-simulate steps the latest save point again in a separate process, without the autopilot."
                      : "This world is not growing, so there is nothing to follow. Re-simulate steps its latest save point in a separate process, without the autopilot.")
                Button(stream.control.paused ? "Run" : "Pause") { model.perform(.togglePause) }
                    .disabled(!live)
                    .help("Space")
                Button(following ? "Newest" : "Step") { model.perform(.step) }
                    .disabled(!live)
                    .help(following ? "Fetch the newest save point once (.)" : "Run one step and pause (.)")
                Spacer(minLength: 4)
                ViewerColourPicker(state: model.colour, stream: stream)
            }
            .controlSize(.small)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            Divider()
            HStack(spacing: 8) {
                Menu("Show") {
                    Toggle("Agents", isOn: $model.display.agents)
                    Toggle("Bonds", isOn: $model.display.bonds)
                    Toggle("Frame", isOn: $model.display.frame)
                    Toggle("Field slice", isOn: Binding(get: { stream.control.fields }, set: { stream.setFields($0) })).disabled(!live)
                }
                .fixedSize()
                .help("What the picture shows: the animals, their bonds, the box edges and floor, and one layer of food and chemicals")
                if let header = stream.header, header.fieldLayers > 1, stream.control.fields {
                    Stepper("layer \(stream.control.fieldLayer)", value: Binding(
                        get: { stream.control.fieldLayer }, set: { stream.setFieldLayer($0) }), in: 0...(header.fieldLayers - 1))
                        .fixedSize()
                        .help("Field layer ([ and ])")
                }
                if !following {
                    HStack(spacing: 4) {
                        Button("-") { model.perform(.slower) }
                        Text("\(stream.control.every)").font(.callout.monospacedDigit()).frame(minWidth: 28)
                        Button("+") { model.perform(.faster) }
                        Text("steps/frame").font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    }
                    .fixedSize()
                    .disabled(!live)
                    .help("Simulation steps per frame (- and +)")
                }
                Spacer(minLength: 4)
                Picker("Camera", selection: $model.cameraMode) {
                    ForEach(CameraMode.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 110)
                .help("Orbit turns about the target; Fly turns about the eye and the arrow keys move (o)")
                Button("Reset view") { model.perform(.resetCamera) }.help("r")
                Button("Snapshot") { model.perform(.snapshot) }
                    .disabled(stream.frame == nil)
                    .help("Save a PNG under the world's media folder (s)")
            }
            .controlSize(.small)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
        }
    }
}
