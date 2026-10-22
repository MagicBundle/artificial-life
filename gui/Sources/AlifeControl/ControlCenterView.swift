import AlifeCore
import SwiftUI

/// The dock of the World window (DESIGN §33.4.6, §33.4.8): the clock strip, always visible; the Cast and Chronicle tabs; the
/// inspector pinned at the bottom whenever something is selected.
struct ControlCenterView: View {
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldModel
    @Binding var collapsed: Bool
    var side: DockSide
    @State private var tab = Tab.cast

    enum Tab: String, CaseIterable, Identifiable {
        case cast = "Cast"
        case chronicle = "Chronicle"
        var id: String { rawValue }
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if !collapsed {
                ClockStripView(play: play)
                    .padding(10)
                Divider()
                Picker("", selection: $tab) {
                    ForEach(Tab.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                Group {
                    switch tab {
                    case .cast: CastView(play: play, model: model)
                    case .chronicle: ChronicleView(play: play, model: model)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                // Whatever is selected is pinned under the tabs, whichever is open (watching first, few inputs).
                if let card = model.inspectorCard {
                    Divider()
                    InspectorCardView(card: card, model: model)
                }
            } else {
                Spacer(minLength: 0)
            }
        }
        .background(.regularMaterial)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var header: some View {
        HStack(spacing: 8) {
            if !collapsed {
                Text(play.world ?? "No world").font(.headline).lineLimit(1)
                Spacer()
                ViewMenu(play: play, model: model)
            } else {
                Spacer()
            }
            Button {
                collapsed.toggle()
            } label: {
                Image(systemName: collapseSymbol)
            }
            .buttonStyle(.borderless)
            .help(collapsed ? "Show the control center" : "Hide the control center")
        }
        .padding(.horizontal, 8)
        .frame(height: 28)
    }

    private var collapseSymbol: String {
        switch (side, collapsed) {
        case (.right, false), (.left, true): return "sidebar.right"
        case (.left, false), (.right, true): return "sidebar.left"
        case (.bottom, _): return "rectangle.bottomhalf.inset.filled"
        }
    }
}

/// The View menu: the colour mode, labels, and the layers.
struct ViewMenu: View {
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldModel
    @State private var labelsOn = true

    var body: some View {
        Menu("View") {
            let modes = play.store.header?.colorModes ?? []
            let current = play.store.frame.flatMap { play.store.header?.colorMode(index: $0.colorIndex) }
            Picker("Colour", selection: Binding(
                get: { current ?? "" }, set: { play.setColor($0) })
            ) {
                ForEach(modes, id: \.self) { Text($0).tag($0) }
            }
            Divider()
            Toggle("Labels", isOn: Binding(
                get: { labelsOn }, set: { labelsOn = $0; play.send(.labels($0)) }))
            Divider()
            // The agent toggle refers to the layer of the level in view: instances close up, the 1 px points at the planet level.
            if model.level == .planet {
                Toggle("Agent points", isOn: $model.display.agentPoints)
            } else {
                Toggle("Agents", isOn: $model.display.agents)
            }
            Toggle("Bonds", isOn: $model.display.bonds)
            Toggle("Territory", isOn: $model.display.territory)
            Toggle("Weather map", isOn: $model.display.map)
            Toggle("Frame", isOn: $model.display.frame)
            Divider()
            Button("Reset view") { model.resetCamera() }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .disabled(play.phase != .live)
    }
}

/// The clock strip: the rate control, the step buttons and the readout (DESIGN §33.4.6). It stays above the tabs so that
/// pausing and the "saving..." stall are visible whichever tab is open.
struct ClockStripView: View {
    @ObservedObject var play: PlayController

    var body: some View {
        let baseRate = play.store.play?.baseRate ?? 14.4
        let strip = ClockStrip(clock: play.store.clock, requested: play.requestedRate, baseRate: baseRate)
        let live = play.phase == .live
        VStack(alignment: .leading, spacing: 8) {
            Picker("Rate", selection: Binding<ClockSegment?>(
                get: { strip.selected },
                set: { if let segment = $0 { play.send(segment.command) } })
            ) {
                ForEach(ClockSegment.all, id: \.self) { Text($0.label).tag(Optional($0)) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .disabled(!live)
            HStack(spacing: 8) {
                if strip.paused {
                    Button("Resume") { play.resume() }.disabled(!live)
                        .help("Resume at the rate the world holds (space)")
                }
                Button("+1") { play.step(1) }.disabled(!live)
                    .help("Run one step and pause (.)")
                if strip.paused {
                    Button("+25") { play.step(25) }.disabled(!live)
                        .help("Run one tick of the storm tracker (25 steps) and pause")
                }
                Spacer()
                Text(strip.step.map { "step \($0)" } ?? "step -")
                    .font(.callout.monospacedDigit())
            }
            HStack(spacing: 10) {
                Text(strip.readout).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                if strip.lateSteps > 0 {
                    Text("late \(strip.lateSteps)").font(.caption.monospacedDigit()).foregroundStyle(.orange)
                }
                Spacer()
                if strip.saving {
                    Label("saving…", systemImage: "externaldrive")
                        .font(.caption).foregroundStyle(.orange)
                }
            }
        }
    }
}
