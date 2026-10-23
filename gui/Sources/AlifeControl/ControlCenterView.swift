import AlifeCore
import SwiftUI

/// The dock of the World window (DESIGN §33.4.6, §33.4.8): the clock strip, always visible; the Cast and Chronicle tabs; the
/// inspector pinned at the bottom whenever something is selected.
struct ControlCenterView: View {
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldModel
    @Binding var collapsed: Bool
    var side: DockSide
    /// The height the dock has (the alert rows that fit depend on it).
    var height: CGFloat = 700

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if collapsed {
                Spacer(minLength: 0)
            } else if side == .bottom {
                bottomDock
            } else {
                sideDock
            }
        }
        .background(.regularMaterial)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    /// On a side: the clock strip, the tabs, and the inspector pinned under them.
    private var sideDock: some View {
        // (while a power is being set up the inspector card waits: the editor needs the room, and the selection is kept)
        let card = model.editingPower ? nil : model.inspectorCard
        return VStack(spacing: 0) {
            ClockStripView(play: play, model: model)
                .padding(.horizontal, 10)
                .padding(.top, 10)
                .padding(.bottom, 4)
            StorageStripView(play: play, model: model)
                .padding(.horizontal, 10)
                .padding(.bottom, 6)
            // The alerts sit between the strip and the tabs; the rows give way before the list does (DESIGN §34.7).
            AlertStackView(play: play, model: model,
                           maxRows: DockBudget.alertRows(dockHeight: height, inspectorOpen: card != nil, editing: model.editingPower,
                                             storageNote: model.storageNoteShown))
                .padding(.horizontal, 10)
                .padding(.bottom, 6)
            Divider()
            tabs
            // Whatever is selected is pinned under the tabs, whichever is open (watching first, few inputs).
            if let card {
                Divider()
                // (its own height, up to the cap: the frame must not stretch to the cap and leave a blank under the card)
                scrollingCard(card, fill: false).frame(maxHeight: Self.cardMaxHeight, alignment: .top)
            }
        }
    }

    /// At the bottom, which is wide and short: the clock strip on the left, the tabs in the middle and the inspector on the
    /// right, each in its own column so that nothing is pushed out of the dock.
    private var bottomDock: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                ClockStripView(play: play, model: model).padding(.horizontal, 10).padding(.top, 10).padding(.bottom, 4)
                StorageStripView(play: play, model: model).padding(.horizontal, 10).padding(.bottom, 6)
                Spacer(minLength: 0)
            }
            .frame(width: Self.clockColumn)
            Divider()
            VStack(spacing: 0) { tabs }
                .frame(maxWidth: .infinity)
            if let card = model.inspectorCard {
                Divider()
                scrollingCard(card, fill: true).frame(width: Self.cardColumn)
            }
        }
    }

    private var tabs: some View {
        VStack(spacing: 0) {
            Picker("", selection: $model.tab) {
                ForEach(DockTab.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            Group {
                switch model.tab {
                case .cast: CastView(play: play, model: model)
                case .chronicle: ChronicleView(play: play, model: model)
                case .report: ReportTabView(play: play, model: model)
                case .powers: PowersTabView(play: play, model: model)
                }
            }
            .frame(maxWidth: .infinity, minHeight: Self.listMinHeight, maxHeight: .infinity)
        }
    }

    /// The card as it is when it fits, and scrolling when the dock gives it less than it needs (a bottom dock, a short window).
    /// `fill`: the card's column takes the dock's whole height (the bottom dock); otherwise it is as tall as its card.
    @ViewBuilder
    private func scrollingCard(_ card: InspectorCard, fill: Bool) -> some View {
        let fitting = ViewThatFits(in: .vertical) {
            InspectorCardView(card: card, model: model)
            ScrollView { InspectorCardView(card: card, model: model) }
        }
        if fill {
            fitting.frame(maxHeight: .infinity, alignment: .top)
        } else {
            fitting
        }
    }

    /// The tallest the card gets in a side dock; the list under the tabs keeps at least `listMinHeight`.
    static let cardMaxHeight: CGFloat = 300
    static let listMinHeight: CGFloat = 120
    static let clockColumn: CGFloat = 300
    static let cardColumn: CGFloat = 320

    private var header: some View {
        HStack(spacing: 8) {
            if !collapsed {
                Text(WorldTitle.heading(world: play.world, header: play.store.play)).font(.headline).lineLimit(1)
                if let subtitle = WorldTitle.subtitle(header: play.store.play) {
                    Text(subtitle).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
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

    var body: some View {
        Menu("View") {
            let modes = play.store.header?.colorModes ?? []
            let current = play.store.frame.flatMap { play.store.header?.colorMode(index: $0.colorIndex) }
            // The archetype and intent modes paint labels: they are tagged like every other presentation of one (DESIGN §33.7).
            let labels = play.store.play?.labels
            let validity = LabelValidity(labels: labels, environment: play.store.environment?.environment)
            Picker("Colour", selection: Binding(
                get: { current ?? "" }, set: { play.setColor($0) })
            ) {
                ForEach(modes, id: \.self) {
                    Text(validity.colorModeTitle($0, arch: labels?.arch ?? [], intent: labels?.intent ?? [])).tag($0)
                }
            }
            Divider()
            // The world's `labels on|off`: whether it sends the intent labels (INTN) at all. The record itself paints nothing; the
            // archetype and intent colour modes do (from the host's colours), and the picker above tags them.
            Toggle("Intent labels (stream)", isOn: Binding(
                get: { play.labelsOn }, set: { play.send(.labels($0)) }))
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
            // The pauses the world takes by itself. The player's choice is kept and sent to every world after its header.
            Menu("Pauses") {
                Toggle("At the end of a chapter", isOn: Binding(
                    get: { play.store.clock?.chapterPauseArmed ?? true }, set: { model.setChapterPause($0) }))
                let classes = model.pausableClasses
                if !classes.isEmpty { Divider() }
                ForEach(classes, id: \.name) { entry in
                    Toggle(entry.label, isOn: Binding(
                        get: { model.alertPauseOn(entry.name) }, set: { model.setAlertPause(entry.name, $0) }))
                }
            }
            Toggle("Skip the second click for a cost-1 power", isOn: Binding(
                get: { model.defaults.bool(forKey: WorldModel.skipConfirmKey) },
                set: { model.defaults.set($0, forKey: WorldModel.skipConfirmKey) }))
            Toggle("Raw kin events (developer)", isOn: $model.debugFilters)
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
    @ObservedObject var model: WorldModel

    var body: some View {
        let _ = play.revision
        let strip = model.clockStrip
        let live = play.phase == .live
        VStack(alignment: .leading, spacing: 8) {
            ChapterBarView(play: play, model: model)
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
                Text(verbatim: strip.step.map { Numbers.stepLabel($0) } ?? "step -")
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

/// The clock in one line over the map while the dock is collapsed (the strip is hidden with it): the pause, the lateness and
/// the checkpoint stall stay visible whichever way the dock is.
struct ClockBadge: View {
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldModel

    var body: some View {
        let _ = play.revision
        let strip = model.clockStrip
        HStack(spacing: 8) {
            Text(strip.readout).foregroundStyle(strip.paused ? Color.primary : Color.secondary)
            if strip.lateSteps > 0 {
                Text("late \(strip.lateSteps)").foregroundStyle(.orange)
            }
            if strip.saving {
                Label("saving…", systemImage: "externaldrive").foregroundStyle(.orange)
            }
            StorageBadge(storage: model.storage)
        }
        .font(.caption.monospacedDigit())
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(.regularMaterial, in: Capsule())
        .padding(10)
        .allowsHitTesting(false)
    }
}
