import AlifeCore
import SwiftUI

/// The tabs of the Run panel.
enum RunPanelTab: String, CaseIterable, Identifiable {
    case numbers = "Numbers"
    case autopilot = "Autopilot"
    case log = "Log"
    var id: String { rawValue }
}

/// The Run panel of the research Watch (docs/APP.md §3.4, §3.7, W5): beside the observer's picture, the numbers of the world, the autopilot's
/// journal and the log, on the same dock side and with the same divider as the control center of a played world. It reads the world's own files
/// (`WorldReader`), so it shows the numbers of a world that is not growing as well as of one that is. When this app's run grows the world, the
/// panel says how far it has got and holds the Stop.
struct RunPanel: View {
    let world: String
    @ObservedObject var reader: WorldReader
    @ObservedObject var controller: RunController
    @ObservedObject var library: WorldLibrary
    @Binding var collapsed: Bool
    var side: DockSide
    /// Opens the world's Details.
    var showDetails: () -> Void
    @State private var tab: RunPanelTab = .numbers

    private var row: LibraryRow? { library.state.row(named: world) }
    private var growing: Bool { controller.runningRunName.map { NameFolding.same($0, world) } ?? false }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            if !collapsed {
                VStack(alignment: .leading, spacing: 10) {
                    progress
                    Picker("", selection: $tab) {
                        ForEach(RunPanelTab.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    content
                }
                .padding(.horizontal, 10)
                .padding(.bottom, 10)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(.regularMaterial)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Run panel")
    }

    private var header: some View {
        HStack(spacing: 8) {
            if !collapsed {
                Text("Run").font(.headline)
                Text(DetailsGuide.title(world)).font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                if let row { StateChip(state: row.state) }
                Spacer()
                Button("Details", action: showDetails)
                    .controlSize(.small)
                    .help("What this world is, how it is doing, and what you can do with it")
            } else {
                Spacer()
            }
            Button { collapsed.toggle() } label: { Image(systemName: Self.collapseSymbol(side: side, collapsed: collapsed)) }
                .buttonStyle(.borderless)
                .help(collapsed ? "Show the Run panel" : "Hide the Run panel")
                .accessibilityLabel(collapsed ? "Show the Run panel" : "Hide the Run panel")
        }
        .padding(.horizontal, 8)
        .frame(height: 28)
    }

    static func collapseSymbol(side: DockSide, collapsed: Bool) -> String {
        switch (side, collapsed) {
        case (.right, false), (.left, true): return "sidebar.right"
        case (.left, false), (.right, true): return "sidebar.left"
        case (.bottom, _): return "rectangle.bottomhalf.inset.filled"
        }
    }

    // MARK: Progress and Stop

    @ViewBuilder private var progress: some View {
        let metrics = reader.reading?.metrics
        VStack(alignment: .leading, spacing: 6) {
            if growing {
                let step = metrics?.step ?? row?.step ?? 0
                let target = row?.targetSteps
                HStack {
                    Text(target.map { "Step \(WorldsGuide.grouped(step)) of \(WorldsGuide.grouped($0))" } ?? "Step \(WorldsGuide.grouped(step))")
                        .font(.callout.monospacedDigit())
                    Spacer()
                    if controller.stopRequested {
                        Text("Stopping…").font(.caption).foregroundStyle(.secondary)
                        Button("Force stop") { controller.forceStop() }.controlSize(.small)
                    } else {
                        Button("Stop") { controller.stop() }
                            .controlSize(.small)
                            .help("Ends the run at a save point. Keep growing continues it later.")
                    }
                }
                if let target, target > 0 { ProgressView(value: min(Double(step) / Double(target), 1)) }
            } else if let row {
                Text(DetailsGuide.progress(row, now: Date())).font(.callout).fixedSize(horizontal: false, vertical: true)
                if let ended = DetailsGuide.ended(status: reader.reading?.status), row.state == .saved {
                    Text(ended).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
            } else {
                Text("Not in the library yet.").font(.callout).foregroundStyle(.secondary)
            }
        }
    }

    // MARK: Tabs

    @ViewBuilder private var content: some View {
        switch tab {
        case .numbers:
            ScrollView {
                RunNumbersView(
                    metrics: reader.reading?.metrics,
                    clock: growing ? RunNumbersView.Clock(elapsed: controller.elapsed, remaining: controller.remaining) : nil,
                    emptyText: DetailsGuide.numbersCaption(hasMetrics: false, growing: growing), titled: false)
            }
        case .autopilot:
            DirectorSnapshotView(
                snapshot: reader.reading?.director, title: "Autopilot", titled: false,
                offText: "The autopilot is off for this world.", listHeight: nil)
        case .log:
            LogView(controller: controller)
        }
    }
}
