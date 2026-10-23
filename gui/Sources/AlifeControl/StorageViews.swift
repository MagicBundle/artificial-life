import AlifeCore
import SwiftUI

/// The words of the Storage strip (DESIGN §34.10), in one place so that the tooltip, the details and the tests agree.
enum StorageWords {
    static let help = "Recording off keeps only the newest save; your world stays resumable; logs are still written."
    static let pending = "Older saves are deleted at the next save, not now."
    static func label(_ mode: SaveMode) -> String {
        switch mode {
        case .all: return "All saves"
        case .newest: return "Newest only"
        }
    }
    static func tooltip(_ mode: SaveMode, keep: Int?) -> String {
        switch mode {
        case .all: return "Keeping the last \(keep ?? 3) saves and the origin. " + help
        case .newest: return "Keeping only the newest save and the origin. " + pending + " " + help
        }
    }
}

/// The Storage strip under the clock strip: the size of the world on disk and the switch that decides how many saves it keeps. One
/// line (the details are in a popover), so the dock keeps its room; the number turns orange when the disk is getting full.
struct StorageStripView: View {
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldModel
    @ObservedObject var storage: StorageMonitor
    @State private var details = false

    init(play: PlayController, model: WorldModel) {
        self.play = play
        self.model = model
        storage = model.storage
    }

    var body: some View {
        let _ = play.revision
        let mode = play.savesMode
        let readout = storage.readout(mode: mode)
        let live = play.phase == .live
        let known = play.store.play?.saves != nil  // a server that predates the line cannot take it
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 6) {
                Button { details.toggle() } label: {
                    HStack(spacing: 4) {
                        Image(systemName: readout.health == .ok ? "externaldrive" : "externaldrive.badge.exclamationmark")
                        Text(readout.total).font(.callout.monospacedDigit()).lineLimit(1)
                    }
                    .foregroundStyle(Self.color(readout.health))
                }
                .buttonStyle(.borderless)
                .help(readout.warning ?? "How much this world takes on disk. Click for the details.")
                .accessibilityLabel("Storage \(readout.total)")
                .popover(isPresented: $details, arrowEdge: .bottom) { StorageDetailsView(readout: readout, keep: play.store.play?.saves?.keep) }
                Spacer(minLength: 4)
                Picker("Saves kept", selection: Binding(get: { mode }, set: { choose($0) })) {
                    ForEach(SaveMode.allCases, id: \.self) { Text(StorageWords.label($0)).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .controlSize(.small)
                .fixedSize()
                .disabled(!live || !known)
                .help(known ? StorageWords.tooltip(mode, keep: play.store.play?.saves?.keep)
                            : "This world's server does not offer the choice: it keeps its last saves.")
            }
            if model.storageNoteShown {
                Text(StorageWords.pending).font(.caption2).foregroundStyle(.secondary)
            }
        }
    }

    private func choose(_ mode: SaveMode) {
        guard mode != play.savesMode else { return }
        play.send(.saves(mode))
    }

    static func color(_ health: StorageHealth) -> Color {
        switch health {
        case .ok: return .primary
        case .low: return .orange
        case .critical: return .red
        }
    }
}

/// The popover of the strip: the split between saves and logs, the growth, the room left and what the switch does.
struct StorageDetailsView: View {
    let readout: StorageReadout
    let keep: Int?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Storage").font(.headline)
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                row("Saves", "\(readout.saves) in \(readout.saveCount) file\(readout.saveCount == 1 ? "" : "s")")
                row("Logs", readout.logs)
                row("Total", readout.total)
                if let sinceChapter = readout.sinceChapter { row("This chapter", sinceChapter) }
                if let sinceStart = readout.sinceStart { row("This session", sinceStart) }
                if let free = readout.free { row("Disk", free, tint: StorageStripView.color(readout.health)) }
            }
            if let warning = readout.warning {
                Label(warning, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption).foregroundStyle(StorageStripView.color(readout.health))
                    .fixedSize(horizontal: false, vertical: true)
            }
            Divider()
            Text(StorageWords.help).font(.caption).fixedSize(horizontal: false, vertical: true)
            Text(readout.mode == .all ? "Now keeping the last \(keep ?? 3) saves and the origin." : "Now keeping the newest save and the origin. " + StorageWords.pending)
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .frame(width: 300, alignment: .leading)
    }

    private func row(_ name: String, _ value: String, tint: Color = .primary) -> some View {
        GridRow {
            Text(name).foregroundStyle(.secondary)
            Text(value).monospacedDigit().foregroundStyle(tint)
        }
        .font(.callout)
    }
}

/// The disk warning over the map while the dock is collapsed (the strip is hidden with it).
struct StorageBadge: View {
    @ObservedObject var storage: StorageMonitor

    var body: some View {
        let health = StorageReadout(sample: storage.sample, mode: .all).health
        if health != .ok {
            Label(health == .critical ? "disk almost full" : "low disk", systemImage: "externaldrive.badge.exclamationmark")
                .foregroundStyle(StorageStripView.color(health))
        }
    }
}
