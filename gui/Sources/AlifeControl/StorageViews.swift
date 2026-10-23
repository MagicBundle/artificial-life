import AlifeCore
import SwiftUI

/// The words of the Storage strip (DESIGN §34.10), in one place so that the tooltip, the details and the tests agree.
enum StorageWords {
    /// The switch is named for what the player asked for: recording. On keeps the last saves, off keeps the newest one.
    static let switchLabel = "Recording"
    static let help = "Recording off keeps only the newest save; your world stays resumable; logs are still written."
    static let pending = "Older saves are deleted at the next save, not now."
    static func tooltip(_ mode: SaveMode, keep: Int?) -> String {
        switch mode {
        case .all:
            return "Recording is on: the last \(keep ?? 3) saves and the origin are kept. Switching it off keeps only the newest save"
                + " and the origin, and deletes the older ones at the next save. Your world stays resumable; logs are still written."
        case .newest:
            return "Recording is off: only the newest save and the origin are kept. " + pending
                + " Switch it on to keep the last \(keep ?? 3) saves again. Your world stays resumable; logs are still written."
        }
    }
    /// Said in the details when the disk is getting full and saves are what fills it.
    static let advice = "Switching Recording off keeps only the newest save, about one save of disk instead of three."
    /// Why the switch cannot be turned off in a research run played in place (the world refuses `saves newest` there).
    static let adopted = "Recording stays on: this world is a research run played in place, and keeping only the newest save would"
        + " delete the run's own checkpoints. To do that on purpose, start it with: alife play RUN --saves newest"
    static let unknown = "This world's server does not offer the choice: it keeps its last saves."

    /// The tooltip of the switch: what the state in force keeps and does, or why it cannot be changed.
    static func switchHelp(_ mode: SaveMode, header: PlayHeader.Saves?) -> String {
        guard header != nil else { return unknown }
        let other: SaveMode = mode == .all ? .newest : .all
        if !SaveModeView.offers(other, header: header) { return adopted }
        return tooltip(mode, keep: header?.keep)
    }
}

/// The Storage strip under the clock strip: the size of the world on disk and the recording switch that decides how many saves it
/// keeps. One line (the details are in a popover), so the dock keeps its room; the number turns orange or red, and shows the room
/// left, when the disk is getting full.
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
        let saves = play.store.play?.saves  // nil: a server that predates the line cannot take it
        let other: SaveMode = mode == .all ? .newest : .all
        HStack(spacing: 4) {
            Button { details.toggle() } label: {
                HStack(spacing: 3) {
                    Image(systemName: readout.health == .ok ? "externaldrive" : "externaldrive.badge.exclamationmark")
                    Text(readout.total).font(.callout.monospacedDigit()).lineLimit(1)
                    if let room = readout.room { Text(room).font(.caption.monospacedDigit()).lineLimit(1) }
                }
                .foregroundStyle(Self.color(readout.health))
            }
            .buttonStyle(.borderless)
            .help(readout.warning ?? "How much this world takes on disk. Click for the details.")
            .accessibilityLabel("Storage \(readout.stripText)")
            .popover(isPresented: $details, arrowEdge: .bottom) {
                StorageDetailsView(readout: readout, keep: saves?.keep, offersNewest: SaveModeView.offers(.newest, header: saves))
            }
            Spacer(minLength: 2)
            Toggle(StorageWords.switchLabel, isOn: Binding(get: { mode == .all }, set: { choose($0 ? .all : .newest) }))
                .toggleStyle(.switch)
                .controlSize(.small)
                .fixedSize()
                .disabled(!live || !SaveModeView.offers(other, header: saves))
                .help(StorageWords.switchHelp(mode, header: saves))
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
    /// Whether the world takes `saves newest` (not a research run played in place): only then is switching off suggested.
    var offersNewest = true

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Storage on disk").font(.headline)
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                row("Saves", "\(readout.saves) in \(readout.saveCount) save\(readout.saveCount == 1 ? "" : "s")")
                row("Logs", readout.logs)
                row("Total", readout.total)
                if let sinceChapter = readout.sinceChapter { row("This chapter (approx.)", sinceChapter) }
                if let sinceStart = readout.sinceStart { row("This session", sinceStart) }
                if let free = readout.free { row("Disk", free, tint: StorageStripView.color(readout.health)) }
            }
            if let warning = readout.warning {
                Label(warning, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption).foregroundStyle(StorageStripView.color(readout.health))
                    .fixedSize(horizontal: false, vertical: true)
                if readout.mode == .all, offersNewest {
                    Text(StorageWords.advice).font(.caption).fixedSize(horizontal: false, vertical: true)
                }
            }
            Divider()
            Text(StorageWords.help).font(.caption).fixedSize(horizontal: false, vertical: true)
            Text(readout.mode == .all ? "Recording is on: keeping the last \(keep ?? 3) saves and the origin."
                 : "Recording is off: keeping the newest save and the origin." + (readout.pruningPending ? " " + StorageWords.pending : ""))
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
        let readout = StorageReadout(sample: storage.sample, mode: .all)
        let health = readout.health
        if health != .ok {
            let room = readout.free.map { " (\($0))" } ?? ""
            Label((health == .critical ? "disk almost full" : "low disk") + room, systemImage: "externaldrive.badge.exclamationmark")
                .foregroundStyle(StorageStripView.color(health))
        }
    }
}
