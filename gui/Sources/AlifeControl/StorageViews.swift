import AlifeCore
import SwiftUI

/// The words of the Storage strip (DESIGN §34.10, §34.11), in one place so that the segments, the tooltip, the details and the tests
/// agree. The choice names what each mode keeps: the newest save and the origin (the save the world began from), or the newest
/// save alone.
enum StorageWords {
    /// The segment of each mode, in the order of `SaveMode.allCases`. Short, because the strip is one line in the narrowest dock (260
    /// points beside the size): the full names are `name(_:)`, in the details and the tooltip.
    static func segment(_ mode: SaveMode) -> String {
        switch mode {
        case .all: return "All"
        case .newest: return "Start"
        case .single: return "Newest"
        }
    }
    /// The full name of a mode.
    static func name(_ mode: SaveMode) -> String {
        switch mode {
        case .all: return "All saves"
        case .newest: return "Newest + start"
        case .single: return "Newest only"
        }
    }
    /// The accessibility label of the choice (it is not printed: the segments are its text).
    static let label = "Saves kept"
    /// What every choice leaves true: the world can be resumed, and the logs are kept whatever the saves.
    static let resumable = "Your world stays resumable; logs are still written."
    /// Said where a choice deletes saves: they go at the next save, not now.
    static let pending = "Older saves are deleted at the next save, not now."
    /// Whether a choice deletes the older saves at the next save: every choice but All saves. Those are confirmed first.
    static func deletesOlder(_ mode: SaveMode) -> Bool { mode != .all }
    static func confirmTitle(_ mode: SaveMode) -> String { "Keep \(name(mode).lowercased())?" }
    static func confirmMessage(_ mode: SaveMode, keep: Int?) -> String {
        keeps(mode, keep: keep) + " The older saves are deleted at the next save, and they cannot be restored."
    }
    /// What a mode keeps, in one sentence (the details list the three, the tooltip the one in force).
    static func keeps(_ mode: SaveMode, keep: Int?) -> String {
        switch mode {
        case .all: return name(.all) + ": the last \(keep ?? 3) saves and the origin, the save the world began from."
        case .newest: return name(.newest) + ": the newest save and the origin."
        case .single: return name(.single) + ": the newest save. The origin goes too."
        }
    }
    static func tooltip(_ mode: SaveMode, keep: Int?) -> String {
        switch mode {
        case .all:
            return keeps(.all, keep: keep) + " Choosing Newest + start or Newest only deletes the older saves at the next save. "
                + resumable
        case .newest:
            return keeps(.newest, keep: keep) + " " + pending + " All saves keeps the last \(keep ?? 3) again. " + resumable
        case .single:
            return keeps(.single, keep: keep) + " " + pending + " All saves keeps the last \(keep ?? 3) again. " + resumable
        }
    }
    /// Said in the details when the disk is getting full and saves are what fills it.
    static let advice = "Choosing Newest only keeps one save of disk instead of three."
    /// Why the choice is fixed in a research run played in place (the world refuses the other modes there, which would delete the
    /// run's own checkpoints).
    static let adopted = "Only All saves: this world is a research run played in place, and keeping fewer saves would delete the run's"
        + " own checkpoints. To do that on purpose, start it with: alife play RUN --saves newest"
    static let unknown = "This world's server does not offer the choice: it keeps its last saves."

    /// The tooltip of the choice: what the mode in force keeps and does, or why it cannot be changed.
    static func choiceHelp(_ mode: SaveMode, header: PlayHeader.Saves?) -> String {
        guard header != nil else { return unknown }
        if header?.adopted == true { return adopted }
        return tooltip(mode, keep: header?.keep)
    }
}

/// The Storage strip under the clock strip: the size of the world on disk and the choice of the saves it keeps (all, newest and
/// the origin, or the newest only). One line (the details are in a popover), so the dock keeps its room; the number turns orange or
/// red when the disk is getting full (the room left is in the tooltip and the details).
struct StorageStripView: View {
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldModel
    @ObservedObject var storage: StorageMonitor
    @State private var details = false
    @State private var confirming: SaveMode?  // a choice that deletes saves, waiting for the player's yes

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
        HStack(spacing: 4) {
            Button { details.toggle() } label: {
                HStack(spacing: 3) {
                    Image(systemName: readout.health == .ok ? "externaldrive" : "externaldrive.badge.exclamationmark")
                    // (the room left is not printed here: the choice takes the line. It is in the tooltip and the details)
                    Text(storage.sample == nil ? "…" : readout.total).font(.callout.monospacedDigit()).lineLimit(1)
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
            // (the choice is one line: the two modes that delete saves are offered together, so the picker is fixed as a whole in a
            // research run played in place, where only All saves is kept)
            Picker(StorageWords.label, selection: Binding(get: { mode }, set: { choose($0) })) {
                ForEach(SaveMode.allCases, id: \.self) { Text(StorageWords.segment($0)).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .controlSize(.small)
            .fixedSize()
            .disabled(!live || !SaveModeView.offers(.newest, header: saves))
            .help(StorageWords.choiceHelp(mode, header: saves))
        }
        .confirmationDialog(
            confirming.map(StorageWords.confirmTitle) ?? "Keep these saves?",
            isPresented: Binding(get: { confirming != nil }, set: { if !$0 { confirming = nil } }),
            titleVisibility: .visible
        ) {
            Button("Delete the older saves", role: .destructive) {
                if let mode = confirming {
                    confirming = nil
                    confirmed(mode)
                }
            }
            Button("Cancel", role: .cancel) { confirming = nil }
        } message: {
            if let mode = confirming { Text(StorageWords.confirmMessage(mode, keep: saves?.keep)) }
        }
    }

    private func choose(_ mode: SaveMode) {
        guard Self.command(choosing: mode, current: play.savesMode, header: play.store.play?.saves) != nil else { return }
        if StorageWords.deletesOlder(mode) {
            confirming = mode  // one click on a segment is not a yes to deleting saves
            return
        }
        confirmed(mode)
    }

    private func confirmed(_ mode: SaveMode) {
        guard let command = Self.command(choosing: mode, current: play.savesMode, header: play.store.play?.saves) else { return }
        _ = play.send(command)
    }

    /// The line a choice sends: nil when it is the mode in force, or one the world does not take (DESIGN 34.10).
    static func command(choosing mode: SaveMode, current: SaveMode, header: PlayHeader.Saves?) -> PlayCommand? {
        guard mode != current, SaveModeView.offers(mode, header: header) else { return nil }
        return .saves(mode)
    }

    static func color(_ health: StorageHealth) -> Color {
        switch health {
        case .ok: return .primary
        case .low: return .orange
        case .critical: return .red
        }
    }
}

/// The popover of the strip: the split between saves and logs, the growth, the room left, what each choice keeps and the
/// shared saves.
struct StorageDetailsView: View {
    let readout: StorageReadout
    let keep: Int?
    /// Whether the world takes `saves newest` (not a research run played in place): only then is a smaller choice suggested.
    var offersNewest = true

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Storage on disk").font(.headline)
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                row("Saves", "\(readout.saves) in \(readout.saveCount) save\(readout.saveCount == 1 ? "" : "s")")
                if readout.saving { row("Saving", "a save is being written; its file counts until it is whole") }
                row("Logs", readout.logs)
                row("Total", readout.total)
                if let sinceChapter = readout.sinceChapter { row("This chapter (approx.)", sinceChapter) }
                if let sinceStart = readout.sinceStart { row("This session", sinceStart) }
                if let free = readout.free { row("Disk", free, tint: StorageStripView.color(readout.health)) }
            }
            if let shared = readout.shared {
                Text(shared).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
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
            ForEach(SaveMode.allCases, id: \.self) { mode in
                Text(StorageWords.keeps(mode, keep: keep))
                    .font(.caption).fontWeight(mode == readout.mode ? .semibold : .regular)
                    .foregroundStyle(mode == readout.mode ? .primary : .secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Text(StorageWords.resumable).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            if readout.mode != .all, readout.pruningPending {
                Text(StorageWords.pending).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
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
