import AlifeCore
import SwiftUI

// The pieces of the world library that both the sidebar and the Worlds page draw (docs/APP.md §2.4, §5.2).

extension WorldState {
    /// The colour of the state's dot. Never the only sign: the state's name is always beside it.
    var tint: Color {
        switch self {
        case .saved: return .secondary
        case .live: return .green
        case .paused: return .yellow
        case .starting, .growing: return .blue
        case .saving: return .orange
        case .inUseElsewhere: return .purple
        case .unfinished: return .orange
        }
    }
}

extension WorldKind {
    /// An SF Symbol: a played world is for playing, a research world came out of a flask.
    var symbol: String { self == .played ? "play.circle" : "flask" }
}

/// What a world is in, as a small capsule: a dot and the state's name.
struct StateChip: View {
    let state: WorldState

    var body: some View {
        HStack(spacing: 5) {
            Circle().fill(state.tint).frame(width: 7, height: 7)
            Text(state.title).font(.caption.weight(.medium)).lineLimit(1)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(state.tint.opacity(0.16), in: Capsule())
        .help(state.meaning)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(state.title)
        .accessibilityHint(state.meaning)
        .fixedSize()
    }
}

/// The folder icon on a card: shows the world's folder in Finder. Icon only, with its name for the screen reader and the pointer.
struct RevealButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) { Label("Reveal in Finder", systemImage: "folder") }
            .labelStyle(.iconOnly)
            .buttonStyle(.borderless)
            .help("Reveal in Finder")
    }
}

/// A button with the reason beside it when it is grey (the reasons are gathered under the buttons by `ReasonLines`).
struct CardAction: Identifiable {
    var label: String
    var reason: String?
    var id: String { label }
}

/// The reasons under a card's buttons: "Play and Make planet…: Saving…". The same reason on two buttons is said once.
struct ReasonLines: View {
    let actions: [CardAction]
    /// Name the buttons the reason is for; with one button on the card the reason beside it needs no name.
    var names = true

    var body: some View {
        let grey = actions.filter { $0.reason != nil }
        let groups = Dictionary(grouping: grey, by: { $0.reason ?? "" })
        let ordered = grey.compactMap(\.reason).reduce(into: [String]()) { if !$0.contains($1) { $0.append($1) } }
        VStack(alignment: .leading, spacing: 2) {
            ForEach(ordered, id: \.self) { reason in
                let names = (groups[reason] ?? []).map(\.label).joined(separator: " and ")
                Text(self.names ? "\(names): \(reason)" : reason).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

/// One world, as a card: what it is, its state, and the actions that make sense for it.
struct WorldCard: View {
    let row: LibraryRow
    let seedOf: String?
    let selected: Bool
    let now: Date
    let play: CardAction
    let makePlanet: CardAction
    let isOpen: Bool
    var keepGrowing = CardAction(label: "Keep growing…", reason: nil)
    var branch = CardAction(label: "Branch…", reason: nil)
    /// Watch a research world as an observer (a played world is watched while it plays: its Watch button is the open world's).
    var watch = CardAction(label: "Watch", reason: nil)
    /// Check world, for either kind of world (the same rule as Details: `DetailsGuide.reasonToCheck`).
    var check = CardAction(label: "Check world", reason: nil)
    let onSelect: () -> Void
    let onPlay: () -> Void
    let onWatch: () -> Void
    let onMakePlanet: () -> Void
    var onKeepGrowing: () -> Void = {}
    var onBranch: () -> Void = {}
    var onWatchResearch: () -> Void = {}
    var onDetails: () -> Void = {}
    var onOpenClassic: () -> Void = {}
    var onCheck: () -> Void = {}
    let onReveal: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            header
            info
            Spacer(minLength: 2)
            if row.isPlaceholder {
                Text("It opens in Watch when it is ready.").font(.caption).foregroundStyle(.secondary)
            } else {
                buttons
                ReasonLines(actions: reasons)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, minHeight: 150, alignment: .topLeading)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(border, lineWidth: selected ? 2 : 1))
        .contentShape(RoundedRectangle(cornerRadius: 10))
        .onTapGesture(perform: onSelect)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(row.name), \(row.kind.title), \(row.state.title)")
    }

    /// The reasons under the buttons: only for the ones that are grey and that the card shows.
    private var reasons: [CardAction] {
        var out: [CardAction] = []
        if row.kind == .played, !isOpen { out.append(play) }
        if row.kind == .research { out.append(watch) }
        out.append(makePlanet)
        if row.kind == .research { out.append(keepGrowing) }
        out.append(branch)
        return out.filter { $0.reason != nil }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Image(systemName: row.kind.symbol).foregroundStyle(.secondary).accessibilityHidden(true)
            Text(row.name).font(.headline).lineLimit(1).truncationMode(.middle).help(row.name)
            Spacer(minLength: 4)
            StateChip(state: row.state)
        }
    }

    @ViewBuilder private var info: some View {
        Text(row.facts).font(.caption).foregroundStyle(.secondary).lineLimit(1)
        if !row.isPlaceholder { Text(savedLine).font(.caption).foregroundStyle(.secondary).lineLimit(2) }
        if let note = row.note { Text(note).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true) }
        if let planet = row.unmadePlanet {
            Label("The planet \(planet) was not made. Make planet… finishes it.", systemImage: "arrow.right.circle")
                .font(.caption).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
        } else if row.kind == .research {
            Text(researchLine).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private var buttons: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                if row.kind == .played {
                    if isOpen {
                        Button("Watch", action: onWatch).buttonStyle(.borderedProminent)
                    } else {
                        Button("Play", action: onPlay).buttonStyle(.borderedProminent).disabled(play.reason != nil)
                    }
                }
                if row.kind == .research {
                    Button("Watch", action: onWatchResearch).buttonStyle(.borderedProminent).disabled(watch.reason != nil)
                }
                Button("Make planet…", action: onMakePlanet).buttonStyle(.bordered).disabled(makePlanet.reason != nil)
                Spacer(minLength: 0)
                Button(action: onDetails) { Label("Details", systemImage: "info.circle") }
                    .labelStyle(.iconOnly)
                    .buttonStyle(.borderless)
                    .help("Details: what this world is, how it is doing, and what you can do with it")
                RevealButton(action: onReveal)
            }
            HStack(spacing: 8) {
                if row.kind == .research {
                    Button("Keep growing…", action: onKeepGrowing).disabled(keepGrowing.reason != nil)
                }
                Button("Branch…", action: onBranch).disabled(branch.reason != nil)
                Spacer(minLength: 0)
                moreMenu
            }
        }
        .controlSize(.small)
    }

    /// The classic viewer of a flat world, and the check of a research world.
    private var moreMenu: some View {
        Menu {
            if row.shape == .flat { Button("Open Classic 2D Viewer", action: onOpenClassic) }
            Button("Check world", action: onCheck).disabled(check.reason != nil).help(check.reason ?? "Ask the engine to verify this world")
        } label: { Label("More", systemImage: "ellipsis.circle") }
            .menuStyle(.borderlessButton)
            .labelStyle(.iconOnly)
            .fixedSize()
            .help("Check this world, or open it in the classic 2D viewer (a flat world)")
            .accessibilityLabel("More actions")
    }

    private var border: Color { selected ? Color.accentColor : Color.secondary.opacity(0.25) }

    private var researchLine: String {
        if row.shape == .flat { return "A flat world is for study: it has no weather, so it cannot become a planet." }
        guard let parent = seedOf else { return "To play it, make a planet from it." }
        return "The grown world of \(parent). To play it, make a planet from it."
    }

    private var savedLine: String {
        var line = row.savedText(now: now)
        if let from = row.madeFrom { line += " · made from \(from.name)" }
        return line
    }
}

/// A build that did not finish: why, and what can be done (nothing is deleted).
struct UnfinishedCard: View {
    let build: UnfinishedRow
    let selected: Bool
    let now: Date
    let again: CardAction
    let onSelect: () -> Void
    let onMakeAgain: () -> Void
    let onReveal: () -> Void

    private var border: Color { selected ? Color.accentColor : Color.orange.opacity(0.5) }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Image(systemName: "exclamationmark.triangle").foregroundStyle(.orange).accessibilityHidden(true)
                Text(build.name).font(.headline).lineLimit(1).truncationMode(.middle).help(build.name)
                Spacer(minLength: 4)
                StateChip(state: .unfinished)
            }
            Text(build.reason).font(.caption).fixedSize(horizontal: false, vertical: true)
            if let source = build.source {
                Text("Made from \(source.name)" + (source.step.map { " at step \(WorldsGuide.grouped($0))" } ?? ""))
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text("Last touched " + WorldsGuide.relative(build.lastTouched, now: now)).font(.caption).foregroundStyle(.secondary)
            Spacer(minLength: 2)
            HStack(spacing: 8) {
                Button("Make again with a new name", action: onMakeAgain).disabled(again.reason != nil)
                Spacer(minLength: 0)
                RevealButton(action: onReveal)
            }
            .controlSize(.small)
            ReasonLines(actions: [again], names: false)
        }
        .padding(12)
        .frame(maxWidth: .infinity, minHeight: 150, alignment: .topLeading)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(border, lineWidth: selected ? 2 : 1))
        .contentShape(RoundedRectangle(cornerRadius: 10))
        .onTapGesture(perform: onSelect)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(build.name), unfinished build")
    }
}

// MARK: Sidebar

/// What a row of the sidebar selects.
enum SidebarItem: Hashable {
    case page(ShellPage)
    case world(String)
    case allWorlds
    case unfinished
}

extension SidebarItem {
    /// What choosing a row of the sidebar does: a page opens; a world opens the Worlds page on its card (a search or filter that would hide
    /// it is cleared); the rows that open a list show the list and leave nothing chosen.
    @MainActor
    func choose(shell: ShellModel, library: WorldLibrary) {
        library.sidebarAnchor = self == .page(.worlds) ? nil : self
        switch self {
        case .page(let page):
            if page == .worlds { library.selected = nil }
            shell.select(page)
        case .world(let name):
            // A world's row opens its Details (docs/APP.md §2.4); the Worlds page's card is a click away.
            library.select(name, scroll: false)
            shell.select(.details)
        case .allWorlds:
            library.filter = .all
            library.query = ""
            library.selected = nil
            shell.select(.worlds)
        case .unfinished:
            library.filter = .unfinished
            library.query = ""
            library.selected = nil
            shell.select(.worlds)
        }
    }
}

/// A world in the sidebar: its name (cut in the middle when long) and, under it, its state.
struct WorldSidebarRow: View {
    let row: LibraryRow

    var body: some View {
        // ("Saved 2 hours ago" is read again every minute: the time does not freeze on the page that was drawn first.)
        TimelineView(.everyMinute) { context in line(now: context.date) }
    }

    private func line(now: Date) -> some View {
        HStack(spacing: 8) {
            Image(systemName: row.kind.symbol).foregroundStyle(.secondary).frame(width: 16).accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 1) {
                Text(row.name).lineLimit(1).truncationMode(.middle)
                Text(row.sidebarLine(now: now)).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer(minLength: 0)
            if row.state.isActive { Circle().fill(row.state.tint).frame(width: 8, height: 8).accessibilityHidden(true) }
        }
        .padding(.vertical, 1)
        .help("\(row.name): \(row.facts). \(row.state.meaning)")
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(row.name), \(row.kind.title), \(row.state.title)")
    }
}

/// The worlds in the sidebar: the first few (the open one, what grows, then the most recent), a research world nested under the played world
/// it is the seed of, a way to the rest, and the unfinished builds.
struct LibrarySidebarSection: View {
    @ObservedObject var library: WorldLibrary
    @State private var expanded: [String: Bool] = [:]

    /// How many worlds the sidebar lists before it offers the page: a sidebar of sixty rows is not a sidebar.
    static let cap = 8

    var body: some View {
        let rows = library.state.rows
        let nodes = LibraryState.tree(rows)
        Section("Your worlds") {
            if !library.hasScanned {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("Looking at your worlds…").font(.caption).foregroundStyle(.secondary)
                }
            } else if nodes.isEmpty {
                Text("No worlds yet").font(.caption).foregroundStyle(.secondary)
            }
            ForEach(nodes.prefix(Self.cap)) { node in
                if node.children.isEmpty {
                    WorldSidebarRow(row: node.row).tag(SidebarItem.world(node.row.name))
                } else {
                    DisclosureGroup(isExpanded: binding(node)) {
                        ForEach(node.children) { child in
                            WorldSidebarRow(row: child).tag(SidebarItem.world(child.name))
                        }
                    } label: {
                        WorldSidebarRow(row: node.row).tag(SidebarItem.world(node.row.name))
                    }
                }
            }
            if nodes.count > Self.cap {
                Label("All \(rows.count) worlds…", systemImage: "list.bullet").tag(SidebarItem.allWorlds)
                    .help("Show every world on the Worlds page, with search and filters.")
            }
            if !library.state.unfinished.isEmpty {
                Label("Unfinished (\(library.state.unfinished.count))", systemImage: "exclamationmark.triangle").tag(SidebarItem.unfinished)
                    .help("Planets whose build was cancelled or failed. Nothing is deleted.")
            }
        }
    }

    /// A nested world is shown open when its seed needs the eye (it grows, or is in use), else closed until the player opens it.
    private func binding(_ node: LibraryNode) -> Binding<Bool> {
        Binding(
            get: { expanded[node.row.name] ?? node.children.contains { $0.state.isActive } },
            set: { expanded[node.row.name] = $0 })
    }
}
