import AlifeCore
import SwiftUI

// The tabs and cards of the control center (DESIGN §33.4.8): the Cast, the Chronicle, the inspector pinned under them, and the
// toasts and the follow badge over the map. Every list is built by a pure type of AlifeCore (`CastPolicy`,
// `ChronicleReducer`, `Inspector`); the views only lay it out and send clicks to the `WorldModel`.

// MARK: - Cast

/// The Cast tab: every live storm, the eight largest lineages, and the organism in focus.
struct CastView: View {
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldModel

    var body: some View {
        let _ = play.revision
        let store = play.store
        let baseRate = store.play?.baseRate ?? 14.4
        let storms = CastPolicy.storms(store.storms, baseRate: baseRate)
        let lineages = CastPolicy.lineages(store.identities, territory: store.territory)
        List {
            if !storms.isEmpty {
                Section("Storms") {
                    ForEach(storms) { storm in
                        StormRow(storm: storm, selected: model.selection == .storm(id: storm.id), following: model.followTarget == .storm(id: storm.id))
                            .contentShape(Rectangle())
                            .onTapGesture(count: 2) { model.selectStorm(storm.id, fly: true, follow: true) }
                            .onTapGesture { model.selectStorm(storm.id, fly: true, follow: false) }
                            .onHover { model.overlay.setCastHover($0 ? .storm(id: storm.id) : nil) }
                    }
                }
            }
            if !lineages.isEmpty {
                Section {
                    ForEach(lineages) { lineage in
                        LineageRow(lineage: lineage, names: store.play?.labels.arch ?? [],
                                   validity: LabelValidity(labels: store.play?.labels, environment: store.environment?.environment),
                                   selected: model.selection == .kin(id: lineage.id),
                                   watched: model.watch.contains(lineage.id), onWatch: { model.toggleWatch(kin: lineage.id) })
                            .contentShape(Rectangle())
                            .onTapGesture(count: 2) { model.selectKin(id: lineage.id, fly: true, follow: true) }
                            .onTapGesture { model.selectKin(id: lineage.id, fly: true, follow: false) }
                            .onHover { model.overlay.setCastHover($0 ? .kin(id: lineage.id) : nil) }
                    }
                } header: {
                    HStack {
                        Text("Lineages")
                        if store.identities?.tables.exact == false {
                            Text("dynasties approximate").font(.caption2).foregroundStyle(.orange)
                        }
                    }
                }
            }
            Section("Focus") {
                if let uid = model.selection.organismUID {
                    HStack {
                        Text("Organism \(uid)").font(.callout.monospacedDigit())
                        Spacer()
                        if model.followTarget == .organism(uid: uid) {
                            Text("following").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                } else {
                    Text("Click an organism to follow it").font(.callout).foregroundStyle(.secondary)
                }
            }
            if storms.isEmpty && lineages.isEmpty {
                Text(play.phase == .live ? "Nothing to list yet" : "No world is live").font(.callout).foregroundStyle(.secondary)
            }
        }
        .listStyle(.sidebar)
    }
}

struct StormRow: View {
    let storm: CastStorm
    let selected: Bool
    let following: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(storm.title).font(.callout.weight(storm.named ? .semibold : .regular))
                Text(storm.hemisphere).font(.caption).foregroundStyle(.secondary)
                Spacer()
                if following { Image(systemName: "scope").font(.caption).help("The camera follows it") }
                Text(storm.coasting ? "coasting" : String(format: "%.1f min", storm.ageMinutes))
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            HStack(spacing: 6) {
                Bar(fraction: storm.peakFraction, tint: storm.named ? .orange : .gray)
                    .frame(height: 5)
                Text(String(format: "%.2f f0", storm.peak)).font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
        .listRowBackground(selected ? Color.accentColor.opacity(0.18) : Color.clear)
    }
}

struct LineageRow: View {
    let lineage: CastLineage
    let names: [String]
    /// The N1 gate on the labels (DESIGN §33.7): a mix is tagged provisional unless every archetype in it is valid.
    var validity = LabelValidity(labels: nil, environment: nil)
    let selected: Bool
    /// The group is in the watch set (the lineage alerts read it); `onWatch` toggles it. Nil: no star (a stand-in row).
    var watched = false
    var onWatch: (() -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                RoundedRectangle(cornerRadius: 2).fill(Color(WorldPalette.slotColor(UInt8(clamping: lineage.slot)) ?? WorldPalette.other))
                    .frame(width: 10, height: 10)
                Text(lineage.name).font(.callout.weight(.semibold)).lineLimit(1)
                Spacer()
                if let onWatch {
                    Button(action: onWatch) { Image(systemName: watched ? "star.fill" : "star") }
                        .buttonStyle(.borderless).controlSize(.small)
                        .help(watched ? "Stop watching this group" : "Watch this group: its alerts reach the stack")
                }
                Text(String(format: "%.1f%%", 100 * lineage.share)).font(.caption.monospacedDigit())
            }
            if let house = lineage.namedAfter {
                Text("named after House \(house)").font(.caption2).foregroundStyle(.secondary)
            }
            if let mix = lineage.archetypes, !mix.isEmpty {
                HStack(spacing: 6) {
                    ArchetypeBar(mix: mix).frame(height: 5).help(mixHelp(mix))
                    if !validity.mixIsValid(mix) { Text("provisional").font(.caption2).foregroundStyle(.secondary) }
                }
            }
            HStack(spacing: 8) {
                // (verbatim, grouped by us: a localised Int reads "6.216" in much of Europe, DESIGN §34.7)
                if let births = lineage.births { Text(verbatim: "\(Numbers.grouped(births)) born").font(.caption2).foregroundStyle(.secondary) }
                Spacer()
                Text(verbatim: "\(lineage.territoryCells) cells").font(.caption2).foregroundStyle(.secondary)
            }
            if let deaths = CastPolicy.deathSummary(lineage.deaths) {
                // (its own line, and as many as it needs: it was cut to "died: 3091 starved, 3099 killed, 3...")
                Text(verbatim: "died: \(deaths)").font(.caption2).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.vertical, 2)
        .listRowBackground(selected ? Color.accentColor.opacity(0.18) : Color.clear)
    }

    /// The archetype shares in words, for the bar's tooltip.
    private func mixHelp(_ mix: [Double]) -> String {
        zip(mix, 1...).filter { $0.0 >= 0.01 }.map { "\(Names.label(UInt8($0.1), in: names)) \(Int(($0.0 * 100).rounded()))%" }
            .joined(separator: ", ") + (validity.mixIsValid(mix) ? " (validated labels)" : " (provisional labels)")
    }
}

/// A thin horizontal bar filled to `fraction`.
struct Bar: View {
    let fraction: Double
    var tint: Color = .accentColor

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.primary.opacity(0.12))
                Capsule().fill(tint).frame(width: max(geometry.size.width * min(max(fraction, 0), 1), 0))
            }
        }
    }
}

/// The archetype mix as a stacked bar in the archetype palette (index `i` of `mix` is archetype `i + 1`).
struct ArchetypeBar: View {
    let mix: [Double]

    var body: some View {
        GeometryReader { geometry in
            HStack(spacing: 0) {
                ForEach(mix.indices, id: \.self) { i in
                    Rectangle().fill(Color(WorldPalette.archetypes[min(i + 1, WorldPalette.archetypes.count - 1)]))
                        .frame(width: geometry.size.width * CGFloat(max(mix[i], 0)))
                }
                Spacer(minLength: 0)
            }
            .clipShape(Capsule())
            .background(Capsule().fill(Color.primary.opacity(0.12)))
        }
    }
}

// MARK: - Chronicle

/// The Chronicle tab: the world's events newest first, grouped by minute of game time, filtered by family.
struct ChronicleView: View {
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldModel

    var body: some View {
        let _ = play.revision
        let baseRate = play.store.play?.baseRate ?? 14.4
        let groups = ChronicleReducer.groups(
            play.store.chronicle, filter: model.chronicleFilter, baseRate: baseRate,
            hidingRawKin: play.store.play?.alerts != nil && !model.debugFilters)
        VStack(spacing: 0) {
            filters
            List {
                ForEach(groups) { group in
                    Section {
                        ForEach(group.events, id: \.id) { event in
                            EventRow(event: event)
                                .contentShape(Rectangle())
                                .onTapGesture { model.activate(event) }
                        }
                    } header: {
                        HStack {
                            Text(group.title)
                            Text(group.subtitle).font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                }
                if groups.isEmpty {
                    Text(play.store.chronicle.events.isEmpty ? "Nothing has happened yet" : "Nothing under these filters")
                        .font(.callout).foregroundStyle(.secondary)
                }
            }
            .listStyle(.sidebar)
        }
    }

    private var filters: some View {
        HStack(spacing: 4) {
            ForEach(ChronicleFilter.choices, id: \.label) { choice in
                Toggle(choice.label, isOn: Binding(
                    get: { model.chronicleFilter.contains(choice.filter) },
                    set: { on in
                        if on { model.chronicleFilter.insert(choice.filter) } else { model.chronicleFilter.remove(choice.filter) }
                    }))
                    .toggleStyle(.button)
                    .controlSize(.small)
            }
        }
        .padding(.horizontal, 10)
        .padding(.bottom, 4)
    }
}

struct EventRow: View {
    let event: WorldEvent

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(event.text).font(.callout).fontWeight(event.imp >= 2 ? .semibold : .regular)
            Text(verbatim: Numbers.stepLabel(event.step)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - The inspector

/// The card pinned at the bottom of the dock whenever something is selected.
struct InspectorCardView: View {
    let card: InspectorCard
    @ObservedObject var model: WorldModel

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text(card.title).font(.headline).lineLimit(1)
                Spacer()
                Button {
                    model.clearSelection()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.borderless)
                .help("Clear the selection (esc)")
            }
            if let subtitle = card.subtitle {
                Text(subtitle).font(.callout).foregroundStyle(card.alive ? Color.secondary : Color.orange)
            }
            if let energy = card.energy {
                HStack(spacing: 6) {
                    Text("Energy").font(.caption).foregroundStyle(.secondary).frame(width: 78, alignment: .leading)
                    Bar(fraction: energy.fraction, tint: .green).frame(height: 6)
                    Text(energy.text).font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                }
            }
            ForEach(card.rows.indices, id: \.self) { i in
                let row = card.rows[i]
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(row.label).font(.caption).foregroundStyle(.secondary).frame(width: 78, alignment: .leading)
                    Text(row.value).font(.callout).lineLimit(2)
                    if row.provisional { Text("provisional").font(.caption2).foregroundStyle(.secondary) }
                    Spacer(minLength: 0)
                }
            }
            ForEach(card.notes, id: \.self) { note in
                Text(note).font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: 8) {
                if let parent = card.selectableParent {
                    Button("Select parent") { model.selectOrganism(parent) }
                        .help("Select the parent: it is alive")
                }
                if let successor = card.successor {
                    Button("Go to successor") { model.selectOrganism(successor) }
                }
                if card.alive, !model.selection.isNone {
                    Button("Follow") { model.followSelection() }.help("Follow it and fly in (F)")
                }
            }
            .controlSize(.small)
        }
        .padding(10)
        .background(.thinMaterial)
    }
}

// MARK: - Over the map

/// The toasts of important events (a storm named or dissolved, a kin group split or dissolved, extinction): three seconds
/// each, at most three, a click flies there.
struct ToastStackView: View {
    @ObservedObject var model: WorldModel

    var body: some View {
        VStack(spacing: 6) {
            ForEach(model.toasts) { toast in
                Text(toast.text)
                    .font(.callout)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.primary.opacity(0.15)))
                    .onTapGesture {
                        model.activate(toast.action)
                        model.dismissToast(id: toast.id)
                    }
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .animation(.easeInOut(duration: 0.2), value: model.toasts.map(\.id))
        .padding(.top, 12)
        .frame(maxWidth: .infinity, alignment: .top)
        .allowsHitTesting(!model.toasts.isEmpty)
    }
}

/// "Following X" at the corner of the map while the camera follows something; esc stops it.
struct FollowBadge: View {
    @ObservedObject var model: WorldModel
    @ObservedObject var play: PlayController

    var body: some View {
        let _ = play.revision
        if let text = label {
            Text(text + "  ·  drag or esc to stop")
                .font(.caption)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(.regularMaterial, in: Capsule())
                .padding(10)
                .allowsHitTesting(false)
        }
    }

    private var label: String? { Self.label(for: model.followTarget, store: play.store) }

    /// The words for what the camera follows; nil when it follows nothing.
    static func label(for target: FollowTarget, store: WorldStore) -> String? {
        switch target {
        case .none: return nil
        case .organism(let uid): return "Following organism \(uid)"
        case .storm(let id):
            return "Following " + (store.storms?.storms.storms.first { $0.id == id }.map(Names.storm) ?? "storm \(id)")
        case .lineage(let id):
            return "Following " + (store.identities?.tables.kin.first { $0.id == id }?.name ?? "kin group \(id)")
        }
    }
}
