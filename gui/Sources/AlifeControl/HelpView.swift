import AlifeCore
import SwiftUI

/// Which page of Help is open. The Help menu sets it before it opens the window, so *World Controls* opens on its page.
@MainActor
final class HelpModel: ObservableObject {
    @Published var topic = HelpTopics.all[0].id
    func show(_ id: String) { topic = HelpTopics.topic(id) == nil ? HelpTopics.all[0].id : id }
}

/// Help (docs/APP.md §2.4): a window with a list of short topics. *World Controls* shows the key table of the Watch in view first, and the other
/// beneath it: the two sets of keys are never merged.
struct HelpView: View {
    @ObservedObject var model: HelpModel
    @ObservedObject var shell: ShellModel

    var body: some View {
        NavigationSplitView {
            List(HelpTopics.all, selection: Binding(get: { Optional(model.topic) }, set: { if let id = $0 { model.topic = id } })) { topic in
                Label(topic.title, systemImage: topic.symbol).tag(topic.id)
            }
            .navigationSplitViewColumnWidth(min: 240, ideal: 260, max: 300)
        } detail: {
            if let topic = HelpTopics.topic(model.topic) {
                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        Text(topic.title).font(.largeTitle.weight(.semibold))
                        ForEach(topic.paragraphs, id: \.self) { Text($0).fixedSize(horizontal: false, vertical: true) }
                        if topic.id == "watch" { watchTable }
                        if topic.id == HelpTopics.controlsID { controls }
                    }
                    .padding(24)
                    .frame(maxWidth: 760, alignment: .leading)
                    .frame(maxWidth: .infinity)
                }
                .background(Color(nsColor: .windowBackgroundColor))
            }
        }
        .frame(minWidth: 760, minHeight: 520)
    }

    /// What each form of Watch can and cannot do, side by side.
    private var watchTable: some View {
        VStack(alignment: .leading, spacing: 12) {
            list("A played world can", WatchGuide.playedCan)
            list("A research world can", WatchGuide.researchCan)
            list("A research world cannot", WatchGuide.researchCannot)
        }
        .padding(.top, 4)
    }

    private func list(_ title: String, _ items: [String]) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.headline)
            ForEach(items, id: \.self) { item in
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text("•")
                    Text(item).fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    // MARK: World Controls

    @ViewBuilder private var controls: some View {
        Text(HelpTopics.controlsIntro).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        // The table of the form in view first; with none in view, the played world's.
        let first = shell.watchForm ?? .played
        let order: [WatchForm] = [first, first == .played ? .research : .played]
        ForEach(order, id: \.self) { form in
            table(form, inView: form == shell.watchForm)
        }
    }

    private func table(_ form: WatchForm, inView: Bool) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text(HelpTopics.controlsHeading(form)).font(.title3.weight(.semibold))
                if inView { Text("in view now").font(.caption).padding(.horizontal, 7).padding(.vertical, 2).background(.tint.opacity(0.2), in: Capsule()) }
            }
            Grid(alignment: .topLeading, horizontalSpacing: 18, verticalSpacing: 5) {
                ForEach(ControlTables.entries(form)) { entry in
                    GridRow {
                        Text(entry.hint).font(.body.monospaced().weight(.medium)).frame(minWidth: 70, alignment: .leading)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(entry.title)
                            if let detail = entry.detail { Text(detail).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true) }
                        }
                    }
                }
            }
            ForEach(ControlTables.pointer(form), id: \.self) { Text($0).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true) }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(inView ? 0.6 : 0.3), in: RoundedRectangle(cornerRadius: 10))
    }
}
