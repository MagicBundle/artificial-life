import AlifeCore
import SwiftUI

/// The Details page of a world (docs/APP.md §2.4, §3.7, §7.5): what it is, how it is doing, and what can be done with it, in plain words, with
/// the engine's names in smaller type. It is about the world chosen in the sidebar or on a card (`WorldLibrary.selected`), and it reads that
/// world's own files (`WorldReader`): a world that is not growing has its numbers, its autopilot journal and the settings it was made with here,
/// which the Lab showed only for the run it was set to. The page starts nothing by itself.
struct DetailsPage: View {
    @ObservedObject var shell: ShellModel
    @ObservedObject var controller: RunController
    @ObservedObject var play: PlayController
    @ObservedObject var library: WorldLibrary
    @ObservedObject var setup: SetupModel
    @ObservedObject var reader: WorldReader
    var switcher: WorldSwitcher?

    private var name: String? { library.selected }
    private var row: LibraryRow? { name.flatMap { library.state.row(named: $0) } }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                if let row {
                    content(row)
                } else if let build = name.flatMap({ n in library.state.unfinished.first { $0.name == n } }) {
                    unfinished(build)
                } else {
                    nothingChosen
                }
            }
            .padding(24)
            .frame(maxWidth: 860, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .navigationTitle(name.map(DetailsGuide.title) ?? "Details")
        .navigationSubtitle("Details")
        .onAppear {
            reader.follow(name)
            reader.retain()
        }
        .onDisappear { reader.release() }
        .onChange(of: name) { _, new in reader.follow(new) }
        .onChange(of: row?.step) { _, _ in reader.readSoon() }
        .onChange(of: row?.state) { _, _ in reader.readSoon() }
    }

    // MARK: A world

    @ViewBuilder private func content(_ row: LibraryRow) -> some View {
        let actions = WorldActionSet.make(row: row, controller: controller, play: play, library: library, setup: setup)
        let growing = controller.runningRunName.map { NameFolding.same($0, row.name) } ?? false
        let seeds = WorldsPage.seedParents(LibraryState.tree(library.state.rows))
        let reading = reader.reading?.world == row.name ? reader.reading : nil
        header(row)
        section("What this world is") {
            ForEach(DetailsGuide.whatItIs(row, grownFor: seeds[row.name]), id: \.self) { Text($0).fixedSize(horizontal: false, vertical: true) }
            let branches = DetailsGuide.branches(of: row.name, in: library.state.rows)
            if !branches.isEmpty {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(branches.count == 1 ? "Branch:" : "Branches:").foregroundStyle(.secondary)
                    ForEach(branches, id: \.self) { b in
                        Button(b) { library.select(b, scroll: false) }.buttonStyle(.link)
                    }
                }
            }
        }
        section("What you can do") { doing(row, actions) }
        section("How it is doing") {
            Text(DetailsGuide.progress(row, now: Date())).fixedSize(horizontal: false, vertical: true)
            if let ended = DetailsGuide.ended(status: reading?.status), row.state == .saved {
                Text(ended).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            if let note = row.note { Text(note).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true) }
            RunNumbersView(
                metrics: reading?.metrics, clock: growing ? RunNumbersView.Clock(elapsed: controller.elapsed, remaining: controller.remaining) : nil,
                emptyText: DetailsGuide.numbersCaption(hasMetrics: false, growing: growing), titled: false)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
            if reading?.metrics != nil {
                Text(DetailsGuide.numbersCaption(hasMetrics: true, growing: growing)).font(.caption).foregroundStyle(.secondary)
            }
        }
        section("Autopilot") {
            DirectorSnapshotView(
                snapshot: reading?.director, title: "Autopilot", titled: false,
                offText: "The autopilot is off for this world.", listHeight: 200)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
        }
        if let made = reading?.madeWith { madeWith(made) }
    }

    private func header(_ row: LibraryRow) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Image(systemName: row.kind.symbol).font(.title2).foregroundStyle(.secondary).accessibilityHidden(true)
                Text(DetailsGuide.title(row.name)).font(.largeTitle.weight(.semibold)).lineLimit(2)
                StateChip(state: row.state)
            }
            Text(row.facts).foregroundStyle(.secondary)
            Text(DetailsGuide.folderLine(row.name)).font(.caption.monospaced()).foregroundStyle(.tertiary).textSelection(.enabled)
        }
    }

    private func section(_ title: String, @ViewBuilder _ content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.title3.weight(.semibold))
            content()
        }
    }

    // MARK: Made with

    private func madeWith(_ made: MadeWith) -> some View {
        section("Settings it was made with") {
            Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 3) {
                fact("Shape", made.shape)
                if let seed = made.seed { fact("Seed", String(seed)) }
                if let steps = made.steps { fact("Asked to grow to", WorldsGuide.grouped(steps) + " steps") }
                if let created = made.created { fact("Created", created.formatted(date: .abbreviated, time: .shortened)) }
                fact("Autopilot", made.autopilot)
            }
            DisclosureGroup("All \(made.settings.count) settings") {
                Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 3) {
                    ForEach(made.settings) { setting in
                        GridRow {
                            Text(setting.title).foregroundStyle(.secondary).help(setting.key)
                            Text(setting.value).font(.body.monospacedDigit()).textSelection(.enabled)
                        }
                    }
                }
                .padding(.vertical, 4)
            }
        }
    }

    private func fact(_ title: String, _ value: String) -> some View {
        GridRow {
            Text(title).foregroundStyle(.secondary)
            Text(value).textSelection(.enabled)
        }
    }

    // MARK: What can be done

    @ViewBuilder private func doing(_ row: LibraryRow, _ a: WorldActionSet) -> some View {
        let isOpen = play.isActive && play.world == row.name
        HStack(spacing: 8) {
            if row.kind == .played {
                if isOpen {
                    Button("Watch") { shell.select(.watch) }.buttonStyle(.borderedProminent)
                } else {
                    Button("Play") { playWorld(row) }.buttonStyle(.borderedProminent).disabled(a.play.reason != nil)
                }
            } else {
                Button("Watch") { shell.observe(row.name) }.buttonStyle(.borderedProminent).disabled(a.watch.reason != nil)
            }
            Button("Make planet…") { setup.openMakePlanet(from: row.name) }.disabled(a.makePlanet.reason != nil)
            if row.kind == .research { Button("Keep growing…") { setup.openKeepGrowing(world: row.name) }.disabled(a.keepGrowing.reason != nil) }
            Button("Branch…") { setup.openBranch(from: row.name) }.disabled(a.branch.reason != nil)
        }
        HStack(spacing: 8) {
            if row.kind == .research {
                Button("Check world") { check(row) }.disabled(a.check.reason != nil)
                if row.shape == .flat { Button("Open Classic 2D Viewer") { controller.openClassicViewer(world: row.name) }.disabled(a.classic.reason != nil) }
            }
            Button("Reveal in Finder") { Task { await library.reveal(row.name) } }
        }
        let grey: [CardAction] = [row.kind == .played && !isOpen ? a.play : nil, row.kind == .research ? a.watch : nil, a.makePlanet,
                                  row.kind == .research ? a.keepGrowing : nil, a.branch, row.kind == .research ? a.check : nil]
            .compactMap { $0 }.filter { $0.reason != nil }
        ReasonLines(actions: grey)
    }

    private func playWorld(_ row: LibraryRow) {
        let facts = WorldActionSet.worldsFacts(controller: controller, play: play, library: library)
        Task { await WorldOpener(shell: shell, library: library, launcher: setup.launcher, switcher: switcher).play(row.name, facts: facts) }
    }

    /// The check runs as the engine's one job at a time; its result is in the log (Activity).
    private func check(_ row: LibraryRow) {
        controller.inspect(world: row.name)
        shell.select(.activity)
    }

    // MARK: Nothing, or a build that did not finish

    private var nothingChosen: some View {
        VStack(alignment: .leading, spacing: 10) {
            Image(systemName: "info.circle").font(.system(size: 30)).foregroundStyle(.secondary).accessibilityHidden(true)
            Text("Choose a world").font(.title2.weight(.semibold))
            Text(name.map { "\($0) is not in the library any more. It may have been moved or removed." }
                 ?? "Details shows what a world is, how it is doing, and what you can do with it. Choose one in the sidebar, or press Details on its card.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Button("Go to Worlds") { shell.select(.worlds) }.buttonStyle(.borderedProminent)
        }
        .padding(.top, 40)
    }

    private func unfinished(_ build: UnfinishedRow) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Text(build.name).font(.largeTitle.weight(.semibold))
                StateChip(state: .unfinished)
            }
            Text(build.reason).fixedSize(horizontal: false, vertical: true)
            Text("It cannot be played, and nothing is deleted. The Worlds page lists it under Unfinished, with Make again with a new name.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            HStack {
                Button("Go to Worlds") { library.filter = .unfinished; shell.select(.worlds) }
                Button("Reveal in Finder") { Task { await library.reveal(build.name) } }
            }
        }
    }
}
