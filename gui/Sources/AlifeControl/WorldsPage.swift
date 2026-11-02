import AlifeCore
import SwiftUI

/// The Worlds page of the `alife` window (docs/APP.md §2.4, PR-C): what to do next first, then the library of worlds with its search, its
/// filters and a card for each world (its kind, shape, state in plain words and the actions that make sense for it), and the unfinished
/// builds with the reason each stopped. Every grey button has its reason beside it. Make planet, Branch and Keep growing open the set-up
/// page (PR-D) on the world of the card.
struct WorldsPage: View {
    @ObservedObject var shell: ShellModel
    @ObservedObject var controller: RunController
    @ObservedObject var play: PlayController
    @ObservedObject var setup: SetupModel
    @ObservedObject var library: WorldLibrary
    /// The sheet that stops the open world before another is played (R5).
    var switcher: WorldSwitcher?

    private var model: WorldLauncherModel { setup.launcher }
    private var worldModel: WorldModel { shell.session.model }

    private var facts: WorldsGuide.Facts { WorldActionSet.worldsFacts(controller: controller, play: play, library: library) }

    /// What the page says about a folder that is not there: what the folder is for. (The form below says what is wrong with the one
    /// chosen, in the words of the Lab's Repository box.)
    static func folderSentence(_ controller: RunController) -> String {
        "alife needs the folder that holds its engine (.venv/bin/alife) and your worlds (runs/)."
    }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    if let note = shell.launchNote { launchNote(note) }
                    if let failure = shell.startFailure { failureCard(failure) }
                    if let gone = library.vanished { goneNote(gone) }
                    // (while a world is being made the bar says so, and the card would offer to start another)
                    if !setup.coordinator.isActive { nextStepCard }
                    if play.phase == .starting, !setup.coordinator.isActive { startingBox }
                    // With no folder or no engine the card above says what to do, and that is all the page needs to say.
                    if controller.repoReady, showsLibrary { librarySection }
                }
                .padding(24)
                .frame(maxWidth: 860, alignment: .leading)
                .frame(maxWidth: .infinity)
            }
            .onChange(of: library.reveal) { _, request in
                if let request { withAnimation { proxy.scrollTo(request.name, anchor: .center) } }
            }
            .onAppear {
                if let request = library.reveal, request.name == library.selected { proxy.scrollTo(request.name, anchor: .center) }
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .onAppear {
            library.refresh()
            handlePlanetFormRequest()
        }
        .onChange(of: shell.planetFormRequests) { _, _ in handlePlanetFormRequest() }
    }

    /// File > Make Planet… opens the form on the world that is chosen, else on the newest.
    private func handlePlanetFormRequest() {
        guard shell.planetFormRequests != shell.planetFormHandled else { return }
        shell.planetFormHandled = shell.planetFormRequests
        openPlanetForm(from: library.selected)
    }

    private func openPlanetForm(from name: String?) {
        setup.openMakePlanet(from: name)
    }

    // Watch (the observer) is a page now; the classic viewer of a flat world and the check stay on the card's menu.
    private func openClassic(_ row: LibraryRow) { controller.openClassicViewer(world: row.name) }

    /// The check runs as the engine's one job at a time: when another job holds it, the page says so instead of jumping to Activity.
    private func check(_ row: LibraryRow) {
        guard controller.repoReady, controller.job == nil else {
            shell.launchNote = controller.repoReady ? "Another job is running. Wait for it to finish, then check \(row.name)." : "Set up the engine first."
            return
        }
        controller.inspect(world: row.name)
        shell.select(.activity)
    }

    /// Play is the player's own action: the world that goes live from it is shown (`ShellModel.playerStartsWorld`).
    private func playWorld(_ name: String) {
        Task { await WorldOpener(shell: shell, library: library, launcher: model, switcher: switcher).play(name, facts: facts) }
    }

    // MARK: Notices

    private func launchNote(_ text: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
            Text(text).textSelection(.enabled)
            Spacer()
            Button("Dismiss") { shell.launchNote = nil }
        }
        .padding(12)
        .background(.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
    }

    /// A world that was there a moment ago and is not: it was removed outside the app, or its drive went away.
    private func goneNote(_ name: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: "questionmark.folder").foregroundStyle(.orange)
            Text("\(name) is no longer in this folder. It may have been moved or removed, or its drive may have been disconnected.")
                .fixedSize(horizontal: false, vertical: true)
            Spacer()
            Button("Dismiss") { library.vanished = nil }
        }
        .padding(12)
        .background(.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
    }

    /// A world that did not open: what happened in a sentence, the engine's own message under it, and where the whole story is.
    private func failureCard(_ failure: StartFailure) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(failure.explanation, systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
                .font(.headline)
                .fixedSize(horizontal: false, vertical: true)
            Text(failure.message).font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                Button("Show Activity") { shell.select(.activity) }.buttonStyle(.borderedProminent)
                Button("Dismiss") { shell.startFailure = nil }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
    }

    // MARK: The next step

    @ViewBuilder private var nextStepCard: some View {
        let step = WorldsGuide.nextStep(facts)
        VStack(alignment: .leading, spacing: 10) {
            switch step {
            case .chooseFolder(let problem):
                header("folder.badge.questionmark", "Where is alife?")
                Text(problem).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                RepoFolderForm(controller: controller)
                Text("You can change it later in Settings (Command-comma).").font(.caption).foregroundStyle(.secondary)
            case .setUpEngine:
                header("wrench.and.screwdriver", "Set up the engine")
                Text("The folder \(controller.repoPath) has no engine yet (.venv/bin/alife is missing). Setting it up installs the packages with uv; it takes a minute or two and removes nothing.")
                    .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                HStack {
                    Button("Set up engine") { controller.setupEnvironment() }
                        .buttonStyle(.borderedProminent)
                        .disabled(controller.job != nil)
                    Button("Check again") { controller.refreshRepo() }
                    if controller.job == .setup {
                        ProgressView().controlSize(.small)
                        Text("Setting up. Progress is in Activity.").font(.caption).foregroundStyle(.secondary)
                        Button("Show Activity") { shell.select(.activity) }.buttonStyle(.link)
                    } else if controller.job != nil {
                        Text("Another job is running; wait for it or stop it in Activity.").font(.caption).foregroundStyle(.secondary)
                    }
                }
                RepoFolderForm(controller: controller)
                Text("uv is needed. If Set up engine reports that it is missing, install it from docs.astral.sh/uv and press Check again.")
                    .font(.caption).foregroundStyle(.secondary)
            case .makeFirstRun:
                header("sparkles", "Start your first world")
                Text("alife grows a small planet for you and then opens it. It takes about two minutes, and you see how far it has got all the time.")
                    .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                HStack {
                    Button("Create a small planet") { setup.openCreate(.recipe(Recipes.smallPlanet.id)) }
                        .buttonStyle(.borderedProminent)
                    Button("Choose something else…") { setup.openCreate() }
                        .buttonStyle(.link)
                }
            case .makePlanet(let source):
                header("globe.europe.africa", "Make your first planet")
                Text("There is no world to play yet. A planet is made from a world with a save point, and \(source) has one. It takes a short while, mostly the weather warming up.")
                    .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                if let growing = WorldsGuide.growingNote(source: source, growing: controller.runningRunName) {
                    Text(growing).font(.caption).foregroundStyle(.secondary)
                }
                HStack {
                    Button("Make planet…") { openPlanetForm(from: source) }
                        .buttonStyle(.borderedProminent)
                        .disabled(WorldsGuide.reasonToMakePlanet(facts) != nil)
                    if let why = WorldsGuide.reasonToMakePlanet(facts) {
                        Text(why).font(.caption).foregroundStyle(.secondary)
                    } else {
                        Text("You can choose the name and the weather before it starts.").font(.caption).foregroundStyle(.secondary)
                    }
                }
            case .play(let name):
                header("play.circle", "Play \(name)")
                if let world = library.state.catalog.worlds.first(where: { $0.name == name }) {
                    Text(WorldsGuide.summary(world)).foregroundStyle(.secondary)
                }
                HStack {
                    Button("Play") { playWorld(name) }
                        .buttonStyle(.borderedProminent)
                    Text("Opens at 1x. Nothing starts until you press Play.").font(.caption).foregroundStyle(.secondary)
                }
            case .waitForSave:
                header("hourglass", "The worlds are busy")
                Text("Every played world is being saved, or is in use by another app. Wait a moment, or choose a world below and press Make planet…")
                    .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            case .watch(let name):
                header("eye", "\(name) is open")
                HStack {
                    Button("Watch") { shell.select(.watch) }.buttonStyle(.borderedProminent)
                    Button("Close World") { shell.closeWorld() }
                    Text("Close World saves first.").font(.caption).foregroundStyle(.secondary)
                }
                // D4: the world is paused while the player is on another page, and the page says so.
                if shell.awayHold.isAway {
                    Label(worldModel.picksUpWhereLeft ? "Paused while you are on this page. It continues when you go back to Watch."
                          : "Paused while you are on this page. It stays paused until you press Run in Watch.", systemImage: "pause.circle")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if let note = SessionRules.gpuNote(runGoing: controller.hasRunJob, worldOpen: true) {
                    Label(note, systemImage: "gauge.with.needle").font(.caption).foregroundStyle(.secondary)
                }
            case .starting(let name):
                header("hourglass", "Starting \(name)")
                Text("A planet's fork and weather warm-up take a while. The page moves to Watch when it is ready.")
                    .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
    }

    private func header(_ symbol: String, _ title: String) -> some View {
        Label(title, systemImage: symbol).font(.title2.weight(.semibold))
    }

    // MARK: Starting

    private var startingBox: some View {
        GroupBox("Starting \(play.world ?? "the world")") {
            VStack(alignment: .leading, spacing: 6) {
                ScrollView {
                    Text(play.progress.suffix(40).joined(separator: "\n"))
                        .font(.caption.monospaced())
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(height: 80)
                HStack {
                    Text("Cancelling leaves an unfinished directory; nothing is deleted.")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button("Cancel") { play.stop() }
                }
            }
        }
    }


    // MARK: The library

    /// With nothing in the folder (no world, no build, nothing left out) the card above says what to do, and a list with zeros would repeat it.
    private var showsLibrary: Bool {
        let s = library.state
        return !library.hasScanned || !s.rows.isEmpty || !s.unfinished.isEmpty || !s.offline.isEmpty || !s.others.isEmpty
    }

    private var librarySection: some View {
        let state = library.state
        return VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Text("Your worlds").font(.title3.weight(.semibold))
                Spacer()
                searchField
                Button { setup.openCreate() } label: { Label("New world", systemImage: "plus") }
                    .buttonStyle(.borderedProminent)
                    .help("Make a new world from a starting point")
            }
            Picker("Show", selection: $library.filter) {
                ForEach(LibraryFilter.allCases) { Text("\($0.title) \(state.counts.count($0))").tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
            .accessibilityLabel("Show")
            if !library.hasScanned {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Looking at your worlds…").foregroundStyle(.secondary)
                }
                .padding(.vertical, 8)
            } else if library.filter == .unfinished {
                unfinishedGrid
            } else {
                worldGrid
            }
            footnotes
        }
    }

    private var searchField: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass").foregroundStyle(.secondary).accessibilityHidden(true)
            TextField("Search worlds", text: $library.query)
                .textFieldStyle(.plain)
                .accessibilityLabel("Search worlds")
            if !library.query.isEmpty {
                Button { library.query = "" } label: { Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary) }
                    .buttonStyle(.plain)
                    .help("Clear the search")
                    .accessibilityLabel("Clear the search")
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .frame(width: 240)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 7))
        .overlay(RoundedRectangle(cornerRadius: 7).stroke(Color.secondary.opacity(0.3)))
    }

    private static let columns = [GridItem(.adaptive(minimum: 250, maximum: 420), spacing: 12, alignment: .top)]

    @ViewBuilder private var worldGrid: some View {
        let state = library.state
        let rows = state.visible(filter: library.filter, query: library.query)
        if rows.isEmpty {
            emptyMessage
        } else {
            let f = facts
            let playable = Dictionary(state.catalog.worlds.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })
            let seeds = Self.seedParents(LibraryState.tree(state.rows))
            let ruleFacts = setup.ruleFacts
            TimelineView(.everyMinute) { context in
            let now = context.date
            LazyVGrid(columns: Self.columns, alignment: .leading, spacing: 12) {
                ForEach(rows) { row in
                    WorldCard(
                        row: row, seedOf: seeds[row.name], selected: library.selected == row.name, now: now,
                        play: CardAction(label: "Play", reason: playable[row.name].flatMap { WorldsGuide.reason(forPlaying: $0, in: f) }),
                        makePlanet: CardAction(label: "Make planet…", reason: WorldsGuide.reasonToMakePlanet(from: row, in: f) ?? SetupRules.reason(.makePlanet, on: row, in: ruleFacts)),
                        isOpen: f.openWorld == row.name,
                        keepGrowing: CardAction(label: "Keep growing…", reason: SetupRules.reason(.keepGrowing, on: row, in: ruleFacts)),
                        branch: CardAction(label: "Branch…", reason: SetupRules.reason(.branch, on: row, in: ruleFacts)),
                        watch: CardAction(label: "Watch", reason: WatchGuide.reasonToWatch(row, openWorld: f.openWorld, runGrowing: controller.runningRunName)),
                        check: CardAction(label: "Check world", reason: DetailsGuide.reasonToCheck(row, engineReady: controller.repoReady, jobGoing: controller.job != nil)),
                        onSelect: { library.select(row.name, scroll: false) },
                        onPlay: { playWorld(row.name) },
                        onWatch: { shell.select(.watch) },
                        onMakePlanet: { library.select(row.name, scroll: false); openPlanetForm(from: row.name) },
                        onKeepGrowing: { library.select(row.name, scroll: false); setup.openKeepGrowing(world: row.name) },
                        onBranch: { library.select(row.name, scroll: false); setup.openBranch(from: row.name) },
                        onWatchResearch: { library.select(row.name, scroll: false); shell.observe(row.name) },
                        onDetails: { library.select(row.name, scroll: false); shell.select(.details) },
                        onOpenClassic: { openClassic(row) },
                        onCheck: { check(row) },
                        onReveal: { Task { await library.reveal(row.name) } })
                        .id(row.name)
                }
            }
            }
        }
    }

    /// For each research world that nests under a played world, the played world's name.
    static func seedParents(_ nodes: [LibraryNode]) -> [String: String] {
        var out: [String: String] = [:]
        for node in nodes { for child in node.children { out[child.name] = node.row.name } }
        return out
    }

    @ViewBuilder private var unfinishedGrid: some View {
        let state = library.state
        let builds = state.unfinishedRows(query: library.query)
        if builds.isEmpty {
            emptyMessage
        } else {
            let f = facts
            Text("A planet that was cancelled or failed while it was being made. It cannot be played, and nothing is deleted.")
                .font(.caption).foregroundStyle(.secondary)
            TimelineView(.everyMinute) { context in
            let now = context.date
            LazyVGrid(columns: Self.columns, alignment: .leading, spacing: 12) {
                ForEach(builds) { build in
                    let spec = model.startedSpecs[build.name]
                    let source = spec?.source ?? build.source?.name
                    let step = spec.map { $0.sourceStep } ?? build.source?.step
                    UnfinishedCard(
                        build: build, selected: library.selected == build.name, now: now,
                        again: CardAction(
                            label: "Make again with a new name",
                            reason: WorldsGuide.reasonToMakeAgain(
                                source: source, sourceStep: step ?? nil,
                                sourceSteps: source.flatMap { s in state.catalog.sources.first { $0.name == s }?.checkpoints }, in: f)),
                        onSelect: { library.select(build.name, scroll: false) },
                        onMakeAgain: { setup.openMakeAgain(build) },
                        onReveal: { Task { await library.reveal(build.name) } })
                        .id(build.name)
                }
            }
            }
        }
    }

    private var emptyMessage: some View {
        let text = WorldsGuide.emptyList(filter: library.filter, query: library.query, counts: library.state.counts)
        return VStack(alignment: .leading, spacing: 8) {
            Text(text).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            if !library.query.isEmpty {
                Button("Clear the search") { library.query = "" }
            } else if library.filter != .all {
                Button("Show all worlds") { library.filter = .all }
            }
        }
        .padding(.vertical, 8)
    }

    /// What the list leaves out, said once under it: links to a drive that is not connected, and folders that are not worlds.
    @ViewBuilder private var footnotes: some View {
        let unfinished = library.state.unfinished.count
        if library.hasScanned {
            VStack(alignment: .leading, spacing: 4) {
                if let offline = library.offlineSentence {
                    Label(offline, systemImage: "externaldrive.badge.xmark").font(.caption).foregroundStyle(.secondary)
                }
                if library.filter != .unfinished, unfinished > 0 {
                    Text(unfinished == 1 ? "1 unfinished build is under Unfinished." : "\(unfinished) unfinished builds are under Unfinished.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                ForEach(library.problemSentences, id: \.self) { sentence in
                    Label(sentence, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.secondary)
                }
                if let others = library.othersSentence { Text(others).font(.caption).foregroundStyle(.secondary) }
            }
            .fixedSize(horizontal: false, vertical: true)
        }
    }
}
