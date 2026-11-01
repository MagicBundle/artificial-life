import AlifeCore
import SwiftUI

/// The Worlds page of the `alife` window (docs/APP.md §2.4, PR-C): what to do next first, then the library of worlds with its search, its
/// filters and a card for each world (its kind, shape, state in plain words and the actions that make sense for it), and the unfinished
/// builds with the reason each stopped. Every grey button has its reason beside it. The planet form opens from a world's card (a sheet);
/// PR-D replaces it with the set-up page.
struct WorldsPage: View {
    @ObservedObject var shell: ShellModel
    @ObservedObject var controller: RunController
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldLauncherModel
    @ObservedObject var library: WorldLibrary
    @AppStorage("worlds.advanced") private var advancedOpen = false
    @State private var planetSheet = false

    private var facts: WorldsGuide.Facts {
        WorldsGuide.Facts(
            folderExists: controller.repoFolderExists, folderProblem: Self.folderSentence(controller), engineReady: controller.repoReady,
            catalog: library.state.catalog, openWorld: play.isActive ? play.world : nil, openWorldIsStarting: play.phase == .starting,
            saving: Set(play.retiredWorlds))
    }

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
                    nextStepCard
                    if play.phase == .starting { startingBox }
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
        .sheet(isPresented: $planetSheet) {
            PlanetSheet(model: model, advancedOpen: $advancedOpen, reasons: planetReasons, onMake: makePlanet)
        }
    }

    /// File > Make Planet… opens the form on the world that is chosen, else on the newest.
    private func handlePlanetFormRequest() {
        guard shell.planetFormRequests != shell.planetFormHandled else { return }
        shell.planetFormHandled = shell.planetFormRequests
        openPlanetForm(from: library.selected)
    }

    private func openPlanetForm(from name: String?) {
        model.prepareForm(from: name)
        planetSheet = true
    }

    /// Play and Make planet are the player's own action: the world that goes live from them is shown (`ShellModel.playerStartsWorld`).
    private func playWorld(_ name: String) {
        guard library.stillThere(name) else { return library.noteGone(name) }
        guard let world = library.state.catalog.worlds.first(where: { $0.name == name }) else { return }
        shell.playerStartsWorld()
        model.resume(world)
    }

    private func makePlanet() {
        planetSheet = false
        shell.playerStartsWorld()
        model.startPlanet()
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
                header("flask", "Start with a small run")
                Text("A planet grows from a research run. The Lab will open with a small one ready (about two minutes): press Start there, and this page offers the planet as soon as the run has a save point.")
                    .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Button("Open the Lab") {
                    controller.prepareSmallPlanetRun()
                    shell.select(.lab)
                }
                .buttonStyle(.borderedProminent)
            case .makePlanet(let source):
                header("globe.europe.africa", "Make your first planet")
                Text("There is no world to play yet. A planet is made from a research run, and \(source) has a save point to start from. It takes a few minutes, mostly the weather warming up.")
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
            let now = Date()
            LazyVGrid(columns: Self.columns, alignment: .leading, spacing: 12) {
                ForEach(rows) { row in
                    WorldCard(
                        row: row, seedOf: seeds[row.name], selected: library.selected == row.name, now: now,
                        play: CardAction(label: "Play", reason: playable[row.name].flatMap { WorldsGuide.reason(forPlaying: $0, in: f) }),
                        makePlanet: CardAction(label: "Make planet…", reason: WorldsGuide.reasonToMakePlanet(from: row, in: f)),
                        isOpen: f.openWorld == row.name,
                        onSelect: { library.select(row.name, scroll: false) },
                        onPlay: { playWorld(row.name) },
                        onWatch: { shell.select(.watch) },
                        onMakePlanet: { library.select(row.name, scroll: false); openPlanetForm(from: row.name) },
                        onReveal: { library.reveal(row.name) })
                        .id(row.name)
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
            let now = Date()
            Text("A planet that was cancelled or failed while it was being made. It cannot be played, and nothing is deleted.")
                .font(.caption).foregroundStyle(.secondary)
            LazyVGrid(columns: Self.columns, alignment: .leading, spacing: 12) {
                ForEach(builds) { build in
                    let source = model.startedSpecs[build.name]?.source ?? build.source?.name
                    UnfinishedCard(
                        build: build, selected: library.selected == build.name, now: now,
                        again: CardAction(
                            label: "Make again with a new name",
                            reason: WorldsGuide.reasonToMakeAgain(
                                source: source, sourceHasSavePoint: source.map { s in state.catalog.sources.contains { $0.name == s } } ?? false,
                                in: f)),
                        onSelect: { library.select(build.name, scroll: false) },
                        onMakeAgain: {
                            model.prepareAgain(build)
                            planetSheet = true
                        },
                        onReveal: { library.reveal(build.name) })
                        .id(build.name)
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
                if let others = library.othersSentence { Text(others).font(.caption).foregroundStyle(.secondary) }
            }
            .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// The words beside a grey Make planet in the form's sheet: the page's reason first, then the form's own.
    private var planetReasons: [String] {
        if let why = WorldsGuide.reasonToMakePlanet(facts) { return [why] }
        return model.planetErrors
    }
}

/// The planet form in a sheet, opened from a world's card or File > Make Planet…
struct PlanetSheet: View {
    @ObservedObject var model: WorldLauncherModel
    @Binding var advancedOpen: Bool
    let reasons: [String]
    let onMake: () -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Make a planet").font(.title2.weight(.semibold))
            Text("A planet is a new world made from a save point of a world you have. Nothing is changed in the world it starts from.")
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            ScrollView {
                PlanetForm(model: model, advancedOpen: $advancedOpen).padding(.trailing, 6)
            }
            .frame(maxHeight: 420)
            Divider()
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(reasons, id: \.self) { Text($0).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true) }
                }
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Make planet", action: onMake)
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
                    .disabled(!reasons.isEmpty)
            }
        }
        .padding(20)
        .frame(width: 540)
    }
}

/// The planet form, in the words of docs/APP.md §2.5: *Start from* and *Save point* (were Source and Checkpoint), *Weather warm-up*,
/// and the technical choices behind an *Advanced* disclosure that is remembered. It edits the launcher model's `PlanetSpec`, so the
/// command it makes (`PlayLaunch.arguments`) is the one the launcher made.
struct PlanetForm: View {
    @ObservedObject var model: WorldLauncherModel
    @Binding var advancedOpen: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if model.catalog.sources.isEmpty {
                Text("There is nothing to start from yet: a planet is made from a research run that has a save point.")
                    .foregroundStyle(.secondary)
            } else {
                Picker("Start from", selection: Binding(get: { model.planet.source }, set: { model.selectSource($0) })) {
                    ForEach(model.catalog.sources) { Text(LauncherPlanetBox.title(of: $0)).tag($0.name) }
                }
                if let source = model.selectedSource, source.geometry == "sphere" { note(Self.shapeNote(source)) }
                if model.sourceIsWorld { note("You can already play this world with Play. Making a planet from it makes a new world from one of its save points.") }
                if let source = model.selectedSource {
                    Picker("Save point", selection: Binding(
                        get: { model.planet.sourceStep ?? source.checkpoints.last ?? 0 }, set: { model.selectCheckpoint($0) })
                    ) {
                        ForEach(source.checkpoints, id: \.self) { Text("step \(WorldsGuide.grouped($0))").tag($0) }
                    }
                }
                Toggle("Turn the weather on", isOn: Binding(get: { model.planet.storm }, set: { model.setStorm($0) }))
                    .disabled(model.selectedSource?.weatherOn == true || model.weatherObstacle != nil)
                if let obstacle = model.weatherObstacle { note(obstacle) }
                if model.planet.storm {
                    Stepper("Weather warm-up: \(WorldsGuide.grouped(model.planet.weatherSpinup)) steps",
                            value: $model.planet.weatherSpinup, in: 1...200_000, step: 1000)
                    if model.planet.sourceGeometry == "sphere" {
                        note("The weather needs time to settle before play starts. Repeating the same world, save point and length reuses the saved weather and starts at once.")
                    }
                }
                LabeledContent("Name") { TextField("", text: $model.planet.name).textFieldStyle(.roundedBorder) }
                DisclosureGroup("Advanced", isExpanded: $advancedOpen) {
                    VStack(alignment: .leading, spacing: 8) {
                        Toggle("This world takes powers", isOn: $model.planet.directorEnabled)
                        note("On by default. Without it the Powers tab has nothing to offer in this world.")
                        Picker("How family lines are rebuilt", selection: $model.planet.houses) {
                            ForEach(PlayHouses.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                        }
                        .pickerStyle(.segmented)
                        if let housesNote = model.housesNote { note(housesNote) }
                    }
                    .padding(.top, 6)
                }
            }
        }
    }

    /// What a planet source means for the new planet: it keeps its shape, and its weather has the overlay its shape calls for.
    static func shapeNote(_ source: PlaySource) -> String {
        let keeps = "The new planet keeps this world's shape, a sphere"
        guard !source.weatherOn, let overlay = PlanetSpec.overlay(forShape: source.planetShape) else { return keeps + "." }
        return keeps + ", with the weather overlay \(overlay)."
    }

    private func note(_ text: String) -> some View {
        Text(text).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
    }
}
