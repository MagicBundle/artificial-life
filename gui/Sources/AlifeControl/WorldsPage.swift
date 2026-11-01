import AlifeCore
import SwiftUI

/// The Worlds page of the `alife` window (docs/APP.md §2.4, PR-B): the launcher's content as a page. What to do next is said first;
/// below are the worlds that can be played and the form that makes a planet (the launcher's "New planet" form, with the words of
/// §2.5). Every grey button has its reason beside it. PR-C gives the list its states and nesting; PR-D replaces the form.
struct WorldsPage: View {
    @ObservedObject var shell: ShellModel
    @ObservedObject var controller: RunController
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldLauncherModel
    @AppStorage("worlds.advanced") private var advancedOpen = false
    @State private var showUnfinished = false

    private var facts: WorldsGuide.Facts {
        WorldsGuide.Facts(
            folderExists: controller.repoFolderExists, folderProblem: Self.folderSentence(controller), engineReady: controller.repoReady,
            catalog: model.catalog, openWorld: play.isActive ? play.world : nil, openWorldIsStarting: play.phase == .starting,
            saving: model.saving)
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
                    nextStepCard
                    if play.phase == .starting { startingBox }
                    // With nothing to start from the card above says what to do, and that is all the page needs to say.
                    if controller.repoReady, !(model.catalog.worlds.isEmpty && model.catalog.sources.isEmpty) {
                        playSection
                        planetSection.id("planet-form")
                    }
                    if controller.repoReady, !model.catalog.unfinished.isEmpty { unfinishedSection }
                }
                .padding(24)
                .frame(maxWidth: 860, alignment: .leading)
                .frame(maxWidth: .infinity)
            }
            .onChange(of: shell.planetFormRequests) { _, _ in withAnimation { proxy.scrollTo("planet-form", anchor: .top) } }
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .onAppear { refresh() }
        .onChange(of: controller.repoReady) { _, _ in refresh() }
        .onChange(of: controller.runs) { _, _ in refresh() }
        .onChange(of: play.retiredWorlds) { _, _ in refresh() }
        .onChange(of: play.phase) { _, _ in refresh() }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in refresh() }
    }

    private func refresh() { model.refresh(saving: play.retiredWorlds) }

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
                header("flask", "Start with a research run")
                Text("A planet is made from a research run that has a save point, and this alife folder has none yet. Start a small one in the Lab (the preset tiny3s takes about two minutes), then come back and press Make planet.")
                    .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Button("Open the Lab") { shell.select(.lab) }.buttonStyle(.borderedProminent)
            case .makePlanet(let source):
                header("globe.europe.africa", "Make your first planet")
                Text("There is no world to play yet. A planet is made from a research run: \(source) is ready. It takes a few minutes, mostly the weather warming up.")
                    .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                makePlanetButton(prominent: true)
            case .play(let name):
                header("play.circle", "Play \(name)")
                if let world = model.catalog.worlds.first(where: { $0.name == name }) {
                    Text(WorldsGuide.summary(world)).foregroundStyle(.secondary)
                }
                HStack {
                    Button("Play") { if let w = model.catalog.worlds.first(where: { $0.name == name }) { model.resume(w) } }
                        .buttonStyle(.borderedProminent)
                    Text("Opens at 1x. Nothing starts until you press Play.").font(.caption).foregroundStyle(.secondary)
                }
            case .waitForSave:
                header("hourglass", "The worlds are busy")
                Text("Every world is being saved or played in another window. Wait for a save to finish, or make a new planet below.")
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

    // MARK: Play

    private var playSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 0) {
                if model.catalog.worlds.isEmpty {
                    Text("No world to play yet. Make a planet from one of your runs below.")
                        .foregroundStyle(.secondary).padding(8)
                } else {
                    ForEach(model.catalog.worlds) { world in
                        if world.id != model.catalog.worlds.first?.id { Divider() }
                        row(world)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Text("Play").font(.headline)
        }
    }

    private func row(_ world: PlayWorld) -> some View {
        let f = facts
        let reason = WorldsGuide.reason(forPlaying: world, in: f)
        return HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(world.name).font(.body.weight(.medium))
                Text(WorldsGuide.summary(world)).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if let reason { Text(reason).font(.caption).foregroundStyle(.secondary) }
            Button("Play") { model.resume(world) }.disabled(reason != nil)
        }
        .padding(8)
    }

    // MARK: Make a planet

    private var planetSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                PlanetForm(model: model, advancedOpen: $advancedOpen)
                Divider()
                HStack(alignment: .firstTextBaseline) {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(planetReasons, id: \.self) { Text($0).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true) }
                    }
                    Spacer()
                    makePlanetButton(prominent: false)
                }
            }
            .padding(4)
        } label: {
            Text("Make a planet").font(.headline)
        }
    }

    /// The words beside a grey Make planet: the page's reason first, then the form's own.
    private var planetReasons: [String] {
        if let why = WorldsGuide.reasonToMakePlanet(facts) { return [why] }
        return model.planetErrors
    }

    private func makePlanetButton(prominent: Bool) -> some View {
        let reasons = planetReasons
        return HStack {
            if prominent {
                Button("Make planet") { model.startPlanet() }.buttonStyle(.borderedProminent).disabled(!reasons.isEmpty)
                if let first = reasons.first { Text(first).font(.caption).foregroundStyle(.secondary) }
            } else {
                Button("Make planet") { model.startPlanet() }.disabled(!reasons.isEmpty)
            }
        }
    }

    // MARK: Unfinished

    private var unfinishedSection: some View {
        DisclosureGroup(isExpanded: $showUnfinished) {
            VStack(alignment: .leading, spacing: 2) {
                Text("A planet that was cancelled or failed while it was being made. It cannot be played, and nothing is deleted.")
                    .font(.caption).foregroundStyle(.secondary)
                ForEach(model.catalog.unfinished, id: \.self) { Text($0).font(.caption.monospaced()) }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.top, 4)
        } label: {
            Text("Unfinished (\(model.catalog.unfinished.count))").font(.callout).foregroundStyle(.secondary)
        }
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
                if let source = model.selectedSource, source.geometry == "sphere" {
                    note(LauncherPlanetBox.sphereNote(source))
                }
                if model.sourceIsWorld { note("This is a world you can already play. Making a planet from it starts a new branch from one of its save points.") }
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
                    if model.planet.sourceGeometry == "sphere" { note(LauncherPlanetBox.spinupNote) }
                }
                TextField("Name", text: $model.planet.name).textFieldStyle(.roundedBorder)
                DisclosureGroup("Advanced", isExpanded: $advancedOpen) {
                    VStack(alignment: .leading, spacing: 8) {
                        Toggle("This world takes powers (director.enabled)", isOn: $model.planet.directorEnabled)
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

    private func note(_ text: String) -> some View {
        Text(text).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
    }
}
