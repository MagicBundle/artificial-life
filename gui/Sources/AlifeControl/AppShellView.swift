import AlifeCore
import AppKit
import SwiftUI

/// The one window of `alife` (docs/APP.md §2.4): a sidebar of pages and the page beside it. The Watch layer (the map and its Metal
/// view) is always in the hierarchy under the pages, so it is made once and kept; the pages above it come and go, and each brings its own
/// title and toolbar. The window has one minimum size for every page, so that entering Watch never resizes it.
struct AppShellView: View {
    @ObservedObject var shell: ShellModel
    @ObservedObject var controller: RunController
    @ObservedObject var play: PlayController
    @ObservedObject var stream: StreamController
    @ObservedObject var model: WorldModel
    @ObservedObject var library: WorldLibrary
    @StateObject private var setup: SetupModel
    @Environment(\.dismissWindow) private var dismissWindow
    @Environment(\.openWindow) private var openWindow
    private let arguments: [String]

    /// `setup` is the app's set-up model (shared with the set-up window); without one the view makes its own (the tests').
    init(shell: ShellModel, controller: RunController, library: WorldLibrary, setup: SetupModel? = nil, stream: StreamController = .shared,
         arguments: [String] = CommandLine.arguments) {
        self.shell = shell
        self.controller = controller
        self.library = library
        play = shell.session.play
        self.stream = stream
        model = shell.session.model
        self.arguments = arguments
        let play = shell.session.play
        _setup = StateObject(wrappedValue: setup ?? SetupModel.forApp(controller: controller, play: play, library: library))
    }

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 190, ideal: 215, max: 280)
        } detail: {
            VStack(spacing: 0) {
                // The pipeline's bar is on every page but the one that shows it in full: the wait is never out of sight.
                if showsStrip {
                    PipelineStrip(coordinator: setup.coordinator) { shell.select(.setup) }
                }
                ZStack {
                    WatchLayer(play: play, model: model, visible: shell.page == .watch)
                    page
                }
            }
        }
        .frame(minWidth: AppShell.minWidth, minHeight: AppShell.minHeight)
        // The 3D viewer window (until PR-F) goes with the main window, so that closing the main window ends the app as it always did.
        // The window is the app's: closing it is closing the last window (the Settings window and the 3D Viewer go with it, so that the app
        // ends through the question and the wait of the quit, as the old apps do).
        .background(WindowCloseObserver(group: "main") {
            dismissWindow(id: AppWindow.viewer3d.rawValue)
            AppDelegate.current?.mainWindowClosed()
        })
        .onAppear {
            // A click on the Dock icon brings a closed window back (`AppDelegate.applicationShouldHandleReopen`).
            AppDelegate.current?.reopenMainWindow = { openWindow(id: AppShell.mainWindowID) }
            // A world that stopped while it was starting is an unfinished build; the library keeps what this session knows of it.
            shell.buildEnded = { [weak library] world, note in library?.note(note, forBuild: world) }
            controller.pollsRunList = shell.page == .setup
            // The set-up page does not continue a played world, or a directory another process or this window is writing.
            controller.resumeRefusal = { [weak library] name in library?.reasonNotToResume(name) }
            wireSetup()
            library.update(live: liveFacts)
            applyCatalog()
            library.refresh()
            library.startTimer()
        }
        // The library follows the folder, what this app does with its worlds, and the planet form follows the library.
        .onChange(of: controller.repoReady) { _, _ in library.refresh() }
        .onChange(of: controller.repoPath) { _, _ in library.refresh() }
        .onChange(of: liveFacts) { _, facts in library.update(live: facts) }
        .onChange(of: library.state.catalog) { _, _ in applyCatalog() }
        .onChange(of: play.retiredWorlds) { _, _ in applyCatalog() }
        .onChange(of: setup.coordinator.pendingPlanets) { _, _ in library.update(live: liveFacts) }
        // The set-up page checks names and sources against the run list, which is polled only while it is shown; the library scans on its own.
        .onChange(of: shell.page) { _, page in
            controller.pollsRunList = page == .setup
            if page == .setup { controller.refreshRuns() }
        }
        .task {
            // `--world RUN` (a fresh launch): started once, through the same path as Play.
            shell.handleLaunch(arguments: arguments, controller: controller)
        }
    }

    /// The pipeline's bar is on every page but the one that shows it in full, and not on Watch once the planet it made is there to see.
    private var showsStrip: Bool {
        guard setup.coordinator.banner != nil, shell.page != .setup else { return false }
        return !(shell.page == .watch && setup.coordinator.plan?.state == .live)
    }

    @ViewBuilder private var page: some View {
        switch shell.page {
        case .watch:
            WatchChrome(shell: shell, play: play, model: model)
        case .worlds:
            WorldsPage(shell: shell, controller: controller, play: play, setup: setup, library: library)
                .navigationTitle("Worlds")
        case .activity:
            ActivityPage(shell: shell, controller: controller, play: play, stream: stream)
                .navigationTitle("Activity")
        case .setup:
            SetupPage(model: setup, controller: controller, coordinator: setup.coordinator, library: library, launcher: setup.launcher)
        }
    }

    // MARK: Sidebar

    private var sidebar: some View {
        List(selection: sidebarSelection) {
            ForEach(ShellPage.allCases) { page in
                let row = rowText(of: page)
                SidebarRow(page: page, detail: row.text, marked: row.marked)
                    .tag(SidebarItem.page(page))
                    .help(page.purpose)
            }
            LibrarySidebarSection(library: library)
        }
        .safeAreaInset(edge: .bottom) { engineStatus }
    }

    /// The page, or the world that is chosen on the Worlds page. A row that opens a list (all the worlds, the unfinished builds) is a
    /// button more than a place: choosing it shows the list and the selection goes back to the page.
    private var sidebarSelection: Binding<SidebarItem?> {
        Binding(
            get: { SidebarItem.current(page: shell.page, library: library) },
            set: { item in item?.choose(shell: shell, library: library) })
    }

    /// What this app is doing with its worlds, for the library's states (no file shows it). Cheap: it is read whenever the model changes.
    private var liveFacts: LiveFacts {
        var facts = LiveFacts()
        if play.isActive, let world = play.world {
            facts.openWorld = world
            if play.phase == .starting {
                facts.openPhase = .starting
            } else {
                let stopped = play.requestedRate.map { $0 == .multiplier(0) } ?? (play.store.clock?.rate == 0)
                facts.openPhase = stopped ? .paused : .live
            }
        }
        facts.growing = controller.runningRunName
        facts.saving = Set(play.retiredWorlds)
        facts.pending = setup.coordinator.pendingPlanets
        return facts
    }

    /// The planet form's catalog is the library's (the scan ran off the main thread), and the worlds still saving are the play controller's.
    private func applyCatalog() {
        setup.launcher.apply(catalog: library.state.catalog, saving: Set(play.retiredWorlds))
    }

    /// What the pipeline needs of the window: where to show things, and what other processes hold (from the library).
    private func wireSetup() {
        setup.show = { [weak shell] in shell?.select(.setup) }
        setup.coordinator.hooks.playerStartsWorld = { [weak shell] in shell?.playerStartsWorld() }
        setup.coordinator.hooks.showActivity = { [weak shell] in shell?.select(.activity) }
        setup.coordinator.hooks.showWatch = { [weak shell] in shell?.select(.watch) }
        setup.coordinator.hooks.dismissStartFailure = { [weak shell] in shell?.startFailure = nil }
        setup.coordinator.hooks.openSetup = { [weak setup] intent, suggested in
            switch intent {
            case .create: setup?.openCreate()
            case .branch(let w): setup?.openBranch(from: w)
            case .keepGrowing(let w): setup?.openKeepGrowing(world: w)
            case .makePlanet(let w): setup?.openMakePlanet(from: w, name: suggested)
            }
        }
        setup.coordinator.elsewhere = { [weak library] in
            let rows = library?.state.rows ?? []
            return Set(rows.filter { $0.state == .inUseElsewhere }.map(\.name))
        }
    }

    /// The second line of a sidebar row, and whether the row is marked (something there waits for the player).
    private func rowText(of page: ShellPage) -> (text: String?, marked: Bool) {
        switch page {
        case .setup:
            guard let banner = setup.coordinator.banner else { return (nil, false) }
            return (banner.title, banner.tone == .attention || banner.tone == .failure)
        case .worlds:
            if shell.startFailure != nil { return ("A world did not open", true) }
            return (nil, false)
        case .watch:
            let cue = model.watchCue(startFailed: shell.startFailure != nil)
            return (cue.text, cue.needsPlayer)
        case .activity:
            return (ActivityGuide.sidebarDetail(ActivityPage.facts(controller: controller, play: play, stream: stream)), false)
        }
    }

    private var engineStatus: some View {
        HStack(spacing: 6) {
            Circle().fill(controller.repoReady ? Color.green : Color.orange).frame(width: 8, height: 8)
            Text(controller.repoReady ? "Engine ready" : (controller.repoFolderExists ? "Engine not set up" : "alife folder not found"))
                .font(.caption).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .help(controller.repoReady ? controller.repoPath : "Open Worlds or Settings to fix it.")
    }
}

private struct SidebarRow: View {
    let page: ShellPage
    let detail: String?
    var marked = false

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            HStack(spacing: 6) {
                Label(page.title, systemImage: page.symbol)
                if marked {
                    Circle().fill(Color.orange).frame(width: 8, height: 8)
                        .accessibilityLabel("Waiting for you")
                }
            }
            if let detail {
                Text(detail).font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle).padding(.leading, 26)
            }
        }
        .padding(.vertical, 2)
    }
}
