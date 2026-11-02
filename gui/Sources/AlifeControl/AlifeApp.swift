import AlifeCore
import AppKit
import SwiftUI

/// The entry point: the preview's and the old apps' preferences are brought in once, before anything reads the folder setting, and the app starts.
/// `--launch-mode X` of the old apps is accepted and ignored (`AppPlan.ignoredArgument`): nothing reads it.
@main
enum AlifeEntry {
    static func main() {
        // An `ALIFE_PREFS_SUITE` that names the app's own domain or an older app's would be ignored or would write that domain: refuse.
        if let problem = RepoPathStore.overrideProblem(environment: ProcessInfo.processInfo.environment) {
            FileHandle.standardError.write(Data(("alife: " + problem + "\n").utf8))
            exit(2)
        }
        // Only a bundle made by make-app.sh has its own preferences domain (the one the old apps shared): an unbundled run
        // (`swift run`) keeps whatever it had, as it always did.
        if Bundle.main.object(forInfoDictionaryKey: AppPlan.infoKey) != nil { PreferenceMigration.forApp()?.run() }
        AlifeApp.main()
    }
}

/// `alife`: one window, a sidebar of pages (docs/APP.md §2.4). It replaced the research Lab and Storm Worlds (PR-H).
struct AlifeApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var controller = RunController.shared
    @StateObject private var shell = ShellModel(session: .shared)
    @StateObject private var library: WorldLibrary
    @StateObject private var setup: SetupModel
    @StateObject private var switcher: WorldSwitcher
    /// The Dock icon's progress bar for a run, and the badge and bounce when it ends (D3).
    @StateObject private var dock: DockTileController
    /// The observer of the research form of Watch, the readers of a world's files (the Run panel's and Details'), Settings > Autopilot and Help.
    @StateObject private var observer: ObserverSession
    @StateObject private var watchReader: WorldReader
    @StateObject private var detailsReader: WorldReader
    @StateObject private var autopilot: AutopilotSettings
    @StateObject private var help = HelpModel.shared
    private let appName = (Bundle.main.object(forInfoDictionaryKey: "CFBundleName") as? String) ?? AppPlan.name

    init() {
        // (the set-up model reads the library, so both are made here and handed to the scenes)
        let library = WorldLibrary.forApp(run: .shared)
        _library = StateObject(wrappedValue: library)
        let setup = SetupModel.forApp(controller: .shared, play: PlayController.shared, library: library)
        _setup = StateObject(wrappedValue: setup)
        _switcher = StateObject(wrappedValue: WorldSwitcher.forApp(controller: .shared, play: PlayController.shared))
        let controller = RunController.shared
        _observer = StateObject(wrappedValue: ObserverSession(controller: controller))
        let runs: () -> URL? = { RunController.shared.repoReady ? RunController.shared.repoURL.appendingPathComponent("runs", isDirectory: true) : nil }
        _watchReader = StateObject(wrappedValue: WorldReader(runsDirectory: runs))
        _detailsReader = StateObject(wrappedValue: WorldReader(runsDirectory: runs))
        let autopilot = AutopilotSettings(controller: controller)
        _autopilot = StateObject(wrappedValue: autopilot)
        controller.autopilotDefaults = { director in autopilot.apply(to: &director) }
        // The run's target: the pipeline's when it grows the run, else the `run.steps` of the run's own config.
        let coordinator = setup.coordinator
        _dock = StateObject(wrappedValue: DockTileController(controller: controller, surface: SystemDockTile()) {
            if let plan = coordinator.plan, case .growing = plan.state, let target = plan.targetSteps { return target }
            guard let run = controller.runningRunName, controller.repoReady else { return nil }
            return DockTileController.configTarget(run: run, repo: controller.repoURL)
        })
    }

    var body: some Scene {
        Window(appName, id: AppShell.mainWindowID) {
            AppShellView(shell: shell, controller: controller, library: library, setup: setup, switcher: switcher, observer: observer,
                         watchReader: watchReader, detailsReader: detailsReader)
        }
        .windowResizability(.contentMinSize)
        .defaultSize(width: 1280, height: 800)
        .commands {
            QuitCommands(appName: appName)
            ShellCommands(shell: shell, controller: controller, library: library, setup: setup)
            WorldCommands(shell: shell, observer: observer, controller: controller, library: library, setup: setup)
            HelpCommands(help: help)
        }

        // The set-up page in a window of its own, so that a form can be composed beside a Watch (docs/APP.md §3.7). One instance.
        Window("Set up", id: AppShell.setupWindowID) {
            SetupWindowView(setup: setup, controller: controller, library: library)
        }
        .windowResizability(.contentMinSize)
        .defaultSize(width: 820, height: 760)

        // Help: short topics, and World Controls (the key tables).
        Window("alife Help", id: AppShell.helpWindowID) {
            HelpView(model: help, shell: shell)
        }
        .windowResizability(.contentMinSize)
        .defaultSize(width: 860, height: 600)

        Settings {
            GeneralSettingsView(controller: controller, autopilot: autopilot)
        }
    }
}

/// The set-up page in its window: the same model as the page of the main window.
struct SetupWindowView: View {
    @ObservedObject var setup: SetupModel
    @ObservedObject var controller: RunController
    @ObservedObject var library: WorldLibrary

    var body: some View {
        SetupPage(model: setup, controller: controller, coordinator: setup.coordinator, library: library, launcher: setup.launcher, inWindow: true)
            .frame(minWidth: 640, minHeight: 560)
    }
}

/// The menu bar of `alife`: File (new world, make a planet, close the world, reveal in Finder) and the Window menu's pages with their Command keys.
/// It observes the shell only (not the play controller, which publishes with every frame: a menu rebuilt that often is unusable).
struct ShellCommands: Commands {
    @ObservedObject var shell: ShellModel
    let controller: RunController
    let library: WorldLibrary
    let setup: SetupModel

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("New World…") { setup.openCreate() }
                .keyboardShortcut("n")
            Button("Make Planet…") { shell.requestPlanetForm() }
            Divider()
            Button("Close World") { shell.closeWorld() }
                .disabled(!shell.worldIsOpen)
            Button("Reveal in Finder") {
                // The world that is chosen while the Worlds page is in view (a world that is gone is told, and no other folder is shown in
                // its place); on any other page the run that is chosen, else the alife folder.
                Task { await library.revealFromMenu(page: shell.page) { controller.revealInFinder() } }
            }
                .keyboardShortcut("r", modifiers: [.command, .shift])
        }
        CommandGroup(before: .windowList) {
            ForEach(ShellPage.allCases.sorted { $0.key < $1.key }) { page in
                Button(page.title) { shell.select(page) }
                    .keyboardShortcut(KeyEquivalent(page.key), modifiers: .command)
            }
            Divider()
        }
    }
}


/// The World menu (docs/APP.md §2.4): the actions of the Watch in view, listed from its key table (`ControlTables`). The menu follows the form:
/// a played world's items, or an observer's. A letter is never a key equivalent (it would fire while a text field has the focus): the key is
/// text in the title, and the keys work on the Watch page itself. The items act only while the Watch page is shown. It observes the shell only
/// (not the play controller or the stream, which publish with every frame: a menu rebuilt that often is unusable).
struct WorldCommands: Commands {
    @ObservedObject var shell: ShellModel
    let observer: ObserverSession
    let controller: RunController
    let library: WorldLibrary
    let setup: SetupModel

    var body: some Commands {
        CommandMenu("World") {
            let form = shell.watchForm
            let enabled = Self.itemsAreEnabled(page: shell.page, form: form)
            if let form {
                Section(form == .played ? "Playing" : "Observing") {
                    ForEach(ControlTables.menu(form)) { entry in
                        Button(entry.menuTitle) { perform(entry) }.disabled(!enabled)
                    }
                }
            } else {
                Text("Nothing is open in Watch").disabled(true)
            }
            if form == .research, let world = shell.observed {
                Divider()
                // Stop ends the run that Watch shows, from Watch only: a key that ended a run from another page would be a trap.
                Button("Stop \(DetailsGuide.title(world))") { controller.stop() }
                    .keyboardShortcut(".", modifiers: .command)
                    .disabled(!Self.canStop(page: shell.page, form: form, observedGrowing: shell.observedGrowing))
                Button("Check World") { check(world) }
                    .disabled(!enabled || !Self.canCheck(runGoing: shell.runningRun != nil, repoReady: controller.repoReady))
                Button("Open Classic 2D Viewer") { controller.openClassicViewer(world: world) }
                    .disabled(!enabled || !Self.canOpenClassic(shape: shell.observedShape, repoReady: controller.repoReady))
            }
        }
    }

    private func perform(_ entry: ControlEntry) {
        Self.dispatch(entry, world: { shell.session.model.perform($0) }, observer: { observer.viewer.perform($0) })
    }

    /// A menu item does what its key does, in its own form: a played world's action goes to the world, an observer's to the viewer.
    static func dispatch(_ entry: ControlEntry, world: (WorldAction) -> Void, observer: (ViewerAction) -> Void) {
        switch entry.action {
        case .world(let action)?: world(action)
        case .observer(let action)?: observer(action)
        case nil: break
        }
    }

    /// Stop ends a run this app grows; Check World needs the engine and no job; the classic viewer shows flat worlds only.
    static func canStop(page: ShellPage, form: WatchForm?, observedGrowing: Bool) -> Bool { page == .watch && form == .research && observedGrowing }
    static func canCheck(runGoing: Bool, repoReady: Bool) -> Bool { !runGoing && repoReady }
    static func canOpenClassic(shape: WorldShape?, repoReady: Bool) -> Bool { shape == .flat && repoReady }

    /// Whether the World menu's items work: only on the Watch page, and only when Watch shows something.
    static func itemsAreEnabled(page: ShellPage, form: WatchForm?) -> Bool { page == .watch && form != nil }

    private func check(_ world: String) {
        guard controller.repoReady, controller.job == nil else {
            shell.launchNote = controller.repoReady ? "Another job is running. Wait for it to finish, then check \(world)." : "Set up the engine first."
            shell.select(.worlds)
            return
        }
        controller.inspect(world: world)
        shell.select(.activity)
    }
}

/// The Help menu: alife Help, and World Controls, which opens the same window on the page with the key tables.
struct HelpCommands: Commands {
    let help: HelpModel
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        CommandGroup(replacing: .help) {
            Button("alife Help") { open(HelpTopics.all[0].id) }
            Button("World Controls") { open(HelpTopics.controlsID) }
        }
    }

    private func open(_ topic: String) {
        help.show(topic)
        openWindow(id: AppShell.helpWindowID)
    }
}
