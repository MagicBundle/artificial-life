import AlifeCore
import Combine
import Foundation

/// A world that did not open, for the Worlds page to say (review H1): the page the player is on when they press Play is Worlds, and
/// the message used to exist only on the Watch canvas and in the log.
struct StartFailure: Equatable {
    var world: String
    var message: String

    /// The sentence in plain words (`WorldsGuide.explainFailure`).
    var explanation: String { WorldsGuide.explainFailure(world: world, message: message) }
}

/// The state of the `alife` window that is not a page's own: which page is selected, and what that means for the world. Selecting a
/// page sets `WorldModel.watchActive` (the world's keys, drawing and attention follow the Watch page) and is remembered between launches
/// (never Watch: opening the app starts nothing, R7). A world that goes live brings Watch forward only when the player's own action
/// started it on the Worlds page, or `--world` did.
@MainActor
final class ShellModel: ObservableObject {
    @Published var page: ShellPage {
        didSet {
            guard page != oldValue else { return }
            refreshWatchActive()
            if page != .watch { defaults.set(page.rawValue, forKey: AppShell.pageKey) }
        }
    }
    /// A note from the launch (`--world RUN` that could not be started), shown on the Worlds page until it is dismissed.
    @Published var launchNote: String?
    /// A world that did not open (it failed at once, or the engine stopped while it started), until the player dismisses it or another
    /// world starts.
    @Published var startFailure: StartFailure?
    /// What became of the world that was open when the window was closed with a run going (it was saved and closed): the Worlds page says so
    /// until the player dismisses it or opens a world.
    @Published var closedNote: String?
    /// The app's own pause of the world while the player is on another page (`AwayPause`): what the Worlds page and the sidebar say.
    @Published private(set) var awayHold = AwayPause.Hold.none
    /// A world is starting or live: what the menus read (they must not observe the play controller, which publishes with every frame).
    @Published private(set) var worldIsOpen: Bool
    /// Counts the requests to scroll the Worlds page to its planet form (File > Make Planet...).
    @Published private(set) var planetFormRequests = 0
    /// The last request the Worlds page opened the form for (the page may not exist when the request is made).
    var planetFormHandled = 0
    /// The research world Watch shows (the observer form) when no played world is open; nil when none was chosen. A world that goes live clears it.
    @Published private(set) var observed: String?
    /// Which Watch the page shows (docs/APP.md §3.4): the World menu, the keys and Help follow it. Nil when there is nothing to watch.
    @Published private(set) var watchForm: WatchForm?
    /// This app grows a research run now, and the shape of the world Watch observes: what the World menu needs without watching the controllers
    /// (they publish with every frame). Set by the window.
    @Published var runningRun: String?
    @Published private(set) var observedShape: WorldShape?
    /// This app grows the world Watch observes (the World menu's Stop is for that run and no other).
    var observedGrowing: Bool {
        guard let run = runningRun, let observed else { return false }
        return NameFolding.same(run, observed)
    }
    /// The shape of a world by name, from the library (set by the window).
    var shapeOf: (String) -> WorldShape? = { _ in nil }

    /// Called when a world that was being started ended before it was live: its directory name and a sentence for the Unfinished card
    /// . Set by the window; the library keeps it for the session.
    var buildEnded: ((String, String) -> Void)?

    let session: WorldSession
    /// Welcome: `uv`, the engine's answer and whether the card was shown. The Worlds page reads it; its changes are this object's.
    let welcome: WelcomeModel
    private let defaults: UserDefaults
    private var welcomeObserver: AnyCancellable?
    private var startingWorld: String?
    private var phaseObserver: AnyCancellable?
    private var awayObserver: AnyCancellable?
    private var retiredObserver: AnyCancellable?
    /// The world whose save the closed-window note waits for, and the number of ends seen before it was asked to stop.
    private var closedWorld: (world: String, after: Int)?
    private var previousPhase: PlayPhase
    /// The player pressed Play or Make planet on the Worlds page, or `--world` started a world: the next world to go live is theirs and
    /// is shown. (A world is never pulled in front of someone who went on to Activity while it started; the set-up page counts as the page it was started from, since it shows the wait.)
    private var startedByThePlayer = false
    private var startedByArgument = false
    private var launchHandled = false

    init(session: WorldSession, defaults: UserDefaults = .standard, welcome: WelcomeModel = WelcomeModel()) {
        self.session = session
        self.defaults = defaults
        self.welcome = welcome
        previousPhase = session.play.phase
        worldIsOpen = session.play.isActive
        page = ShellPage.restored(from: defaults.string(forKey: AppShell.pageKey))
        watchForm = Self.form(of: session.play.phase, observed: nil)
        refreshWatchActive()
        session.model.usesOverviewWord = true  // (the hint on the map says Overview, as the toolbar does; the two old apps say Home)
        session.model.plainNames = true  // (a planet made from "planet-1-grow" is "made from its grown world")
        phaseObserver = session.play.$phase.sink { [weak self] phase in
            MainActor.assumeIsolated { self?.phaseChanged(to: phase) }
        }
        retiredObserver = session.play.$lastRetiredEnd.sink { [weak self] end in
            MainActor.assumeIsolated { self?.retiredEnded(end) }
        }
        welcomeObserver = welcome.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
        awayHold = session.model.awayHold
        awayObserver = session.model.$awayHold.removeDuplicates().sink { [weak self] hold in
            MainActor.assumeIsolated { self?.awayHold = hold }
        }
    }

    /// Which Watch a play phase and an observed world make (`WatchGuide.form`).
    static func form(of phase: PlayPhase, observed: String?) -> WatchForm? {
        var ended = false
        if case .ended = phase { ended = true }
        if case .failed = phase { ended = true }
        return WatchGuide.form(worldOpen: phase == .starting || phase == .live, worldEnded: ended, observed: observed)
    }

    /// The played world's keys, drawing and attention follow the Watch page *and* the played form: with a research world in view the world
    /// model has nothing on the page.
    /// `phase` is the phase being announced: the play controller's `@Published` phase publishes *before* it changes, so the property still holds
    /// the old one when the observer runs (the form lagged a phase behind until another publish came).
    private func refreshWatchActive(phase: PlayPhase? = nil) {
        let form = Self.form(of: phase ?? session.play.phase, observed: observed)
        if form != watchForm { watchForm = form }
        session.model.setWatchActive(page == .watch && form != .research)
    }

    /// Watch shows `world` as an observer (a research world): the card's or Details' Watch, or a run just started. A played world that is
    /// open is what Watch shows, so the caller has checked `WatchGuide.reasonToWatch`.
    func observe(_ world: String) {
        guard !session.play.isActive else { return }
        observed = world
        observedShape = shapeOf(world)
        refreshWatchActive()
        select(.watch)
    }

    /// A research run was started from the set-up page: Watch follows it, and the person who pressed the button is taken there. Nothing moves
    /// a person who went on to another page, and a world that is open stays what Watch shows.
    func researchStarted(_ world: String) {
        // A run started while a world is open: the world stays what Watch shows (one picture at a time, R1), and the run is what Watch shows when
        // the world is closed. It is never lost: Details, Activity and the bar name it.
        observed = world
        observedShape = shapeOf(world)
        refreshWatchActive()
        if page == .setup, !session.play.isActive { select(.watch) }
    }

    private func phaseChanged(to phase: PlayPhase) {
        let was = previousPhase
        previousPhase = phase
        let open = phase == .starting || phase == .live
        if open != worldIsOpen { worldIsOpen = open }
        // A world that opens is what Watch shows; the research world seen before is not brought back behind it.
        if phase == .starting { observed = nil }
        refreshWatchActive(phase: phase)
        switch phase {
        case .starting:
            startFailure = nil
            closedNote = nil
            startingWorld = session.play.world
        case .live:
            startFailure = nil
            closedNote = nil
            startingWorld = nil
            if was != .live, startedByArgument || (startedByThePlayer && (page == .worlds || page == .setup)) { page = .watch }
            startedByThePlayer = false
            startedByArgument = false
        case .failed(let message):
            // A world that was live and then died is told on its Watch page; one that never got live is told where the player is.
            if was != .live {
                startFailure = StartFailure(world: session.play.world ?? "", message: message)
                if let name = session.play.world ?? startingWorld { buildEnded?(name, ShellModel.buildNote(failed: true)) }
            }
            startedByThePlayer = false
            startedByArgument = false
            startingWorld = nil
        case .idle, .ended:
            if was == .starting, let name = startingWorld { buildEnded?(name, ShellModel.buildNote(failed: false)) }
            startedByThePlayer = false
            startedByArgument = false
            startingWorld = nil
        }
    }

    /// Shows a page. With the window closed (the app lives on for a run) a command that names a page brings the window back first.
    func select(_ page: ShellPage) {
        self.page = page
        AppDelegate.current?.showMainWindowIfNeeded()
    }

    /// The window was closed with a run going and stays closed. The world that was open is being saved: the note says "being saved" until its
    /// process has gone, and then what happened (`RetiredEnd`); a world that was still starting was stopped and nothing of it was saved. The
    /// player comes back to the page they left, except Watch, which has no world to show any more (Worlds, with the note).
    func windowClosedWithRun(world: String?, starting: Bool = false) {
        if let world, !world.isEmpty {
            closedNote = SessionRules.closedWorldNote(world, starting ? .neverOpened : .saving)
            closedWorld = starting ? nil : (world, session.play.lastRetiredEnd?.serial ?? 0)
            if page == .watch { page = .worlds }
        }
    }

    /// The process of a world that was asked to stop has gone: the note says how it ended.
    func retiredEnded(_ end: RetiredEnd?) {
        guard let end, let waiting = closedWorld, NameFolding.same(waiting.world, end.world), end.serial > waiting.after else { return }
        closedWorld = nil
        closedNote = SessionRules.closedWorldNote(end.world, end.saved ? .saved : .saveFailed(end.reason))
    }

    /// A switch the window being closed had waiting was dropped: A is closed (saved), B was not opened.
    func switchDropped(_ note: String) { closedNote = note }

    /// What the Unfinished card says about a world that was being started when it ended in this session.
    static func buildNote(failed: Bool) -> String {
        failed ? "It failed" : "You cancelled it"
    }

    /// The player pressed Play or Make planet on the Worlds page: the world they start is shown when it is live.
    func playerStartsWorld() {
        startedByThePlayer = true
        startFailure = nil
    }

    /// Close World: the world saves first (`PlayController.stop`), and the Worlds page is the place to choose the next.
    func closeWorld() {
        session.play.stop()
        page = .worlds
    }

    func requestPlanetForm() {
        page = .worlds
        planetFormRequests += 1
    }

    /// `--world RUN` at launch: starts RUN as Play does. The shell shows Watch once it is live; when it cannot be started the Worlds page
    /// says why. Once only: this object lives as long as the app, so a rebuilt window never starts the world again.
    func handleLaunch(arguments: [String], controller: RunController) {
        guard !launchHandled else { return }
        launchHandled = true
        guard let run = WorldLaunchArgument.parse(arguments) else {
            if arguments.contains(where: { $0 == "--world" || $0.hasPrefix("--world=") }) {
                launchNote = "--world needs the name of a world in the alife folder's runs/ directory."
            }
            return
        }
        guard !session.play.isActive else { return }
        startedByArgument = true
        if controller.openWorldAtLaunch(run, play: session.play) { return }
        startedByArgument = false
        launchNote = !controller.repoReady
            ? "--world \(run): the alife folder is not set up, so the world could not be opened."
            : "--world \(run): there is no run with that name in this alife folder."
    }
}
