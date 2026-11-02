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
            session.model.setWatchActive(page == .watch)
            if page != .watch { defaults.set(page.rawValue, forKey: AppShell.pageKey) }
        }
    }
    /// A note from the launch (`--world RUN` that could not be started), shown on the Worlds page until it is dismissed.
    @Published var launchNote: String?
    /// A world that did not open (it failed at once, or the engine stopped while it started), until the player dismisses it or another
    /// world starts.
    @Published var startFailure: StartFailure?
    /// A world is starting or live: what the menus read (they must not observe the play controller, which publishes with every frame).
    @Published private(set) var worldIsOpen: Bool
    /// Counts the requests to scroll the Worlds page to its planet form (File > Make Planet...).
    @Published private(set) var planetFormRequests = 0
    /// The last request the Worlds page opened the form for (the page may not exist when the request is made).
    var planetFormHandled = 0

    /// Called when a world that was being started ended before it was live: its directory name and a sentence for the Unfinished card
    /// . Set by the window; the library keeps it for the session.
    var buildEnded: ((String, String) -> Void)?

    let session: WorldSession
    private let defaults: UserDefaults
    private var startingWorld: String?
    private var phaseObserver: AnyCancellable?
    private var previousPhase: PlayPhase
    /// The player pressed Play or Make planet on the Worlds page, or `--world` started a world: the next world to go live is theirs and
    /// is shown. (A world is never pulled in front of someone who went on to the Lab or Activity while it started.)
    private var startedByThePlayer = false
    private var startedByArgument = false
    private var launchHandled = false

    init(session: WorldSession, defaults: UserDefaults = .standard) {
        self.session = session
        self.defaults = defaults
        previousPhase = session.play.phase
        worldIsOpen = session.play.isActive
        page = ShellPage.restored(from: defaults.string(forKey: AppShell.pageKey))
        session.model.setWatchActive(page == .watch)
        phaseObserver = session.play.$phase.sink { [weak self] phase in
            MainActor.assumeIsolated { self?.phaseChanged(to: phase) }
        }
    }

    private func phaseChanged(to phase: PlayPhase) {
        let was = previousPhase
        previousPhase = phase
        let open = phase == .starting || phase == .live
        if open != worldIsOpen { worldIsOpen = open }
        switch phase {
        case .starting:
            startFailure = nil
            startingWorld = session.play.world
        case .live:
            startFailure = nil
            startingWorld = nil
            if was != .live, startedByArgument || (startedByThePlayer && page == .worlds) { page = .watch }
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

    func select(_ page: ShellPage) { self.page = page }

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
