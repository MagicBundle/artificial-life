import AlifeCore
import Combine
import Foundation

/// The state of the `alife` window that is not a page's own: which page is selected, and what that means for the world. Selecting a
/// page sets `WorldModel.watchActive` (the world's keys, drawing and attention follow the Watch page) and is remembered between launches
/// (never Watch: opening the app starts nothing, R7). A world that goes live brings its page forward.
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
    /// A world is starting or live: what the menus read (they must not observe the play controller, which publishes with every frame).
    @Published private(set) var worldIsOpen: Bool
    /// Counts the requests to scroll the Worlds page to its planet form (File > Make Planet...).
    @Published private(set) var planetFormRequests = 0

    let session: WorldSession
    private let defaults: UserDefaults
    private var phaseObserver: AnyCancellable?
    private var previousPhase: PlayPhase

    init(session: WorldSession, defaults: UserDefaults = .standard) {
        self.session = session
        self.defaults = defaults
        previousPhase = session.play.phase
        worldIsOpen = session.play.isActive
        page = ShellPage.restored(from: defaults.string(forKey: AppShell.pageKey))
        session.model.setWatchActive(page == .watch)
        // A world that goes live is shown, whichever way it was started (Play, Make planet, --world).
        phaseObserver = session.play.$phase.sink { [weak self] phase in
            MainActor.assumeIsolated {
                guard let self else { return }
                if phase == .live, self.previousPhase != .live { self.page = .watch }
                self.previousPhase = phase
                let open = phase == .starting || phase == .live
                if open != self.worldIsOpen { self.worldIsOpen = open }
            }
        }
    }

    func select(_ page: ShellPage) { self.page = page }

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
    /// says why.
    func handleLaunch(arguments: [String], controller: RunController) {
        guard let run = WorldLaunchArgument.parse(arguments) else {
            if arguments.contains(where: { $0 == "--world" || $0.hasPrefix("--world=") }) {
                launchNote = "--world needs the name of a world in the alife folder's runs/ directory."
            }
            return
        }
        if controller.openWorldAtLaunch(run, play: session.play) { return }
        launchNote = !controller.repoReady
            ? "--world \(run): the alife folder is not set up, so the world could not be opened."
            : "--world \(run): there is no run with that name in this alife folder."
    }
}
