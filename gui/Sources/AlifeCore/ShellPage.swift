import Foundation

/// The pages of the `alife` window (docs/APP.md §2.4, PR-B). Names follow §2.5: *Watch* is the live picture of a world, *Worlds*
/// is where a world is chosen, *Activity* shows what is running, and *New world* is the set-up page (PR-D): Create, Branch, Keep growing
/// and Make planet. It replaced the Lab and the launcher's planet form.
package enum ShellPage: String, CaseIterable, Identifiable, Sendable {
    case worlds
    case watch
    case activity
    case setup
    /// The facts of one world (docs/APP.md §2.4): what it is, how it is doing and what can be done with it. It has no row of its own in the
    /// sidebar: a world's row opens it.
    case details

    /// The pages that have a row in the sidebar, in order; the world rows open `details`.
    package static let sidebarPages: [ShellPage] = [.worlds, .watch, .activity, .setup]

    package var id: String { rawValue }

    package var title: String {
        switch self {
        case .worlds: return "Worlds"
        case .watch: return "Watch"
        case .activity: return "Activity"
        case .setup: return "New world"
        case .details: return "Details"
        }
    }

    /// An SF Symbol.
    package var symbol: String {
        switch self {
        case .worlds: return "globe"
        case .watch: return "eye"
        case .activity: return "list.bullet.rectangle"
        case .setup: return "plus.circle"
        case .details: return "info.circle"
        }
    }

    /// The Command key of the Window-menu item that opens the page: in the order of the sidebar (Worlds 1, Watch 2, Activity 3, New world 4, Details 5;
    /// the spec's §2.4 had Activity on 2, before Watch was a page).
    package var key: Character {
        switch self {
        case .worlds: return "1"
        case .watch: return "2"
        case .activity: return "3"
        case .setup: return "4"
        case .details: return "5"
        }
    }

    /// What the page is for, one line (the sidebar's tooltip and the Window menu's help).
    package var purpose: String {
        switch self {
        case .worlds: return "Choose a world to play, or make a new planet."
        case .watch: return "The live picture of the world you are playing, or the observer picture of a research world."
        case .activity: return "What is running now, and the log."
        case .setup: return "Make a new world, branch one, keep one growing, or make a planet."
        case .details: return "What a world is, how it is doing, and what you can do with it."
        }
    }

    /// The page the app opens on after the last session. Watch is never restored: opening the app starts nothing (R7), and its world
    /// is not open.
    package static func restored(from raw: String?) -> ShellPage {
        guard let raw, let page = ShellPage(rawValue: raw), page != .watch, page != .details else { return .worlds }
        return page
    }
}

/// Names the one window of `alife` (a `Window` scene), outside `AppWindow`, which belongs to the two old modes.
package enum AppShell {
    package static let mainWindowID = "main"
    /// The set-up page in a window of its own (docs/APP.md §3.7), so that a form can be composed beside a Watch.
    package static let setupWindowID = "setup"
    /// Help in a window of its own.
    package static let helpWindowID = "help"
    /// The preference that remembers the page between launches (`ShellPage.restored`).
    package static let pageKey = "shell.page"
    /// What closing the window does while a run is going (`ClosePreference`): ask, keep it running, or stop it and quit (docs/APP.md D3).
    package static let closeWithRunKey = "shell.closeWithRun"
    /// Whether a world paused by leaving Watch continues when Watch is shown again (*Pick up where I left off*, docs/APP.md D4).
    package static let pickUpKey = "shell.pickUp"
    /// The address and the name of the local language model the autopilot may ask (Settings > Autopilot): each new world starts from them.
    package static let autopilotURLKey = "shell.autopilotURL"
    package static let autopilotModelKey = "shell.autopilotModel"
    /// The minimum size of the window for every page (docs/APP.md M12), so that entering Watch never resizes it.
    package static let minWidth: Double = 1100
    package static let minHeight: Double = 700
}
