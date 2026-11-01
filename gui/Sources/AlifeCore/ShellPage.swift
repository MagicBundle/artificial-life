import Foundation

/// The pages of the `alife` window (docs/APP.md §2.4, PR-B). Names follow §2.5: *Watch* is the live picture of a world, *Worlds*
/// is where a world is chosen, *Activity* shows what is running, and *New world* is the set-up page (PR-D): Create, Branch, Keep growing
/// and Make planet. It replaced the Lab and the launcher's planet form.
package enum ShellPage: String, CaseIterable, Identifiable, Sendable {
    case worlds
    case watch
    case activity
    case setup

    package var id: String { rawValue }

    package var title: String {
        switch self {
        case .worlds: return "Worlds"
        case .watch: return "Watch"
        case .activity: return "Activity"
        case .setup: return "New world"
        }
    }

    /// An SF Symbol.
    package var symbol: String {
        switch self {
        case .worlds: return "globe"
        case .watch: return "eye"
        case .activity: return "list.bullet.rectangle"
        case .setup: return "plus.circle"
        }
    }

    /// The Command key of the Window-menu item that opens the page: in the order of the sidebar (Worlds 1, Watch 2, Activity 3, New world 4;
    /// the spec's §2.4 had Activity on 2, before Watch was a page).
    package var key: Character {
        switch self {
        case .worlds: return "1"
        case .watch: return "2"
        case .activity: return "3"
        case .setup: return "4"
        }
    }

    /// What the page is for, one line (the sidebar's tooltip and the Window menu's help).
    package var purpose: String {
        switch self {
        case .worlds: return "Choose a world to play, or make a new planet."
        case .watch: return "The live picture of the world that is open."
        case .activity: return "What is running now, and the log."
        case .setup: return "Make a new world, branch one, keep one growing, or make a planet."
        }
    }

    /// The page the app opens on after the last session. Watch is never restored: opening the app starts nothing (R7), and its world
    /// is not open.
    package static func restored(from raw: String?) -> ShellPage {
        guard let raw, let page = ShellPage(rawValue: raw), page != .watch else { return .worlds }
        return page
    }
}

/// Names the one window of `alife` (a `Window` scene), outside `AppWindow`, which belongs to the two old modes.
package enum AppShell {
    package static let mainWindowID = "main"
    /// The set-up page in a window of its own (docs/APP.md §3.7), so that a form can be composed beside a Watch.
    package static let setupWindowID = "setup"
    /// The preference that remembers the page between launches (`ShellPage.restored`).
    package static let pageKey = "shell.page"
    /// The minimum size of the window for every page (docs/APP.md M12), so that entering Watch never resizes it.
    package static let minWidth: Double = 1100
    package static let minHeight: Double = 700
}
