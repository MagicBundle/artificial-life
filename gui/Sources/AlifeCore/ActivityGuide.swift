import Foundation

/// What the Activity page lists and what the sidebar says about it (docs/APP.md §2.4): the jobs the app has going, each with the one
/// control that ends it. Pure; the page draws it.
package enum ActivityGuide {
    package struct Facts: Equatable, Sendable {
        /// The research run in progress (its directory name, or "a run" when unnamed), nil when none.
        package var runName: String?
        package var runStopping = false
        package var setupRunning = false
        package var checkRunning = false
        /// The world that is open in Watch, and whether it is still starting.
        package var worldName: String?
        package var worldStarting = false
        /// Worlds that were asked to stop and are still saving.
        package var saving: [String] = []
        /// The run the 3D viewer's stream shows, nil when no stream runs.
        package var streamRun: String?
        package var classicViewerOpen = false

        package init(runName: String? = nil, runStopping: Bool = false, setupRunning: Bool = false, checkRunning: Bool = false,
                     worldName: String? = nil, worldStarting: Bool = false, saving: [String] = [], streamRun: String? = nil,
                     classicViewerOpen: Bool = false) {
            self.runName = runName
            self.runStopping = runStopping
            self.setupRunning = setupRunning
            self.checkRunning = checkRunning
            self.worldName = worldName
            self.worldStarting = worldStarting
            self.saving = saving
            self.streamRun = streamRun
            self.classicViewerOpen = classicViewerOpen
        }
    }

    /// The control on a row.
    package enum Control: Equatable, Sendable {
        case stop, forceStop, cancel, closeWorld, closeViewer

        /// The words on the button (§2.5: Stop / Force stop end a background job at a save point, Close World ends a live one).
        package var label: String {
            switch self {
            case .stop: return "Stop"
            case .forceStop: return "Force stop"
            case .cancel: return "Cancel"
            case .closeWorld: return "Close World"
            case .closeViewer: return "Close viewer"
            }
        }
    }

    package struct Item: Equatable, Identifiable, Sendable {
        package enum Kind: String, Sendable { case run, setup, check, world, saving, stream, viewer }
        package var kind: Kind
        package var title: String
        package var detail: String
        package var controls: [Control]
        package var id: String { kind.rawValue + ":" + title }
    }

    /// "Research run planet-2"; the run that grows a planet's world is "Growing the world for planet-1" (its directory name is not for the page).
    package static func runTitle(_ name: String) -> String {
        if name.hasSuffix(SetupPlan.growSuffix), name.count > SetupPlan.growSuffix.count {
            return "Growing the world for " + String(name.dropLast(SetupPlan.growSuffix.count))
        }
        return "Research run \(name)"
    }

    package static func items(_ facts: Facts) -> [Item] {
        var out: [Item] = []
        if facts.runName != nil || facts.runStopping {
            let name = facts.runName ?? "a run"
            out.append(Item(
                kind: .run, title: Self.runTitle(name),
                detail: facts.runStopping ? "Stopping: it writes a save point first." : "Growing. Stop ends it at a save point.",
                controls: facts.runStopping ? [.forceStop] : [.stop]))
        }
        if facts.setupRunning {
            out.append(Item(kind: .setup, title: "Setting up the engine", detail: "Installing packages with uv. Progress is in the log below.",
                            controls: [.cancel]))
        }
        if facts.checkRunning {
            out.append(Item(kind: .check, title: "Checking a world", detail: "The result will be in the log below.", controls: [.cancel]))
        }
        if let world = facts.worldName {
            out.append(Item(
                kind: .world, title: "World \(world)",
                detail: facts.worldStarting ? "Starting: the fork and the weather warm-up take a while." : "Open in Watch. Closing it saves first.",
                controls: [.closeWorld]))
        }
        for name in facts.saving {
            out.append(Item(kind: .saving, title: "Saving \(name)", detail: "Writing its save point. This ends by itself.", controls: []))
        }
        if let run = facts.streamRun {
            out.append(Item(kind: .stream, title: "3D viewer on \(run)", detail: "Close the 3D Viewer window to end it.", controls: []))
        }
        if facts.classicViewerOpen {
            out.append(Item(kind: .viewer, title: "Classic 2D viewer", detail: "A separate window; it shares the GPU with a run.", controls: [.closeViewer]))
        }
        return out
    }

    /// The second line of the sidebar's Activity row.
    package static func sidebarDetail(_ facts: Facts) -> String {
        let items = items(facts).filter { $0.kind != .saving }
        switch items.count {
        case 0: return facts.saving.isEmpty ? "Nothing running" : "Saving a world"
        case 1: return items[0].title
        default: return "\(items.count) running"
        }
    }

    /// What the page says when there is nothing to list: where to go next.
    package static let emptyAdvice = "Nothing is running. Play a world from Worlds, or make a new one."
}
