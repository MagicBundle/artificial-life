import AlifeCore
import SwiftUI

/// What can be done with one world right now, and why not when it cannot: the buttons of its card on the Worlds page, of its Details page and
/// of the Run panel, from one place, so the three never disagree (docs/APP.md §5.2). A `reason` is said beside every grey button.
struct WorldActionSet {
    var play: CardAction
    var watch: CardAction
    var makePlanet: CardAction
    var keepGrowing: CardAction
    var branch: CardAction
    var check: CardAction
    /// The flat world's own viewer: shown for a flat world only.
    var classic: CardAction

    /// The facts the Worlds page's reasons are made from.
    @MainActor
    static func worldsFacts(controller: RunController, play: PlayController, library: WorldLibrary) -> WorldsGuide.Facts {
        WorldsGuide.Facts(
            folderExists: controller.repoFolderExists, folderProblem: WorldsPage.folderSentence(controller), engineReady: controller.repoReady,
            catalog: library.state.catalog, openWorld: play.isActive ? play.world : nil, openWorldIsStarting: play.phase == .starting,
            saving: Set(play.retiredWorlds), growing: controller.runningRunName)
    }

    @MainActor
    static func make(row: LibraryRow, controller: RunController, play: PlayController, library: WorldLibrary, setup: SetupModel) -> WorldActionSet {
        let facts = worldsFacts(controller: controller, play: play, library: library)
        let playable = library.state.catalog.worlds.first { $0.name == row.name }
        var playReason: String?
        if row.kind == .played { playReason = playable.flatMap { WorldsGuide.reason(forPlaying: $0, in: facts) } ?? (playable == nil ? "Not playable yet" : nil) }
        let ruleFacts = setup.ruleFacts
        var checkReason: String?
        if !controller.repoReady {
            checkReason = "Set up the engine first."
        } else if row.state != .saved {
            checkReason = row.state == .growing ? "Wait until it stops growing." : "Only a world that is saved and not in use can be checked."
        } else if controller.job != nil {
            checkReason = "Another job is running. Wait for it to finish."
        }
        return WorldActionSet(
            play: CardAction(label: "Play", reason: playReason),
            watch: CardAction(label: "Watch", reason: WatchGuide.reasonToWatch(row, openWorld: facts.openWorld, runGrowing: controller.runningRunName)),
            makePlanet: CardAction(label: "Make planet…", reason: WorldsGuide.reasonToMakePlanet(from: row, in: facts) ?? SetupRules.reason(.makePlanet, on: row, in: ruleFacts)),
            keepGrowing: CardAction(label: "Keep growing…", reason: SetupRules.reason(.keepGrowing, on: row, in: ruleFacts)),
            branch: CardAction(label: "Branch…", reason: SetupRules.reason(.branch, on: row, in: ruleFacts)),
            check: CardAction(label: "Check world", reason: checkReason),
            classic: CardAction(label: "Open Classic 2D Viewer", reason: controller.viewerRunning ? "The classic viewer is open already." : (controller.repoReady ? nil : "Set up the engine first.")))
    }
}
