import AlifeCore
import Foundation
import Testing

@testable import AlifeControl

/// What can be done with a world and why not (the card's buttons, Details' and the Run panel's): from one place.
@MainActor
@Suite("The actions of a world", .serialized)
struct WorldActionSetTests {
    @MainActor struct Rig {
        let scratch: Scratch
        let controller: RunController
        let play = PlayController()
        let library: WorldLibrary
        let setup: SetupModel
        var runs: URL { scratch.url.appendingPathComponent("folder/runs") }

        init(engine: Bool = true) throws {
            scratch = try Scratch()
            let root = scratch.url.appendingPathComponent("folder")
            try FileManager.default.createDirectory(at: root.appendingPathComponent("runs"), withIntermediateDirectories: true)
            if engine {
                try FileManager.default.createDirectory(at: root.appendingPathComponent(".venv/bin"), withIntermediateDirectories: true)
                let alife = try scratch.script("alife", "exit 0")
                try FileManager.default.moveItem(at: alife, to: root.appendingPathComponent(".venv/bin/alife"))
            }
            try LibraryTestFolders.research(root.appendingPathComponent("runs"), "r")
            try LibraryTestFolders.research(root.appendingPathComponent("runs"), "flat", sphere: false, dims: 2)
            try LibraryTestFolders.research(root.appendingPathComponent("runs"), "dead", status: "extinct")
            try LibraryTestFolders.played(root.appendingPathComponent("runs"), "p", source: "runs/r:30000")
            controller = RunController(storedPath: root.path, savePath: { _ in }, candidates: [], isApp: false)
            library = WorldLibrary(runsDirectory: { root.appendingPathComponent("runs") }, observesApp: false, lockHeld: { _ in false })
            let defaults = UserDefaults(suiteName: "alife-actions-\(UUID().uuidString)")!
            setup = SetupModel.forApp(controller: controller, play: play, library: library, defaults: defaults)
        }

        func ready() async -> Bool {
            library.refresh()
            return await eventually { library.hasScanned && library.state.rows.count == 4 }
        }

        func actions(_ name: String) -> WorldActionSet {
            WorldActionSet.make(row: library.state.row(named: name)!, controller: controller, play: play, library: library, setup: setup)
        }
    }

    @Test func aSavedResearchWorldCanBeWatchedMadeIntoAPlanetContinuedBranchedAndChecked() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        #expect(await rig.ready())
        let a = rig.actions("r")
        #expect(a.watch.reason == nil && a.makePlanet.reason == nil && a.keepGrowing.reason == nil && a.branch.reason == nil && a.check.reason == nil)
        #expect(a.play.reason == nil, "a research world has no Play button: no reason is given for it")
        #expect([a.watch.label, a.makePlanet.label, a.keepGrowing.label, a.branch.label, a.check.label, a.classic.label]
                == ["Watch", "Make planet…", "Keep growing…", "Branch…", "Check world", "Open Classic 2D Viewer"])
    }

    @Test func aFlatWorldAndADiedOutWorldSayWhyTheyCannotBecomeAPlanet() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        #expect(await rig.ready())
        #expect(rig.actions("flat").makePlanet.reason == WorldsGuide.flatHasNoPlanet)
        #expect(rig.actions("dead").makePlanet.reason == WorldsGuide.diedOutHasNoPlanet)
        #expect(rig.actions("dead").keepGrowing.reason?.contains("died out") == true)
        #expect(rig.actions("flat").watch.reason == nil, "a flat world can still be watched")
    }

    @Test func aPlayedWorldIsPlayedNotWatchedUntilItIsOpen() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        #expect(await rig.ready())
        let a = rig.actions("p")
        #expect(a.play.reason == nil)
        #expect(a.watch.reason?.contains("Press Play") == true)
        #expect(a.keepGrowing.reason == "A played world is continued by playing it.")
    }

    @Test func withoutAnEngineNothingCanBeDoneAndTheReasonSaysWhy() async throws {
        let rig = try Rig(engine: false)
        defer { rig.scratch.cleanup() }
        #expect(await rig.ready())
        let a = rig.actions("r")
        #expect(a.check.reason == "Set up the engine first." && a.classic.reason == "Set up the engine first.")
        #expect(a.keepGrowing.reason != nil && a.branch.reason != nil && a.makePlanet.reason != nil)
        #expect(a.watch.reason == nil, "watching needs a save point, and the engine is checked by the page when it starts")
    }

    @Test func aWorldThatGrowsOrSavesCannotBeCheckedAndSaysWhy() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        #expect(await rig.ready())
        rig.library.update(live: LiveFacts(growing: "r"))
        #expect(rig.actions("r").check.reason == "Wait until it stops growing.")
        rig.library.update(live: LiveFacts(saving: ["r"]))
        #expect(rig.actions("r").check.reason == "Only a world that is saved and not in use can be checked.")
        rig.library.update(live: LiveFacts())
        #expect(rig.actions("r").check.reason == nil)
    }
}
