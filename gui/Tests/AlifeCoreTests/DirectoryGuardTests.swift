import Foundation
import Testing

@testable import AlifeCore

/// R6 and the state x action matrix of docs/APP.md §5.2.
@Suite struct DirectoryGuardTests {
    // MARK: One writer per directory

    @Test func aSecondWriterIsRefusedAndTheFirstIsKept() {
        var g = DirectoryGuard()
        let first = g.claim("planet-1-grow", as: .growing)
        let second = g.claim("planet-1-grow", as: .makingPlanet)
        #expect(first)
        #expect(!second, "a second writer of one directory")
        #expect(g.writer(of: "planet-1-grow")?.writer == .growing)
        g.release("planet-1-grow")
        let third = g.claim("planet-1-grow", as: .makingPlanet)
        #expect(third)
    }

    @Test func namesThatDifferOnlyInCapitalsAreOneDirectory() {
        var g = DirectoryGuard()
        let first = g.claim("Planet-1", as: .playing)
        let second = g.claim("planet-1", as: .growing)
        #expect(first)
        #expect(!second, "a case-insensitive volume has one folder for both")
        #expect(g.writeRefusal("PLANET-1") == "Planet-1 is open. Close it first.")
        g.release("pLaNeT-1")
        #expect(g.writer(of: "planet-1") == nil)
    }

    @Test func everyWriterHasASentenceThatSaysWhatToDo() {
        for (writer, expected) in [
            (DirectoryGuard.Writer.playing, "w is open. Close it first."),
            (.growing, "w is growing. Wait for it to finish, or stop it."),
            (.makingPlanet, "w is being made."),
            (.saving, "w is still saving."),
        ] {
            let g = DirectoryGuard(writers: ["w": writer])
            #expect(g.writeRefusal("w") == expected)
            #expect(g.writeRefusal("other") == nil)
        }
    }

    @Test func aDirectoryAnotherProcessHoldsIsRefusedNotJustWarnedAbout() {
        let g = DirectoryGuard(writers: [:], heldElsewhere: ["held"])
        #expect(g.writeRefusal("held") == "held is in use by another app or process.")
        #expect(g.writeRefusal("HELD") != nil)
        #expect(g.writeRefusal("quiet") == nil)
        #expect(g.readWarning("held")?.contains("seems to be in use by another process") == true && g.readWarning("quiet") == nil)
    }

    @Test func recordingReplacesWhatWasThere() {
        var g = DirectoryGuard()
        g.record("w", as: .playing)
        g.record("W", as: .saving)
        #expect(g.writers == ["W": .saving])
    }

    // MARK: The matrix

    private func row(_ state: WorldState, kind: WorldKind = .research, step: Int? = 30_000, name: String = "w") -> LibraryRow {
        LibraryRow(name: name, kind: kind, shape: .planet, state: state, note: nil, step: step, lastSaved: Date(timeIntervalSince1970: 1_000))
    }

    private func facts(runJob: Bool = false, open: String? = nil, starting: Bool = false, guardState: DirectoryGuard = DirectoryGuard(),
                       engine: Bool = true, folder: Bool = true) -> SetupRules.Facts {
        SetupRules.Facts(runJob: runJob, openWorld: open, openWorldStarting: starting, guardState: guardState, folderExists: folder,
                         engineReady: engine)
    }

    @Test func keepGrowingFollowsTheMatrix() {
        let f = facts()
        #expect(SetupRules.reason(.keepGrowing, on: row(.saved), in: f) == nil)
        #expect(SetupRules.reason(.keepGrowing, on: row(.saved), in: facts(runJob: true)) == "A run is going. Wait for it to finish, or stop it.")
        #expect(SetupRules.reason(.keepGrowing, on: row(.growing), in: f) == "Already growing.")
        #expect(SetupRules.reason(.keepGrowing, on: row(.inUseElsewhere), in: f) == "In use by another app or process.")
        #expect(SetupRules.reason(.keepGrowing, on: row(.saving), in: f) == "Saving…")
        #expect(SetupRules.reason(.keepGrowing, on: row(.saved, kind: .played), in: f) == "A played world is continued by playing it.")
        #expect(SetupRules.reason(.keepGrowing, on: row(.saved, step: nil), in: f) == "No save point to continue from.")
        // a world this app writes in any way is not continued a second time
        for writer in [DirectoryGuard.Writer.playing, .growing, .makingPlanet, .saving] {
            let g = DirectoryGuard(writers: ["w": writer])
            #expect(SetupRules.reason(.keepGrowing, on: row(.saved), in: facts(guardState: g)) != nil, "\(writer)")
        }
    }

    @Test func branchFollowsTheMatrix() {
        let f = facts()
        #expect(SetupRules.reason(.branch, on: row(.saved), in: f) == nil)
        #expect(SetupRules.reason(.branch, on: row(.live, kind: .played), in: f) == "Close it first: it is being written.")
        #expect(SetupRules.reason(.branch, on: row(.paused, kind: .played), in: f) == "Close it first: it is being written.")
        #expect(SetupRules.reason(.branch, on: row(.starting, kind: .played), in: f) == "Close it first: it is being written.")
        #expect(SetupRules.reason(.branch, on: row(.growing), in: f) == nil, "from its newest save point, with a note")
        #expect(SetupRules.reason(.branch, on: row(.saving), in: f) == "Saving…")
        #expect(SetupRules.reason(.branch, on: row(.inUseElsewhere), in: f) == nil, "a warning, not a refusal")
        #expect(SetupRules.reason(.branch, on: row(.saved), in: facts(runJob: true)) == "A run is going. Wait for it to finish, or stop it.")
        #expect(SetupRules.reason(.branch, on: row(.saved, step: nil), in: f) == "No save point to start from yet.")
    }

    @Test func makePlanetFollowsTheMatrix() {
        let f = facts()
        #expect(SetupRules.reason(.makePlanet, on: row(.saved), in: f) == nil)
        // R5: another world that is live is stopped by the sheet, so it is not a reason; one that is still starting has to be waited for
        #expect(SetupRules.reason(.makePlanet, on: row(.saved), in: facts(open: "other")) == nil)
        #expect(SetupRules.reason(.makePlanet, on: row(.saved), in: facts(open: "other", starting: true)) == "other is still starting")
        #expect(SetupRules.reason(.makePlanet, on: row(.live, kind: .played), in: facts(open: "w")) == "Close it first")
        #expect(SetupRules.reason(.makePlanet, on: row(.growing), in: f) == nil, "newest save point, with a note")
        #expect(SetupRules.reason(.makePlanet, on: row(.saving), in: f) == "Saving…")
        #expect(SetupRules.reason(.makePlanet, on: row(.inUseElsewhere), in: f) == nil)
        #expect(SetupRules.reason(.makePlanet, on: row(.saved, step: nil), in: f) == "No save point to start from yet.")
        // a run going does not stop a planet being made: it is a different process and the source is read, not written
        #expect(SetupRules.reason(.makePlanet, on: row(.saved), in: facts(runJob: true)) == nil)
    }

    @Test func aFlatWorldCannotBecomeAPlanetAndSaysWhy() {
        var flat = row(.saved)
        flat.shape = .flat
        #expect(SetupRules.reason(.makePlanet, on: flat, in: facts()) == "A flat world has no weather, so it cannot become a planet.")
        #expect(SetupRules.reason(.branch, on: flat, in: facts()) == nil && SetupRules.reason(.keepGrowing, on: flat, in: facts()) == nil)
        let f = WorldsGuide.Facts(folderExists: true, engineReady: true)
        #expect(WorldsGuide.reasonToMakePlanet(from: flat, in: f) == WorldsGuide.flatHasNoPlanet)
        flat.shape = .planet
        #expect(WorldsGuide.reasonToMakePlanet(from: flat, in: f) == nil)
    }

    @Test func theFolderAndTheEngineComeFirst() {
        for action in [SetupAction.keepGrowing, .branch, .makePlanet] {
            #expect(SetupRules.reason(action, on: row(.saved), in: facts(folder: false)) == "The alife folder was not found.")
            #expect(SetupRules.reason(action, on: row(.saved), in: facts(engine: false)) == "Set up the engine first.")
        }
    }

    @Test func aNoteSaysWhenTheSourceIsStillGrowingOrSeemsInUse() {
        #expect(SetupRules.note(.branch, on: row(.growing), in: facts()) == "w is still growing. This starts from its newest save point.")
        #expect(SetupRules.note(.makePlanet, on: row(.growing), in: facts()) != nil)
        #expect(SetupRules.note(.keepGrowing, on: row(.growing), in: facts()) == nil)
        let g = DirectoryGuard(writers: [:], heldElsewhere: ["w"])
        #expect(SetupRules.note(.branch, on: row(.saved), in: facts(guardState: g))?.contains("another process") == true)
        #expect(SetupRules.note(.branch, on: row(.saved), in: facts()) == nil)
    }

    @Test func noReasonUsesARetiredWord() {
        let banned = ["Resume", "Checkpoint", "Fork", "Home", "spin-up", "Source", "the Lab"]
        let states = WorldState.allCases
        for action in [SetupAction.keepGrowing, .branch, .makePlanet] {
            for state in states {
                for kind in WorldKind.allCases {
                    for step in [Optional(30_000), nil] {
                        let reason = SetupRules.reason(action, on: row(state, kind: kind, step: step), in: facts(runJob: true, open: "x"))
                        for word in banned { #expect(!(reason ?? "").contains(word), "\(action) \(state) \(kind): \(reason ?? "")") }
                    }
                }
            }
        }
    }
}
