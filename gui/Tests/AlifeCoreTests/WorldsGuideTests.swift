import Foundation
import Testing

@testable import AlifeCore

/// What the Worlds page says is the next step, and why a button is grey (docs/APP.md §2.4: every disabled control has its reason).
@Suite struct WorldsGuideTests {
    private func world(_ name: String, step: Int? = 100, locked: Bool = false, geometry: String? = "sphere") -> PlayWorld {
        let cadence = PlayWorldFile.Cadence(metricsEvery: 1, heavyMetricsEvery: 1, checkpointEvery: 1, keepCheckpoints: 1)
        let file = PlayWorldFile(playVersion: 1, baseRate: 14.4, cadence: cadence, tick: 1, weather: nil, planet: nil)
        return PlayWorld(name: name, file: file, newestStep: step, status: nil, modified: Date(timeIntervalSince1970: 1_000_000),
                         locked: locked, geometry: geometry)
    }

    private func source(_ name: String) -> PlaySource {
        PlaySource(name: name, checkpoints: [100], dims: 3, weatherOn: false, hasEvents: true, modified: Date())
    }

    private func catalog(worlds: [PlayWorld] = [], sources: [PlaySource] = []) -> PlayCatalog {
        var c = PlayCatalog()
        c.worlds = worlds
        c.sources = sources
        return c
    }

    private func facts(folder: Bool = true, engine: Bool = true, _ catalog: PlayCatalog = PlayCatalog(), open: String? = nil,
                       starting: Bool = false, saving: Set<String> = []) -> WorldsGuide.Facts {
        WorldsGuide.Facts(folderExists: folder, folderProblem: folder ? nil : "Folder not found: /x", engineReady: engine, catalog: catalog,
                          openWorld: open, openWorldIsStarting: starting, saving: saving)
    }

    @Test func everyStateHasOneNextStep() {
        #expect(WorldsGuide.nextStep(facts(folder: false)) == .chooseFolder(problem: "Folder not found: /x"))
        #expect(WorldsGuide.nextStep(facts(engine: false)) == .setUpEngine)
        #expect(WorldsGuide.nextStep(facts()) == .makeFirstRun)
        #expect(WorldsGuide.nextStep(facts(catalog(sources: [source("r1")]))) == .makePlanet(from: "r1"))
        #expect(WorldsGuide.nextStep(facts(catalog(worlds: [world("w1")], sources: [source("w1")]))) == .play(world: "w1"))
    }

    @Test func theMostRecentPlayableWorldIsOffered() {
        let c = catalog(worlds: [world("newest", locked: true), world("older"), world("oldest")])
        #expect(WorldsGuide.nextStep(facts(c)) == .play(world: "older"))
        let allBusy = catalog(worlds: [world("a", locked: true), world("b", step: nil)])
        #expect(WorldsGuide.nextStep(facts(allBusy)) == .waitForSave)
    }

    @Test func anOpenWorldWinsOverEverythingElse() {
        #expect(WorldsGuide.nextStep(facts(folder: false, open: "w")) == .watch(world: "w"))
        #expect(WorldsGuide.nextStep(facts(open: "w", starting: true)) == .starting(world: "w"))
    }

    @Test func aGreyPlayButtonAlwaysHasAReason() {
        let f = facts(catalog(worlds: [world("a"), world("b", locked: true), world("c", step: nil), world("d")]), saving: ["d"])
        #expect(WorldsGuide.reason(forPlaying: world("a"), in: f) == nil)
        #expect(WorldsGuide.reason(forPlaying: world("b", locked: true), in: f) == "Being played in another window")
        #expect(WorldsGuide.reason(forPlaying: world("c", step: nil), in: f) == "No save point")
        #expect(WorldsGuide.reason(forPlaying: world("d"), in: f) == "Saving...")
        let open = facts(open: "a")
        #expect(WorldsGuide.reason(forPlaying: world("b"), in: open) == "Close a first")
        #expect(WorldsGuide.reason(forPlaying: world("a"), in: open) == "Open now")
    }

    @Test func thePlayButtonsUseTheSpecsWordsNotTheOldOnes() {
        let all = [world("a", locked: true), world("a", step: nil), world("a")]
        let f = facts(catalog(worlds: all), saving: ["a"])
        for w in all {
            let text = WorldsGuide.reason(forPlaying: w, in: f) ?? ""
            #expect(!text.localizedCaseInsensitiveContains("checkpoint"))
            #expect(!text.localizedCaseInsensitiveContains("resume"))
        }
    }

    @Test func makingAPlanetSaysWhyItCannotBeDoneYet() {
        #expect(WorldsGuide.reasonToMakePlanet(facts(folder: false)) == "Folder not found: /x")
        #expect(WorldsGuide.reasonToMakePlanet(facts(engine: false)) == "Set up the engine first.")
        #expect(WorldsGuide.reasonToMakePlanet(facts())?.contains("research run") == true)
        #expect(WorldsGuide.reasonToMakePlanet(facts(open: "w"))?.hasPrefix("Close w first") == true)
        #expect(WorldsGuide.reasonToMakePlanet(facts(catalog(sources: [source("r")]))) == nil)
    }

    @Test func theRowSaysWhatTheWorldIs() {
        let now = Date(timeIntervalSince1970: 1_000_000 + 7200)
        #expect(WorldsGuide.summary(world("w", step: 450_000), now: now) == "Planet · step 450,000 · saved 2 hours ago")
        #expect(WorldsGuide.summary(world("w", step: nil, geometry: "box"), now: now) == "World · saved 2 hours ago")
        #expect(WorldsGuide.grouped(30000) == "30,000")
    }
}

@Suite struct WorldsGuideTimeTests {
    @Test func aWorldSavedAMomentAgoSaysSoInsteadOfInZeroSeconds() {
        let now = Date(timeIntervalSince1970: 5_000_000)
        #expect(WorldsGuide.relative(now.addingTimeInterval(-20), now: now) == "just now")
        #expect(WorldsGuide.relative(now.addingTimeInterval(-3600), now: now) == "1 hour ago")
    }
}
