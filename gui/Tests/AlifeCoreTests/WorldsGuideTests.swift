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
        #expect(WorldsGuide.reason(forPlaying: world("b", locked: true), in: f) == "In use elsewhere")
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
        #expect(WorldsGuide.reasonToMakePlanet(facts())?.contains("world with a save point") == true)
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

@Suite struct WorldsGuideFailureTests {
    @Test func aFailedStartIsExplainedInPlainWordsWithAWayOn() {
        let cases: [(String, String)] = [
            ("'w' is still saving: wait until it has closed.", "still being saved"),
            ("Failed to launch /x/alife: no such file", "could not be started"),
            ("alife play found no usable checkpoint (exit code 4)", "no usable save point"),
            ("alife play refused the request (exit code 2)", "refused the request"),
            ("alife play was interrupted (exit code 130)", "stopped before it was ready"),
            ("alife play exited with code 9", "the engine stopped"),
        ]
        for (message, expected) in cases {
            let text = WorldsGuide.explainFailure(world: "w", message: message)
            #expect(text.hasPrefix("w ") && text.contains(expected), "\(message) -> \(text)")
            #expect(!text.localizedCaseInsensitiveContains("checkpoint"), "\(text)")
        }
        #expect(WorldsGuide.explainFailure(world: "", message: "x").hasPrefix("The world "))
    }

    @Test func aRunThatIsStillGrowingSaysThatAPlanetStartsFromItsLatestSavePoint() {
        #expect(WorldsGuide.growingNote(source: "r1", growing: "r1")?.contains("latest save point") == true)
        #expect(WorldsGuide.growingNote(source: "r1", growing: "r2") == nil)
        #expect(WorldsGuide.growingNote(source: "r1", growing: nil) == nil)
    }
}

@Suite struct WatchCueTests {
    @Test func whileLookingTheRowSaysOnlyLive() {
        let cue = WatchCue.cue(phase: .live, world: "w", stop: .chapter, chapter: 2, unread: 3, looking: true)
        #expect(cue == WatchCue.Cue(text: "w · live", needsPlayer: false))
    }

    @Test func offWatchAStopAndTheUnreadAlertsAreInTheRow() {
        let chapter = WatchCue.cue(phase: .live, world: "w", stop: .chapter, chapter: 2, unread: 0, looking: false)
        #expect(chapter == WatchCue.Cue(text: "w · chapter 2 is waiting", needsPlayer: true))
        let alert = WatchCue.cue(phase: .live, world: "w", stop: .alert, chapter: 2, unread: 2, looking: false)
        #expect(alert == WatchCue.Cue(text: "w · stopped for an alert · 2 unread alerts", needsPlayer: true))
        let quiet = WatchCue.cue(phase: .live, world: "w", stop: nil, chapter: 2, unread: 0, looking: false)
        #expect(quiet == WatchCue.Cue(text: "w · live", needsPlayer: false))
        let one = WatchCue.cue(phase: .live, world: "w", stop: nil, chapter: 0, unread: 1, looking: false)
        #expect(one.text == "w · live · 1 unread alert" && one.needsPlayer)
    }

    @Test func theOtherStatesNeverAskForThePlayer() {
        #expect(WatchCue.cue(phase: .none, world: nil, stop: nil, chapter: 0, unread: 0, looking: false).text == "No world open")
        #expect(WatchCue.cue(phase: .starting, world: "w", stop: nil, chapter: 0, unread: 0, looking: false).text == "w · starting")
        #expect(WatchCue.cue(phase: .failedToStart, world: "w", stop: nil, chapter: 0, unread: 0, looking: false).text == "w · did not start")
        #expect(WatchCue.cue(phase: .failed, world: "w", stop: nil, chapter: 0, unread: 0, looking: false).text == "w · stopped unexpectedly")
        for phase in [WatchCue.Phase.none, .starting, .ended, .failed, .failedToStart] {
            #expect(!WatchCue.cue(phase: phase, world: "w", stop: .alert, chapter: 1, unread: 4, looking: false).needsPlayer)
        }
    }
}
