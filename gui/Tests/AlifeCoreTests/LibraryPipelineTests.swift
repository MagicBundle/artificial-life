import Foundation
import Testing

@testable import AlifeCore

/// What the library says about a pipeline's worlds (docs/APP.md §3.2): one click is one world with the seed nested under it, a seed whose
/// planet was never made says so (after a stop or a relaunch), and a run that stopped early says where.
@Suite struct LibraryPipelineTests {
    private let when = Date(timeIntervalSince1970: 1_000_000)

    private func record(_ name: String, steps: [Int] = [30_000], status: String? = "done", geometry: String? = "sphere", target: Int? = nil,
                        played: Bool = false, unfinished: Bool = false) -> LibraryRecord {
        let cadence = PlayWorldFile.Cadence(metricsEvery: 1, heavyMetricsEvery: 1, checkpointEvery: 1, keepCheckpoints: 1)
        let file = PlayWorldFile(playVersion: 1, baseRate: 14.4, cadence: cadence, tick: 1, weather: nil, planet: nil)
        var r = LibraryRecord(
            name: name, play: played ? file : nil, steps: steps, status: status, parent: nil, dims: 3, geometry: geometry, worldSize: 64,
            planetRadius: 36, planetShape: nil, weatherOn: false, hasEvents: true, directoryModified: when, lastSaved: when, hasContent: true,
            looksUnfinished: unfinished, buildMark: unfinished ? "planet" : nil)
        r.targetSteps = target
        return r
    }

    private func state(_ records: [LibraryRecord], live: LiveFacts = LiveFacts()) -> LibraryState {
        var s = LibrarySnapshot()
        s.records = records
        return LibraryState.resolve(s, live: live)
    }

    // MARK: One click, one world

    @Test func aPlanetBeingGrownIsListedBeforeItExistsWithItsSeedNestedUnderIt() {
        let live = LiveFacts(growing: "planet-1-grow", pending: [PendingPlanet(name: "planet-1", seed: "planet-1-grow", stage: .growing, since: when)])
        let s = state([record("planet-1-grow", steps: [50], status: "running")], live: live)
        let row = s.row(named: "planet-1")
        #expect(row?.isPlaceholder == true && row?.kind == .played && row?.state == .growing && row?.shape == .planet)
        #expect(row?.note == "Growing first, then it becomes a planet.")
        let tree = LibraryState.tree(s.rows)
        #expect(tree.map(\.row.name) == ["planet-1"], "the seed is not a second world in the list")
        #expect(tree.first?.children.map(\.name) == ["planet-1-grow"])
    }

    @Test func whileThePlanetIsMadeItIsStartingAndItsHalfBuiltFolderIsNotAnUnfinishedBuild() {
        let live = LiveFacts(openWorld: "planet-1", openPhase: .starting, pending: [PendingPlanet(name: "planet-1", seed: "planet-1-grow", stage: .making, since: when)])
        let building = record("planet-1", steps: [], status: "running", unfinished: true)
        let s = state([record("planet-1-grow"), building], live: live)
        #expect(s.row(named: "planet-1")?.state == .starting && s.row(named: "planet-1")?.note == "Being made.")
        #expect(s.unfinished.isEmpty, "it is a build in progress, not a stopped one")
        #expect(s.catalog.existingNames.contains("planet-1"), "and its name stays taken")
        #expect(s.rows.filter { $0.name == "planet-1" }.count == 1)
    }

    @Test func theRealWorldReplacesThePlaceholderWithoutAGap() {
        let live = LiveFacts(openWorld: "planet-1", openPhase: .live, pending: [PendingPlanet(name: "Planet-1", seed: "planet-1-grow", stage: .making, since: when)])
        let s = state([record("planet-1-grow"), record("planet-1", steps: [30_000], played: true)], live: live)
        #expect(s.rows.filter { $0.name.lowercased() == "planet-1" }.count == 1)
        #expect(s.row(named: "planet-1")?.isPlaceholder == false && s.row(named: "planet-1")?.state == .live)
    }

    @Test func aSeedStaysNestedUnderTheOpenPlanetBeforeTheScanSeesThePlanet() {
        // the pipeline is done (no pending planet) and the open world's `play.json` is not read yet: the seed is not a top-level row
        for phase in [LiveFacts.OpenPhase.starting, .live, .paused] {
            let live = LiveFacts(openWorld: "planet-1", openPhase: phase)
            let s = state([record("planet-1-grow"), record("planet-1", steps: [], status: "running", unfinished: true)], live: live)
            let tree = LibraryState.tree(s.rows)
            #expect(tree.map(\.row.name) == ["planet-1"] && tree.first?.children.map(\.name) == ["planet-1-grow"], "\(phase)")
            #expect(s.unfinished.isEmpty)
        }
        // once the planet is a world, the real row nests as before; and a world that is not open gets no placeholder
        let live = LiveFacts(openWorld: "planet-1", openPhase: .live)
        let real = state([record("planet-1-grow"), record("planet-1", played: true)], live: live)
        #expect(real.rows.filter { $0.name == "planet-1" }.count == 1 && real.row(named: "planet-1")?.isPlaceholder == false)
        #expect(state([record("planet-1-grow")]).rows.map(\.name) == ["planet-1-grow"])
    }

    @Test func theRunThisAppGrowsIsNeverAnUnfinishedBuildWhateverItLooksLikeWhileItIsBeingWritten() {
        let live = LiveFacts(growing: "g")
        let s = state([record("g", steps: [], status: "running", unfinished: true)], live: live)
        #expect(s.row(named: "g")?.state == .growing && s.row(named: "g")?.kind == .research && s.unfinished.isEmpty)
    }

    @Test func aPlaceholderDoesNotChangeBetweenTwoLooks() {
        let live = LiveFacts(pending: [PendingPlanet(name: "p", seed: "p-grow", stage: .growing, since: when)])
        #expect(state([record("p-grow")], live: live) == state([record("p-grow")], live: live), "the date is the pipeline's, so a look is not a change")
    }

    @Test func aPlaceholderIsNotACatalogWorldOrASource() {
        let live = LiveFacts(pending: [PendingPlanet(name: "p", seed: "p-grow", stage: .growing, since: when)])
        let s = state([record("p-grow")], live: live)
        #expect(!s.catalog.worlds.contains { $0.name == "p" } && !s.catalog.sources.contains { $0.name == "p" })
        #expect(s.counts.played == 1, "it counts as one played world")
    }

    // MARK: A seed whose planet was never made

    @Test func aSeedWithoutItsPlanetSaysWhichPlanetWasMeantAndOffersTheNextAction() {
        let s = state([record("planet-1-grow"), record("other"), record("flat-grow", geometry: "box")])
        #expect(s.row(named: "planet-1-grow")?.unmadePlanet == "planet-1")
        #expect(s.row(named: "other")?.unmadePlanet == nil)
        #expect(s.row(named: "flat-grow")?.unmadePlanet == nil, "only a planet can be made")
    }

    @Test func aSeedWhosePlanetExistsInAnyFormIsNotMarked() {
        for taken in [record("planet-1", played: true), record("PLANET-1", played: true), record("planet-1", steps: [], unfinished: true)] {
            let s = state([record("planet-1-grow"), taken])
            #expect(s.row(named: "planet-1-grow")?.unmadePlanet == nil, "\(taken.name)")
        }
        var withLink = LibrarySnapshot()
        withLink.records = [record("planet-1-grow")]
        withLink.problems = ["planet-1": .driveNotConnected(volume: "Archive")]
        #expect(LibraryState.resolve(withLink).row(named: "planet-1-grow")?.unmadePlanet == nil, "a link to a drive that is away has the name")
        #expect(state([record("-grow")]).row(named: "-grow")?.unmadePlanet == nil)
        #expect(state([record("planet-1-grow", steps: [])]).row(named: "planet-1-grow") == nil)
    }

    @Test func aPromisedPlanetIsNotMarkedUnmade() {
        let live = LiveFacts(pending: [PendingPlanet(name: "planet-1", seed: "planet-1-grow", stage: .growing, since: when)])
        #expect(state([record("planet-1-grow")], live: live).row(named: "planet-1-grow")?.unmadePlanet == nil)
    }

    // MARK: Where a run stopped

    @Test func aRunStoppedBeforeItsTargetSaysWhere() {
        let s = state([
            record("a", steps: [12_000], status: "interrupted", target: 30_000), record("b", steps: [30_000], status: "interrupted", target: 30_000),
            record("c", steps: [12_000], status: "done", target: 30_000), record("d", steps: [12_000], status: "interrupted", target: nil),
        ])
        #expect(s.row(named: "a")?.note == "Stopped at step 12,000 of 30,000.")
        #expect(s.row(named: "b")?.note == nil && s.row(named: "c")?.note == nil && s.row(named: "d")?.note == nil)
        #expect(s.row(named: "a")?.targetSteps == 30_000)
    }

    @Test func theScannerReadsTheStepsARunWasAskedFor() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let dir = scratch.url.appendingPathComponent("run-x")
        try LibraryFixture.write(#"{"status": "interrupted"}"#, "meta.json", in: dir)
        try LibraryFixture.write(#"{"world": {"dims": 3, "geometry": "sphere", "radius": 36}, "run": {"steps": 30000}}"#, "config.json", in: dir)
        try LibraryFixture.write("x", "checkpoints/step_0000012000.safetensors", in: dir)
        let snapshot = LibraryScanner().scan(runsDirectory: scratch.url)
        #expect(snapshot.records.first?.targetSteps == 30_000)
        #expect(LibraryState.resolve(snapshot).row(named: "run-x")?.note == "Stopped at step 12,000 of 30,000.")
    }

    @Test func theNewWordsAreNotTheRetiredOnes() {
        let rows = state([record("a-grow", steps: [5], status: "interrupted", target: 30_000)], live: LiveFacts(
            pending: [PendingPlanet(name: "p", seed: "p-grow", stage: .making, since: when)])).rows
        for row in rows {
            for text in [row.note, row.facts, row.state.meaning] {
                for word in ["Resume", "Checkpoint", "Fork", "the Lab"] { #expect(!(text ?? "").contains(word), "\(text ?? "")") }
            }
        }
    }
}
