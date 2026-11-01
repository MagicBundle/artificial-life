import Foundation
import Testing

@testable import AlifeCore

/// What the library says about each world: its kind, shape and state in plain words, its place in the list and in the sidebar's nesting
/// (docs/APP.md §2.2, §2.4, §2.5, §5.2).
@Suite struct LibraryStateTests {
    private let now = Date(timeIntervalSince1970: 1_000_000)

    private func record(
        _ name: String, played: Bool = false, source: String? = nil, steps: [Int] = [100], status: String? = "done", geometry: String? = "sphere",
        dims: Int? = 3, saved: Double = 0, unfinished: Bool = false, mark: String? = nil, content: Bool = true
    ) -> LibraryRecord {
        let cadence = PlayWorldFile.Cadence(metricsEvery: 1, heavyMetricsEvery: 1, checkpointEvery: 1, keepCheckpoints: 1)
        let planet = source.map { PlayWorldFile.Planet(source: $0, step: 0, weatherSpinup: nil, houses: "auto", madeAtStep: 0) }
        let file = PlayWorldFile(playVersion: 1, baseRate: 14.4, cadence: cadence, tick: 1, weather: nil, planet: planet)
        let date = Date(timeIntervalSince1970: 1_000_000 + saved)
        return LibraryRecord(
            name: name, play: played ? file : nil, steps: steps, status: status, parent: nil, dims: dims, geometry: geometry,
            worldSize: 64, planetRadius: geometry == "sphere" ? 36 : nil, planetShape: nil, weatherOn: false, hasEvents: true,
            directoryModified: date, lastSaved: date, hasContent: content, looksUnfinished: unfinished, buildMark: mark)
    }

    private func snapshot(_ records: [LibraryRecord], held: Set<String> = [], written: Set<String> = []) -> LibrarySnapshot {
        var s = LibrarySnapshot()
        s.records = records
        s.held = held
        s.writtenElsewhere = written
        return s
    }

    private func state(_ records: [LibraryRecord], live: LiveFacts = LiveFacts(), held: Set<String> = [], written: Set<String> = [],
                       notes: [String: String] = [:]) -> LibraryState {
        LibraryState.resolve(snapshot(records, held: held, written: written), live: live, notes: notes)
    }

    // MARK: Kind and shape

    @Test func aWorldIsPlayedWithAPlayFileAndResearchWithASavePoint() {
        let s = state([record("p", played: true), record("r"), record("empty", steps: [], content: true)])
        #expect(s.rows.map(\.name).sorted() == ["p", "r"])
        #expect(s.row(named: "p")?.kind == .played && s.row(named: "r")?.kind == .research)
        #expect(s.others == ["empty"], "a folder without a save point is counted, not listed")
    }

    @Test func theShapeIsFlatBoxOrPlanetAndSeparateFromTheKind() {
        let s = state([
            record("a", geometry: "sphere"), record("b", geometry: "box", dims: 3), record("c", geometry: "box", dims: 2),
            record("d", geometry: "box", dims: nil), record("e", played: true, geometry: "sphere"), record("f", geometry: nil, dims: nil),
        ])
        #expect(s.row(named: "a")?.shape == .planet && s.row(named: "b")?.shape == .box && s.row(named: "c")?.shape == .flat)
        #expect(s.row(named: "d")?.shape == .flat, "a config without dims is a run from before 3D")
        #expect(s.row(named: "e")?.shape == .planet && s.row(named: "e")?.kind == .played)
        #expect(s.row(named: "f")?.shape == nil, "an unreadable config says nothing about the shape")
    }

    @Test func aPlayedWorldSaysWhatItWasMadeFrom() {
        let s = state([record("p", played: true, source: "runs/seed:30000")])
        #expect(s.row(named: "p")?.madeFrom == SourceRef(name: "seed", step: 30_000))
    }

    // MARK: States

    @Test func everyStateHasAPlainNameAndASentence() {
        #expect(WorldState.allCases.map(\.title) == ["Saved", "Live", "Paused", "Starting", "Growing", "Saving", "In use elsewhere", "Unfinished"])
        #expect(Set(WorldState.allCases.map(\.title)).count == WorldState.allCases.count)
        for state in WorldState.allCases { #expect(state.meaning.hasSuffix(".") && state.meaning.count > 20, "\(state)") }
    }

    @Test func whatThisAppDoesWinsOverWhatOthersDo() {
        let records = [record("w", played: true), record("r")]
        #expect(state(records).row(named: "w")?.state == .saved)
        #expect(state(records, held: ["w"]).row(named: "w")?.state == .inUseElsewhere)
        #expect(state(records, written: ["r"]).row(named: "r")?.state == .inUseElsewhere)
        // the app's own world holds the lock too: it is Live, never "in use elsewhere"
        #expect(state(records, live: LiveFacts(openWorld: "w", openPhase: .live), held: ["w"]).row(named: "w")?.state == .live)
        #expect(state(records, live: LiveFacts(openWorld: "w", openPhase: .paused), held: ["w"]).row(named: "w")?.state == .paused)
        #expect(state(records, live: LiveFacts(openWorld: "w", openPhase: .starting), held: ["w"]).row(named: "w")?.state == .starting)
        #expect(state(records, live: LiveFacts(saving: ["w"]), held: ["w"]).row(named: "w")?.state == .saving)
        #expect(state(records, live: LiveFacts(growing: "r"), written: ["r"]).row(named: "r")?.state == .growing)
    }

    @Test func theRunThisAppGrowsIsListedBeforeItsFirstSavePoint() {
        let s = state([record("fresh", steps: [], status: "running")], live: LiveFacts(growing: "fresh"))
        #expect(s.row(named: "fresh")?.state == .growing && s.row(named: "fresh")?.step == nil)
        #expect(s.row(named: "fresh")?.facts.contains("no save point yet") == true)
        #expect(state([record("fresh", steps: [], status: "running")]).rows.isEmpty)
    }

    @Test func howAResearchRunEndedIsSaidInPlainWords() {
        let s = state([record("a", status: "extinct"), record("b", status: "violation"), record("c", status: "running"), record("d", status: "done"),
                       record("e", status: "interrupted")])
        #expect(s.row(named: "a")?.note == "Every organism died out.")
        #expect(s.row(named: "b")?.note == "Stopped by an error check.")
        #expect(s.row(named: "c")?.note == "Cut off without a clean stop.")
        #expect(s.row(named: "d")?.note == nil && s.row(named: "e")?.note == nil)
        // only a world that is just saved has an ending to tell
        #expect(state([record("a", status: "extinct")], written: ["a"]).row(named: "a")?.note == nil)
    }

    // MARK: Unfinished builds

    @Test func anUnfinishedBuildSaysWhereItStopped() {
        let cases: [(LibraryRecord, String)] = [
            (record("a", steps: [], status: "interrupted", unfinished: true, mark: "planet"),
             "Cancelled before its first save point, while the world was being copied or its weather was warming up."),
            (record("b", steps: [5], status: "interrupted", unfinished: true, mark: "planet"),
             "Cancelled after its first save point, before the world was marked ready to play."),
            (record("c", steps: [], status: "running", unfinished: true, mark: nil),
             "Cut off right at the start, before anything was written."),
            (record("d", steps: [], status: "violation", unfinished: true, mark: "preset"),
             "Stopped by an error check before its first save point, while the world was being copied or its weather was warming up."),
        ]
        for (r, expected) in cases { #expect(LibraryState.unfinishedReason(r) == expected) }
        let s = state(cases.map(\.0))
        #expect(s.unfinished.count == 4 && s.rows.isEmpty)
        // the one with a save point is still a source for a planet, as in the launcher
        #expect(s.catalog.sources.map(\.name) == ["b"])
    }

    @Test func thisSessionsOwnAccountOfABuildComesFirst() {
        let r = record("a", steps: [], status: "interrupted", unfinished: true, mark: "planet")
        #expect(LibraryState.unfinishedReason(r, sessionNote: "You cancelled it")
            == "You cancelled it before its first save point, while the world was being copied or its weather was warming up.")
        #expect(state([r], notes: ["a": "It failed"]).unfinished.first?.reason.hasPrefix("It failed before") == true)
    }

    @Test func aBuildThatHoldsItsLockIsBeingMadeNotUnfinished() {
        let r = record("a", steps: [], unfinished: true, mark: "planet")
        let s = state([r], held: ["a"])
        #expect(s.unfinished.isEmpty && s.rows.isEmpty && s.others == ["a"])
    }

    @Test func anUnfinishedBuildKeepsItsSource() {
        var r = record("a", steps: [], unfinished: true, mark: "planet")
        r.parent = SourceRef(name: "seed", step: 30_000)
        #expect(state([r]).unfinished.first?.source == SourceRef(name: "seed", step: 30_000))
    }

    // MARK: Order

    @Test func theWorldYouAreInIsFirstThenWhatGrowsThenTheMostRecent() {
        let records = [record("old", saved: 0), record("new", saved: 500), record("open", played: true, saved: -100), record("grows", saved: -200),
                       record("busy", saved: 1000)]
        let s = state(records, live: LiveFacts(openWorld: "open", growing: "grows"), held: ["busy"])
        #expect(s.rows.map(\.name) == ["open", "grows", "busy", "new", "old"])
    }

    @Test func aTieInTimeIsBrokenByName() {
        #expect(state([record("b"), record("a"), record("c")]).rows.map(\.name) == ["a", "b", "c"])
    }

    // MARK: Nesting

    @Test func aResearchWorldNestsUnderThePlayedWorldMadeFromIt() {
        let records = [record("seed", saved: 0), record("planet", played: true, source: "runs/seed:100", saved: 10), record("alone", saved: 5)]
        let tree = LibraryState.tree(state(records).rows)
        #expect(tree.map(\.row.name) == ["planet", "alone"])
        #expect(tree.first?.children.map(\.name) == ["seed"])
    }

    @Test func aSeedHasOneParentTheGrowOneIfThereIsOneElseTheNewest() {
        let records = [
            record("a-grow", saved: 0), record("a", played: true, source: "runs/a-grow:1", saved: 1), record("b", played: true, source: "runs/a-grow:1", saved: 50),
            record("shared", saved: 0), record("old", played: true, source: "runs/shared:1", saved: 1), record("new", played: true, source: "runs/shared:1", saved: 9),
        ]
        let tree = LibraryState.tree(state(records).rows)
        func children(_ name: String) -> [String] { tree.first { $0.row.name == name }?.children.map(\.name) ?? [] }
        #expect(children("a") == ["a-grow"] && children("b") == [], "the pipeline's name wins over a newer maker")
        #expect(children("new") == ["shared"] && children("old") == [])
        #expect(tree.count == 4, "a world is listed once")
    }

    @Test func aPlayedWorldMadeFromAPlayedWorldDoesNotNest() {
        let records = [record("first", played: true), record("second", played: true, source: "runs/first:5")]
        #expect(LibraryState.tree(state(records).rows).map(\.row.name).sorted() == ["first", "second"])
    }

    @Test func aSourceThatIsNotHereLeavesTheWorldAlone() {
        let records = [record("p", played: true, source: "runs/missing:5")]
        let tree = LibraryState.tree(state(records).rows)
        #expect(tree.count == 1 && tree[0].children.isEmpty)
    }

    // MARK: Filter and search

    @Test func theFiltersAreKindsAndUnfinished() {
        let s = state([record("p", played: true), record("r"), record("u", steps: [], unfinished: true, mark: "planet")])
        #expect(s.counts == LibraryCounts(all: 2, played: 1, research: 1, unfinished: 1))
        #expect(s.visible(filter: .all, query: "").map(\.name).sorted() == ["p", "r"])
        #expect(s.visible(filter: .played, query: "").map(\.name) == ["p"])
        #expect(s.visible(filter: .research, query: "").map(\.name) == ["r"])
        #expect(s.visible(filter: .unfinished, query: "").isEmpty)
        #expect(s.unfinishedRows(query: "").map(\.name) == ["u"])
    }

    @Test func searchMatchesEveryWordInAnyOfNameKindShapeStateOrSource() {
        let s = state([record("mini3s_a", played: true, source: "runs/grow_t3s:5", geometry: "sphere"), record("grow_t3s", geometry: "sphere"),
                       record("dev_s1", geometry: "box", dims: 2)], held: ["mini3s_a"])
        func names(_ q: String) -> [String] { s.visible(filter: .all, query: q).map(\.name).sorted() }
        #expect(names("MINI") == ["mini3s_a"])
        #expect(names("research planet") == ["grow_t3s"])
        #expect(names("flat") == ["dev_s1"])
        #expect(names("in use") == ["mini3s_a"])
        #expect(names("grow_t3s") == ["grow_t3s", "mini3s_a"], "a played world is found by the run it was made from")
        #expect(names("planet nothing") == [])
        #expect(names("   ").count == 3)
    }

    @Test func anEmptyListSaysWhatToDo() {
        let none = LibraryCounts()
        #expect(WorldsGuide.emptyList(filter: .all, query: "zz", counts: none).contains("\u{201C}zz\u{201D}"))
        #expect(WorldsGuide.emptyList(filter: .played, query: "", counts: none).contains("Make planet…"))
        #expect(WorldsGuide.emptyList(filter: .research, query: "", counts: none).contains("Lab"))
        #expect(WorldsGuide.emptyList(filter: .unfinished, query: "", counts: none).hasPrefix("No unfinished builds"))
    }

    // MARK: The cards' reasons

    @Test func aGreyMakePlanetOnACardAlwaysHasItsReason() {
        let catalogState = state([record("r"), record("nosave", steps: [])], live: LiveFacts(growing: "nosave"))
        let facts = WorldsGuide.Facts(folderExists: true, engineReady: true, catalog: catalogState.catalog)
        let r = catalogState.row(named: "r")!
        #expect(WorldsGuide.reasonToMakePlanet(from: r, in: facts) == nil)
        #expect(WorldsGuide.reasonToMakePlanet(from: catalogState.row(named: "nosave")!, in: facts) == "No save point to start from yet.")
        let open = WorldsGuide.Facts(folderExists: true, engineReady: true, catalog: catalogState.catalog, openWorld: "r")
        #expect(WorldsGuide.reasonToMakePlanet(from: r, in: open) == "Close it first")
        #expect(WorldsGuide.reasonToMakePlanet(from: catalogState.row(named: "nosave")!, in: open) == "Close r first")
        let saving = WorldsGuide.Facts(folderExists: true, engineReady: true, catalog: catalogState.catalog, saving: ["r"])
        #expect(WorldsGuide.reasonToMakePlanet(from: r, in: saving) == "Saving...")
        let noEngine = WorldsGuide.Facts(folderExists: true, engineReady: false, catalog: catalogState.catalog)
        #expect(WorldsGuide.reasonToMakePlanet(from: r, in: noEngine) == "Set up the engine first.")
    }

    @Test func makingABuildAgainSaysWhyNot() {
        let s = state([record("seed")])
        let facts = WorldsGuide.Facts(folderExists: true, engineReady: true, catalog: s.catalog)
        #expect(WorldsGuide.reasonToMakeAgain(source: "seed", sourceHasSavePoint: true, in: facts) == nil)
        #expect(WorldsGuide.reasonToMakeAgain(source: nil, sourceHasSavePoint: false, in: facts) == "It does not say which world it was made from.")
        #expect(WorldsGuide.reasonToMakeAgain(source: "seed", sourceHasSavePoint: false, in: facts) == "seed is no longer here with a save point.")
    }

    @Test func theSidebarLineIsTheStateOrWhenItWasSaved() {
        let s = state([record("a"), record("b")], live: LiveFacts(growing: "b"))
        #expect(s.row(named: "a")?.sidebarLine(now: now.addingTimeInterval(7200)) == "Saved 2 hours ago")
        #expect(s.row(named: "a")?.sidebarLine(now: now) == "Saved just now")
        #expect(s.row(named: "b")?.sidebarLine(now: now) == "Growing")
        #expect(s.row(named: "a")?.facts == "Research · Planet · step 100")
    }
}
