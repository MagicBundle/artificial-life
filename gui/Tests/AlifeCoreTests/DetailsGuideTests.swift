import Foundation
import Testing

@testable import AlifeCore

@Suite struct DetailsGuideTests {
    private func row(_ kind: WorldKind, shape: WorldShape? = .planet, state: WorldState = .saved, step: Int? = 30_000, name: String = "w",
                     target: Int? = nil, from: SourceRef? = nil, parent: SourceRef? = nil, unmade: String? = nil, note: String? = nil) -> LibraryRow {
        var r = LibraryRow(name: name, kind: kind, shape: shape, state: state, note: note, step: step, lastSaved: Date(), madeFrom: from, targetSteps: target)
        r.parent = parent
        r.unmadePlanet = unmade
        return r
    }

    // MARK: Names

    @Test func theGrownWorldOfAPlanetIsNamedForItAndItsFolderIsSmallType() {
        #expect(DetailsGuide.title("planet-1-grow") == "Grown world of planet-1")
        #expect(DetailsGuide.title("planet-1") == "planet-1" && DetailsGuide.title("-grow") == "-grow")
        #expect(DetailsGuide.folderLine("planet-1-grow") == "runs/planet-1-grow")
    }

    // MARK: What it is

    @Test func aPlayedWorldSaysWhatItCanDoAndWhereItCameFrom() {
        let sentences = DetailsGuide.whatItIs(row(.played, from: SourceRef(name: "p-grow", step: 30_000)), grownFor: nil)
        #expect(sentences[0].contains("A world you play") && sentences[0].contains("powers"))
        #expect(sentences[1] == "It was made from the grown world of p at step 30,000.")
        // a planet made from the world that was grown for it
        let own = DetailsGuide.whatItIs(row(.played, name: "p", from: SourceRef(name: "p-grow", step: 30_000)), grownFor: nil)
        #expect(own[1] == "It was made from its grown world at step 30,000.")
        #expect(DetailsGuide.whatItIs(row(.played, name: "P", from: SourceRef(name: "p-grow", step: nil)), grownFor: nil)[1] == "It was made from its grown world.")
        #expect(DetailsGuide.whatItIs(row(.played), grownFor: nil).count == 1)
    }

    @Test func aResearchWorldSaysItCannotBePlayedAndHowToPlayIt() {
        let plain = DetailsGuide.whatItIs(row(.research), grownFor: nil)
        #expect(plain[0].contains("research world") && plain[0].contains("cannot be played"))
        #expect(plain[1] == "To play it, make a planet from it.")
        let seed = DetailsGuide.whatItIs(row(.research, name: "p-grow"), grownFor: "p")
        #expect(seed[1] == "It is the grown world of p. To play it, make a planet from it.")
        let unmade = DetailsGuide.whatItIs(row(.research, name: "p-grow", unmade: "p"), grownFor: nil)
        #expect(unmade[1].contains("The planet p was not made"))
    }

    @Test func aFlatWorldSaysItHasNoWeather() {
        let sentences = DetailsGuide.whatItIs(row(.research, shape: .flat), grownFor: "ignored")
        #expect(sentences[1] == WorldsGuide.flatHasNoPlanet && sentences.count == 2)
    }

    @Test func aWorldThatDiedOutDoesNotSayItCanBecomeAPlanet() {
        var dead = row(.research, state: .saved)
        dead.diedOut = true
        let sentences = DetailsGuide.whatItIs(dead, grownFor: "p")
        #expect(sentences[1] == WorldsGuide.diedOutHasNoPlanet && sentences.count == 2 && !sentences.joined().contains("To play it"))
    }

    @Test func aBranchSaysWhereItWasBranchedFrom() {
        let sentences = DetailsGuide.whatItIs(row(.research, parent: SourceRef(name: "a", step: 20_000)), grownFor: nil)
        #expect(sentences.last == "Branched from a at step 20,000.")
        #expect(DetailsGuide.whatItIs(row(.research, parent: SourceRef(name: "a", step: nil)), grownFor: nil).last == "Branched from a.")
    }

    @Test func theBranchesOfAWorldAreTheResearchWorldsThatNameItAsParent() {
        let rows = [row(.research, name: "a"), row(.research, name: "b", parent: SourceRef(name: "A", step: 1)),
                    row(.research, name: "c", parent: SourceRef(name: "a", step: 2)), row(.research, name: "d", parent: SourceRef(name: "x", step: 2)),
                    row(.research, name: "a", parent: SourceRef(name: "a", step: 3))]
        // (capitals do not tell worlds apart, as on the disk; a world is not its own branch)
        #expect(DetailsGuide.branches(of: "a", in: rows) == ["b", "c"])
        #expect(DetailsGuide.branches(of: "nobody", in: rows).isEmpty)
        var played = row(.played, name: "p")
        played.parent = SourceRef(name: "a", step: 1)
        #expect(DetailsGuide.branches(of: "a", in: [played]).isEmpty)
    }

    // MARK: How it is doing

    @Test func progressSaysTheStateInAFullSentence() {
        let now = Date()
        #expect(DetailsGuide.progress(row(.research, state: .growing, step: 12_300, target: 30_000), now: now) == "Growing now: step 12,300 of 30,000.")
        #expect(DetailsGuide.progress(row(.research, state: .growing, step: 30_000, target: 30_000), now: now) == "Growing now: step 30,000.")
        #expect(DetailsGuide.progress(row(.research, state: .growing, step: 5), now: now) == "Growing now: step 5.")
        #expect(DetailsGuide.progress(row(.research, state: .growing, step: nil), now: now) == "Growing now.")
        #expect(DetailsGuide.progress(row(.played, state: .live, step: 450_000), now: now) == "Open in Watch and running at step 450,000.")
        #expect(DetailsGuide.progress(row(.played, state: .paused, step: nil), now: now) == "Open in Watch, paused.")
        #expect(DetailsGuide.progress(row(.played, state: .inUseElsewhere, step: 7), now: now) == "Another app or process is using it (step 7).")
        #expect(DetailsGuide.progress(row(.research, state: .saved, step: 30_000), now: now.addingTimeInterval(7200)).hasPrefix("Saved at step 30,000, "))
        #expect(DetailsGuide.progress(row(.research, state: .saved, step: nil), now: now) == "It has no save point yet.")
        #expect(DetailsGuide.progress(row(.played, state: .saving), now: now).contains("last save point"))
        #expect(DetailsGuide.progress(row(.played, state: .starting), now: now).contains("warm-up"))
    }

    @Test func theEnginesStatusIsAnEndNote() {
        #expect(DetailsGuide.ended(status: "done")?.contains("asked to grow to") == true)
        #expect(DetailsGuide.ended(status: "interrupted")?.contains("stopped") == true)
        #expect(DetailsGuide.ended(status: "extinct")?.contains("died out") == true)
        #expect(DetailsGuide.ended(status: "violation")?.contains("Check world") == true)
        #expect(DetailsGuide.ended(status: "running") == nil && DetailsGuide.ended(status: nil) == nil)
        // a played world is closed at a save point as a matter of course; only a death or a violation is news
        #expect(DetailsGuide.ended(status: "interrupted", played: true) == nil && DetailsGuide.ended(status: "done", played: true) == nil)
        #expect(DetailsGuide.ended(status: "extinct", played: true) != nil && DetailsGuide.ended(status: "violation", played: true) != nil)
    }

    @Test func theNumbersCaptionSaysWhyThereAreNone() {
        #expect(DetailsGuide.numbersCaption(hasMetrics: true, growing: true).contains("two seconds"))
        #expect(DetailsGuide.numbersCaption(hasMetrics: true, growing: false).contains("last row"))
        #expect(DetailsGuide.numbersCaption(hasMetrics: false, growing: true).contains("No numbers yet"))
        #expect(DetailsGuide.numbersCaption(hasMetrics: false, growing: false).contains("no numbers file"))
    }

    // MARK: The settings it was made with

    private static let config = """
        {"run": {"seed": 7, "steps": 30000}, "world": {"dims": 3, "geometry": "sphere", "radius": 36, "height": 16, "nside": 8, "layers": 8, "size": 64},
         "mutation": {"rate_min": 0.001, "sigma_min": 0.002}, "physics": {"gravity": 0.03, "buoyancy_max": 0.04},
         "director": {"persona": "trickster", "rule_every": 1000, "llm_every": 25000, "enabled": true}}
        """

    private func world(_ config: String, meta: String? = #"{"status": "done", "created_utc": "2026-11-01T16:29:19Z"}"#) throws -> (URL, () -> Void) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("madewith-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try config.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        if let meta { try meta.write(to: dir.appendingPathComponent("meta.json"), atomically: true, encoding: .utf8) }
        return (dir, { try? FileManager.default.removeItem(at: dir) })
    }

    @Test func theSettingsAWorldWasMadeWithAreReadFromItsConfig() throws {
        let (dir, cleanup) = try world(Self.config)
        defer { cleanup() }
        let made = try #require(MadeWith.read(runDirectory: dir))
        #expect(made.seed == 7 && made.steps == 30_000)
        #expect(made.shape == "Planet, radius 36")
        #expect(made.created == ISO8601DateFormatter().date(from: "2026-11-01T16:29:19Z"))
        #expect(made.autopilot == "Trickster, deciding every 1,000 steps, with a local model every 25,000 steps")
        let byKey = Dictionary(uniqueKeysWithValues: made.settings.map { ($0.key, $0) })
        #expect(byKey["mutation.rate_min"]?.title == "Mutation rate floor" && byKey["mutation.rate_min"]?.value == "0.001")
        #expect(byKey["physics.gravity"]?.value == "0.03" && byKey["world.radius"]?.value == "36")
        #expect(byKey["world.height"] != nil, "a 3D world lists its 3D settings")
    }

    @Test func aFlatWorldListsNoThreeDimensionalSettingsAndNoAutopilotWhenItHadNone() throws {
        let (dir, cleanup) = try world(#"{"run": {"seed": 1, "steps": 500}, "world": {"dims": 2, "geometry": "box", "size": 256}, "mutation": {"rate_min": 0.01}}"#, meta: nil)
        defer { cleanup() }
        let made = try #require(MadeWith.read(runDirectory: dir))
        #expect(made.shape == "Flat, 256 wide" && made.autopilot == "Off" && made.created == nil)
        #expect(!made.settings.contains { $0.key.hasPrefix("world.") || $0.key == "physics.gravity" })
        #expect(made.settings.contains { $0.key == "mutation.rate_min" })
    }

    @Test func aBoxSaysItsSizeAndHeight() throws {
        let (dir, cleanup) = try world(#"{"run": {"seed": 1}, "world": {"dims": 3, "geometry": "box", "size": 128, "height": 24}}"#)
        defer { cleanup() }
        #expect(try #require(MadeWith.read(runDirectory: dir)).shape == "3D box, 128 wide, 24 high")
    }

    @Test func aConfigThatCannotBeReadIsNil() throws {
        let (dir, cleanup) = try world("not json")
        defer { cleanup() }
        #expect(MadeWith.read(runDirectory: dir) == nil)
        #expect(MadeWith.read(runDirectory: dir.appendingPathComponent("nowhere")) == nil)
    }

    @Test func theAutopilotLineNamesTheStyleTheCadenceAndTheModel() {
        var s = DirectorSettings()
        s.persona = .gardener
        #expect(MadeWith.autopilot(s) == "Gardener, deciding every 1,000 steps")
        s.mode = .replay
        #expect(MadeWith.autopilot(s).hasSuffix(", replaying its log"))
        s.mode = .live
        s.llmOn = true
        #expect(MadeWith.autopilot(s) == "Gardener, deciding every 1,000 steps, with a local model every 25,000 steps")
    }
}

/// A research world's `meta.json` parent reaches its library row (the Branches list of Details reads it).
@Suite struct LibraryParentTests {
    private func record(_ name: String, played: Bool, parent: String?) -> LibraryRecord {
        let cadence = PlayWorldFile.Cadence(metricsEvery: 1, heavyMetricsEvery: 1, checkpointEvery: 1, keepCheckpoints: 1)
        let play = played ? PlayWorldFile(playVersion: 1, baseRate: 14.4, cadence: cadence, tick: 1, weather: nil, planet: nil) : nil
        return LibraryRecord(
            name: name, play: play, steps: [100], status: "done", parent: SourceRef.parse(parent), dims: 3, geometry: "sphere", worldSize: nil,
            planetRadius: nil, planetShape: nil, weatherOn: false, hasEvents: true, directoryModified: Date(), lastSaved: Date(), hasContent: true,
            looksUnfinished: false, buildMark: nil)
    }

    @Test func aResearchWorldKeepsItsParentAndAPlayedWorldDoesNot() {
        var snapshot = LibrarySnapshot()
        snapshot.records = [record("branch", played: false, parent: "runs/seed:20000"), record("seed", played: false, parent: nil),
                            record("planet", played: true, parent: "runs/seed:20000")]
        let state = LibraryState.resolve(snapshot)
        #expect(state.row(named: "branch")?.parent == SourceRef(name: "seed", step: 20_000))
        #expect(state.row(named: "seed")?.parent == nil)
        #expect(state.row(named: "planet")?.parent == nil, "a played world says where it came from through its play file, not as a branch")
        #expect(DetailsGuide.branches(of: "seed", in: state.rows) == ["branch"])
    }
}
