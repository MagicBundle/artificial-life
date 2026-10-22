import Foundation
import Testing

@testable import AlifeCore

/// The launcher's view of `runs/`: worlds, unfinished builds, sources, and the lock.
@Suite struct PlayWorldsTests {
    private static let playJSON = """
        {"play_version": 1, "base_rate": 14.4,
         "cadence": {"metrics_every": 250, "heavy_metrics_every": 250, "checkpoint_every": 6000, "keep_checkpoints": 3},
         "tick": 25, "weather": {"dc_ref": 0.5},
         "planet": {"source": "64k3d_s0:500000", "step": 500000, "weather_spinup": 20000, "houses": "exact",
                    "made_at_step": 500000}}
        """

    private func write(_ text: String, _ name: String, in dir: URL) throws {
        try FileManager.default.createDirectory(at: dir.appendingPathComponent(name).deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try text.write(to: dir.appendingPathComponent(name), atomically: true, encoding: .utf8)
    }

    private func checkpoint(_ step: Int, in dir: URL) throws {
        try write("x", "checkpoints/step_\(String(format: "%010d", step)).safetensors", in: dir)
    }

    private func make(_ scratch: ScratchDirectory) throws -> URL {
        let runs = scratch.url.appendingPathComponent("runs")
        // A planet: play.json, two checkpoints, no events.
        let planet = runs.appendingPathComponent("planet_a")
        try write(Self.playJSON, "play.json", in: planet)
        try write(#"{"status": "interrupted", "start_step": 500000}"#, "meta.json", in: planet)
        try write(#"{"world": {"dims": 3}, "weather": {"enabled": true}}"#, "config.json", in: planet)
        try checkpoint(500_000, in: planet)
        try checkpoint(506_000, in: planet)
        // A research run: checkpoints and events, no play.json.
        let research = runs.appendingPathComponent("64k3d_s0")
        try write(#"{"status": "done", "start_step": 0}"#, "meta.json", in: research)
        try write(#"{"world": {"dims": 3}, "weather": {"enabled": false}}"#, "config.json", in: research)
        try checkpoint(250_000, in: research)
        try checkpoint(500_000, in: research)
        try write("{}", "events/chunk_0.json", in: research)
        // A planet whose build was cancelled: config and meta, no checkpoint, no play.json.
        let cancelled = runs.appendingPathComponent("planet_b")
        try write(#"{"status": "interrupted", "start_step": 500000}"#, "meta.json", in: cancelled)
        try write(#"{"world": {"dims": 3}}"#, "config.json", in: cancelled)
        // A directory that is nothing of ours, and a file.
        try FileManager.default.createDirectory(at: runs.appendingPathComponent("empty"), withIntermediateDirectories: true)
        try write("x", "notes.txt", in: runs)
        return runs
    }

    @Test func playJSONIsDecodedAsTheEngineWritesIt() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let url = scratch.url.appendingPathComponent("play.json")
        try Self.playJSON.write(to: url, atomically: true, encoding: .utf8)
        let file = try #require(PlayWorldFile.read(at: url))
        #expect(file.baseRate == 14.4 && file.tick == 25 && file.weather?.dcRef == 0.5)
        #expect(file.cadence.checkpointEvery == 6000 && file.cadence.keepCheckpoints == 3)
        #expect(file.planet?.houses == "exact" && file.planet?.source == "64k3d_s0:500000")
        #expect(file.planet?.weatherSpinup == 20_000 && file.planet?.madeAtStep == 500_000)
    }

    @Test func aPresetWorldHasNoPlanetAndNoWeatherScale() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let url = scratch.url.appendingPathComponent("play.json")
        try """
            {"play_version": 1, "base_rate": 14.4, "cadence": {"metrics_every": 250, "heavy_metrics_every": 250,
             "checkpoint_every": 6000, "keep_checkpoints": 3}, "tick": 25, "weather": null, "planet": null}
            """.write(to: url, atomically: true, encoding: .utf8)
        let file = try #require(PlayWorldFile.read(at: url))
        #expect(file.weather == nil && file.planet == nil)
    }

    @Test(arguments: [#"{"play_version": 2}"#, "not json", "[]", #"{"play_version": 1}"#])
    func aFileThatIsNotAPlayJSONOfVersionOneIsNone(text: String) throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let url = scratch.url.appendingPathComponent("play.json")
        try text.write(to: url, atomically: true, encoding: .utf8)
        #expect(PlayWorldFile.read(at: url) == nil)
        #expect(PlayWorldFile.read(at: scratch.url.appendingPathComponent("missing.json")) == nil)
    }

    @Test func theScanSeparatesWorldsUnfinishedBuildsAndSources() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let catalog = PlayCatalogScanner.scan(runsDirectory: try make(scratch))
        #expect(catalog.worlds.map(\.name) == ["planet_a"])
        let world = try #require(catalog.worlds.first)
        #expect(world.newestStep == 506_000 && world.status == "interrupted" && !world.locked)
        #expect(world.file.planet?.houses == "exact")
        #expect(catalog.unfinished == ["planet_b"], "no play.json and no checkpoint: a build that did not finish")
        #expect(Set(catalog.sources.map(\.name)) == ["planet_a", "64k3d_s0"])
        let research = try #require(catalog.sources.first { $0.name == "64k3d_s0" })
        #expect(research.checkpoints == [250_000, 500_000] && research.dims == 3)
        #expect(research.hasEvents && !research.weatherOn)
        let planet = try #require(catalog.sources.first { $0.name == "planet_a" })
        #expect(planet.weatherOn && !planet.hasEvents && planet.checkpoints == [500_000, 506_000])
        #expect(catalog.existingNames == ["planet_a", "planet_b", "64k3d_s0"])
    }

    @Test func aMissingRunsDirectoryIsAnEmptyCatalog() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        #expect(PlayCatalogScanner.scan(runsDirectory: scratch.url.appendingPathComponent("nothing")) == PlayCatalog())
    }

    @Test func checkpointStepsAreReadFromTheFileNames() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        try checkpoint(12_000, in: scratch.url)
        try checkpoint(6000, in: scratch.url)
        try write("x", "checkpoints/LATEST", in: scratch.url)
        try write("x", "checkpoints/step_abc.safetensors", in: scratch.url)
        try write("x", "checkpoints/step_0000000001.tmp", in: scratch.url)
        #expect(PlayCatalogScanner.checkpointSteps(in: scratch.url) == [6000, 12_000])
    }

    // MARK: The lock

    @Test func aHeldLockIsSeenAndAFreeOneIsNot() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let dir = scratch.url
        #expect(!PlayLockProbe.isHeld(directory: dir), "no lock file")
        let path = dir.appendingPathComponent("play.lock").path
        let fd = open(path, O_RDWR | O_CREAT, 0o644)
        #expect(fd >= 0)
        defer { close(fd) }
        #expect(!PlayLockProbe.isHeld(directory: dir), "a stale file locks nothing")
        #expect(flock(fd, LOCK_EX | LOCK_NB) == 0)
        #expect(PlayLockProbe.isHeld(directory: dir))
        #expect(PlayLockProbe.isHeld(directory: dir), "probing does not release it")
        #expect(flock(fd, LOCK_UN) == 0)
        #expect(!PlayLockProbe.isHeld(directory: dir))
    }

    @Test func theScanMarksAWorldWhoseLockIsHeld() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let runs = try make(scratch)
        let fd = open(runs.appendingPathComponent("planet_a/play.lock").path, O_RDWR | O_CREAT, 0o644)
        defer { close(fd) }
        #expect(flock(fd, LOCK_EX | LOCK_NB) == 0)
        #expect(PlayCatalogScanner.scan(runsDirectory: runs).worlds.first?.locked == true)
        #expect(PlayLockProbe.held(among: ["planet_a", "64k3d_s0", "gone"], in: runs) == ["planet_a"])
    }
}

@Suite struct WorldLaunchArgumentTests {
    @Test func theWorldIsTheArgumentAfterTheFlag() {
        #expect(WorldLaunchArgument.parse(["--world", "planet_a"]) == "planet_a")
        #expect(WorldLaunchArgument.parse(["x", "--world=planet.b-2", "y"]) == "planet.b-2")
        #expect(WorldLaunchArgument.parse(["--world", "a", "--world", "b"]) == "b")
    }

    @Test func nothingElseIsAWorld() {
        #expect(WorldLaunchArgument.parse([]) == nil)
        #expect(WorldLaunchArgument.parse(["--world"]) == nil)
        #expect(WorldLaunchArgument.parse(["--world", "../etc"]) == nil)
        #expect(WorldLaunchArgument.parse(["--world", "a b"]) == nil)
        #expect(WorldLaunchArgument.parse(["--world", "-x"]) == nil)
        #expect(WorldLaunchArgument.parse(["--world="]) == nil)
        #expect(WorldLaunchArgument.parse(["--worlds", "a"]) == nil)
    }
}
