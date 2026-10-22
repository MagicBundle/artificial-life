import AlifeCore
import Foundation
import Testing

@testable import AlifeControl

/// The launcher's logic: what it lists, what it lets be resumed, and the planet it makes.
@MainActor
@Suite struct WorldLauncherTests {
    private static let playJSON = """
        {"play_version": 1, "base_rate": 14.4, "cadence": {"metrics_every": 250, "heavy_metrics_every": 250,
         "checkpoint_every": 6000, "keep_checkpoints": 3}, "tick": 25, "weather": {"dc_ref": 0.5}, "planet": null}
        """

    private func write(_ text: String, _ name: String, in dir: URL) throws {
        try FileManager.default.createDirectory(at: dir.appendingPathComponent(name).deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try text.write(to: dir.appendingPathComponent(name), atomically: true, encoding: .utf8)
    }

    private func runs(_ scratch: Scratch) throws -> URL {
        let runs = scratch.url.appendingPathComponent("runs")
        let world = runs.appendingPathComponent("planet_a")
        try write(Self.playJSON, "play.json", in: world)
        try write(#"{"world": {"dims": 3}, "weather": {"enabled": true}}"#, "config.json", in: world)
        try write("x", "checkpoints/step_0000506000.safetensors", in: world)
        let research = runs.appendingPathComponent("64k3d_s0")
        try write(#"{"world": {"dims": 3}, "weather": {"enabled": false}}"#, "config.json", in: research)
        try write("x", "checkpoints/step_0000250000.safetensors", in: research)
        try write("x", "checkpoints/step_0000500000.safetensors", in: research)
        try write("{}", "events/chunk_0.json", in: research)
        let bare = runs.appendingPathComponent("bare")
        try write(#"{"world": {"dims": 3}, "weather": {"enabled": false}}"#, "config.json", in: bare)
        try write("x", "checkpoints/step_0000000100.safetensors", in: bare)
        return runs
    }

    private final class Launches {
        var requests: [PlayRequest] = []
    }

    private func make(_ runs: URL?, launches: Launches) -> WorldLauncherModel {
        WorldLauncherModel(runsDirectory: { runs }, launch: { launches.requests.append($0) })
    }

    @Test func theLauncherListsTheWorldsAndTheSourcesItFinds() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let launches = Launches()
        let model = make(try runs(scratch), launches: launches)
        model.refresh()
        #expect(model.catalog.worlds.map(\.name) == ["planet_a"])
        #expect(Set(model.catalog.sources.map(\.name)) == ["planet_a", "64k3d_s0", "bare"])
    }

    @Test func noRepositoryIsAnEmptyList() {
        let launches = Launches()
        let model = make(nil, launches: launches)
        model.refresh()
        #expect(model.catalog == PlayCatalog())
        #expect(model.planet.source.isEmpty)
    }

    @Test func aWorldIsResumedAtOneTimes() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let launches = Launches()
        let model = make(try runs(scratch), launches: launches)
        model.refresh()
        let world = try #require(model.catalog.worlds.first)
        #expect(model.obstacle(to: world) == nil)
        model.resume(world)
        #expect(launches.requests == [.resume(PlayResume(run: "planet_a", rate: .multiplier(1)))])
    }

    @Test func aWorldThatIsSavingOrBeingPlayedCannotBeResumed() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let launches = Launches()
        let runsDirectory = try runs(scratch)
        let model = make(runsDirectory, launches: launches)
        model.refresh(saving: ["planet_a"])
        let world = try #require(model.catalog.worlds.first)
        #expect(model.obstacle(to: world) == "saving…")
        model.resume(world)
        #expect(launches.requests.isEmpty, "the retired session's directory is held until its process exits")
        model.refresh(saving: [])
        #expect(model.obstacle(to: world) == nil)

        // Another process holds the lock (a second app, or a play from a terminal).
        let fd = open(runsDirectory.appendingPathComponent("planet_a/play.lock").path, O_RDWR | O_CREAT, 0o644)
        defer { close(fd) }
        #expect(flock(fd, LOCK_EX | LOCK_NB) == 0)
        model.refresh()
        let locked = try #require(model.catalog.worlds.first)
        #expect(model.obstacle(to: locked) == "being played")
        model.resume(locked)
        #expect(launches.requests.isEmpty)
    }

    // MARK: Planets

    @Test func aPlanetStartsFromTheNewestCheckpointOfTheFirstSourceWithTheWeatherOn() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let model = make(try runs(scratch), launches: Launches())
        model.refresh()
        model.selectSource("64k3d_s0")
        #expect(model.planet.sourceStep == 500_000)
        #expect(model.planet.storm && model.planet.weatherSpinup == PlanetSpec.defaultSpinup)
        #expect(model.planet.directorEnabled && model.planet.houses == .auto)
        model.selectCheckpoint(250_000)
        #expect(model.planet.sourceStep == 250_000)
    }

    @Test func aSourceWhoseWeatherIsOnIsForkedAsItIsWithoutASpinup() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let model = make(try runs(scratch), launches: Launches())
        model.refresh()
        model.selectSource("planet_a")
        #expect(!model.planet.storm && model.planet.weatherSpinup == 0)
        model.planet.name = "planet_c"
        #expect(model.planetErrors.isEmpty)
        model.setStorm(true)
        #expect(model.planet.weatherSpinup == PlanetSpec.defaultSpinup)
        model.setStorm(false)
        #expect(model.planet.weatherSpinup == 0)
    }

    @Test func theSuggestedNameIsFree() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runsDirectory = try runs(scratch)
        let model = make(runsDirectory, launches: Launches())
        model.refresh()
        // `planet_a` is a world already: the next free name derived from it.
        #expect(model.planet.name == "planet_a-2")
        #expect(!model.catalog.existingNames.contains(model.planet.name))
        #expect(model.planetErrors.isEmpty, "\(model.planetErrors)")
    }

    @Test func aPlanetIsLaunchedOnlyWhenItIsValid() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let launches = Launches()
        let model = make(try runs(scratch), launches: launches)
        model.refresh()
        model.selectSource("64k3d_s0")
        model.planet.name = "bad name"
        model.startPlanet()
        #expect(launches.requests.isEmpty && !model.planetErrors.isEmpty)
        model.planet.name = "planet_b"
        model.startPlanet()
        guard case .planet(let spec)? = launches.requests.first else {
            Issue.record("no planet was launched")
            return
        }
        #expect(spec.name == "planet_b" && spec.source == "64k3d_s0" && spec.sourceStep == 500_000)
        #expect(PlayLaunch.arguments(planet: spec).contains("runs/64k3d_s0:500000"))
    }

    @Test func aNameThatExistsIsRefused() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let model = make(try runs(scratch), launches: Launches())
        model.refresh()
        model.selectSource("64k3d_s0")
        model.planet.name = "planet_a"
        #expect(model.planetErrors == ["'runs/planet_a' exists."])
    }

    @Test func theHousesNoteSaysWhenTheExactRebuildIsImpossible() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let model = make(try runs(scratch), launches: Launches())
        model.refresh()
        model.selectSource("64k3d_s0")
        #expect(model.housesNote == nil, "it has event chunks")
        model.selectSource("bare")
        #expect(model.housesNote?.contains("clades") == true)
        model.planet.houses = .exact
        #expect(model.housesNote?.contains("cannot be rebuilt") == true)
        model.planet.houses = .clade
        #expect(model.housesNote == nil)
    }

    @Test func aSourceThatVanishedIsReplacedByTheFirstOne() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let model = make(try runs(scratch), launches: Launches())
        model.refresh()
        model.selectSource("bare")
        model.planet.source = "gone"
        model.refresh()
        #expect(model.catalog.sources.contains { $0.name == model.planet.source })
    }
}
