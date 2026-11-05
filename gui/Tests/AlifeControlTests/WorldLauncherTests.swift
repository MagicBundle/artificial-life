import AlifeCore
import AppKit
import Foundation
import SwiftUI
import Testing

@testable import AlifeControl

/// The launcher model's logic (the set-up page's planet form and the Worlds page use it): what it lists, what it lets be resumed, and the planet it makes.
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

    @Test func aPlayedWorldAsThePlanetsSourceSaysToContinueItWithResume() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let model = make(try runs(scratch), launches: Launches())
        model.refresh()
        model.selectSource("planet_a")
        #expect(model.sourceIsWorld, "planet_a has a play.json: it is a world")
        #expect(WorldLauncherModel.continueHint.hasPrefix("To continue this world, use Resume. Make planet starts a new branch"))
        model.selectSource("64k3d_s0")
        #expect(!model.sourceIsWorld, "a research run has no play.json: there is nothing to continue")
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

    @Test func aMiniWorldSourceGetsItsOverlayAndALongerSpinup() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = scratch.url.appendingPathComponent("runs")
        for (name, size) in [("mini3d_s0", "160.0"), ("64k3d_s0", "320.0")] {
            let dir = runs.appendingPathComponent(name)
            try write(#"{"world": {"dims": 3, "size": \#(size)}, "weather": {"enabled": false}}"#, "config.json", in: dir)
            try write("x", "checkpoints/step_0000450000.safetensors", in: dir)
        }
        let launches = Launches()
        let model = make(runs, launches: launches)
        model.refresh()
        model.selectSource("mini3d_s0")
        model.planet.name = "planet_m"
        #expect(model.planet.sourceWorldSize == 160 && model.planet.weatherSpinup == 36_000)
        #expect(PlayLaunch.arguments(planet: model.planet).starts(with: [
            "play", "--init-from", "runs/mini3d_s0:450000", "--overlay", "storm160", "--set", "director.enabled=true",
            "--weather-spinup", "36000",
        ]))
        model.setStorm(false)
        model.setStorm(true)
        #expect(model.planet.weatherSpinup == 36_000, "switching the overlay off and on keeps the source's default")
        model.selectSource("64k3d_s0")
        model.planet.name = "planet_n"
        #expect(model.planet.sourceWorldSize == 320 && model.planet.weatherSpinup == PlanetSpec.defaultSpinup)
        #expect(PlayLaunch.arguments(planet: model.planet).contains("storm"))
        #expect(!PlayLaunch.arguments(planet: model.planet).contains("storm160"))
    }

    @Test func aRefreshFollowsTheSizeOfTheSelectedSourceWhenItsConfigChanges() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = scratch.url.appendingPathComponent("runs")
        let dir = runs.appendingPathComponent("mini3d_s0")
        let config = { (size: String) in
            try self.write(#"{"world": {"dims": 3, "size": \#(size)}, "weather": {"enabled": false}}"#, "config.json", in: dir)
        }
        try config("160.0")
        try write("x", "checkpoints/step_0000450000.safetensors", in: dir)
        let model = make(runs, launches: Launches())
        model.refresh()
        #expect(model.planet.source == "mini3d_s0" && model.planet.sourceWorldSize == 160)
        #expect(model.planet.weatherSpinup == 36_000)
        try config("320.0")
        model.refresh()
        #expect(model.planet.sourceWorldSize == 320, "the size follows the config of the kept selection")
        #expect(model.planet.weatherSpinup == PlanetSpec.defaultSpinup, "an untouched default follows the size")
        model.planet.name = "planet_m"
        #expect(PlayLaunch.arguments(planet: model.planet).contains("storm"))
        #expect(!PlayLaunch.arguments(planet: model.planet).contains("storm160"))
        model.planet.weatherSpinup = 5_000
        try config("160.0")
        model.refresh()
        #expect(model.planet.sourceWorldSize == 160 && model.planet.weatherSpinup == 5_000, "a user's spin-up stays")
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

    @Test func aTwoDimensionalSourceIsForkedWithoutTheWeatherAndSaysWhy() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        let flat = runs.appendingPathComponent("flat2d")
        try write(#"{"world": {"dims": 2, "size": 512.0}, "weather": {"enabled": false}}"#, "config.json", in: flat)
        try write("x", "checkpoints/step_0000001000.safetensors", in: flat)
        let launches = Launches()
        let model = make(runs, launches: launches)
        model.refresh()
        model.selectSource("flat2d")
        model.planet.name = "planet_f"
        #expect(!model.planet.storm && model.planet.weatherSpinup == 0, "a 2D source starts with the weather off")
        #expect(model.weatherObstacle != nil && model.planetErrors.isEmpty)
        #expect(!PlayLaunch.arguments(planet: model.planet).contains("--overlay"))
        model.setStorm(true)
        #expect(model.planetErrors.contains(PlanetSpec.weatherObstacle(dims: 2)!), "forcing it on is refused before launch")
        model.selectSource("64k3d_s0")
        #expect(model.planet.storm && model.weatherObstacle == nil, "a 3D source gets the weather again")
    }

    @Test func aPlanetPicksItsWeatherOverlayByShapeAndAnUncalibratedOneIsForkedWithoutWeatherAndSaysWhy() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = try runs(scratch)
        func planet(_ name: String, radius: Double, height: Int, nside: Int, layers: Int, weather: Bool = false) throws {
            try write(#"{"world": {"dims": 3, "size": 1024.0, "geometry": "sphere", "radius": \#(radius), "height": \#(height), "nside": \#(nside), "layers": \#(layers)}, "weather": {"enabled": \#(weather)}}"#,
                      "config.json", in: runs.appendingPathComponent(name))
            try write("x", "checkpoints/step_0000000200.safetensors", in: runs.appendingPathComponent(name))
        }
        try planet("tiny_s0", radius: 36, height: 16, nside: 8, layers: 8)
        try planet("64k_s0", radius: 90, height: 40, nside: 21, layers: 18)
        try planet("mini_s0", radius: 48, height: 24, nside: 11, layers: 12)
        try planet("k128_s0", radius: 128.6, height: 40, nside: 30, layers: 18)
        try planet("odd_s0", radius: 60, height: 30, nside: 14, layers: 14)
        try planet("mini_wet", radius: 48, height: 24, nside: 11, layers: 12, weather: true)
        let model = make(runs, launches: Launches())
        model.refresh()
        model.planet.name = "planet_t"
        // the spin-up is the planet's own, by its shape (S4c): the table of `PlanetSpec.sphereSpinups`, one length per overlay row
        for (source, overlay, shape) in [("tiny_s0", "stormsph16", PlanetShape(radius: 36, height: 16, nside: 8, layers: 8)),
                                         ("64k_s0", "stormsph", PlanetShape(radius: 90, height: 40, nside: 21, layers: 18)),
                                         ("mini_s0", "stormsph22", PlanetShape(radius: 48, height: 24, nside: 11, layers: 12)),
                                         ("k128_s0", "stormsph128", PlanetShape(radius: 128.6, height: 40, nside: 30, layers: 18))] {
            model.selectSource(source)
            let spinup = try #require(PlanetSpec.sphereSpinups.first { $0.shape == shape }).steps
            #expect(model.planet.storm && model.weatherObstacle == nil && model.planetErrors.isEmpty, "\(source)")
            #expect(model.planet.weatherSpinup == spinup, "\(source) starts at its planet's spin-up")
            #expect(PlayLaunch.arguments(planet: model.planet).starts(with: ["play", "--init-from", "runs/\(source):200", "--overlay", overlay,
                                                                           "--set", "director.enabled=true", "--weather-spinup", "\(spinup)"]))
            // switching the overlay off and on keeps the planet's default, and a user's value stays across a refresh
            model.setStorm(false)
            model.setStorm(true)
            #expect(model.planet.weatherSpinup == spinup)
        }
        model.selectSource("odd_s0")
        let obstacle = try #require(model.weatherObstacle)
        #expect(obstacle.hasPrefix("No weather is calibrated yet for this planet size"))
        #expect(!model.planet.storm && model.planet.weatherSpinup == 0 && model.planetErrors.isEmpty, "the planet is forked without weather")
        #expect(!PlayLaunch.arguments(planet: model.planet).contains("--overlay") && !PlayLaunch.arguments(planet: model.planet).contains("--weather-spinup"))
        model.setStorm(true)
        #expect(model.planetErrors.contains(obstacle), "forcing it on is refused before launch")
        // a source whose weather is on is forked as it is, whatever its size: no caption, no overlay
        model.selectSource("mini_wet")
        #expect(model.weatherObstacle == nil && !model.planet.storm && model.planetErrors.isEmpty)
        #expect(!PlayLaunch.arguments(planet: model.planet).contains("--overlay"))
        #expect(SetupModel.shapeNote(try #require(model.selectedSource)) == "The new planet keeps this world's shape, a sphere.")
        let tiny = try #require(model.catalog.sources.first { $0.name == "tiny_s0" })
        #expect(SetupModel.shapeNote(tiny).hasSuffix("with the weather ready for that size."))
        let mini = try #require(model.catalog.sources.first { $0.name == "mini_s0" })
        #expect(SetupModel.shapeNote(mini).hasSuffix("with the weather ready for that size."))
        let odd = try #require(model.catalog.sources.first { $0.name == "odd_s0" })
        #expect(SetupModel.shapeNote(odd) == "The new planet keeps this world's shape, a sphere.")
        // a refresh follows a changed shape of the selected source
        model.selectSource("tiny_s0")
        try planet("tiny_s0", radius: 36, height: 20, nside: 8, layers: 8)
        model.refresh()
        #expect(model.planet.sourceShape?.height == 20 && !model.planet.storm && model.weatherObstacle != nil)
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

    // MARK: What the model lists

    @Test func researchRunsAreNotListedAsUnfinishedBuildsButTheirNamesAreTaken() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runsDirectory = try runs(scratch)
        // The runs of an archive whose checkpoints are elsewhere, and a planet build that was cancelled.
        for name in ["eval_s0_B", "run-20261011-1137"] {
            try write(#"{"world": {"dims": 3}}"#, "config.json", in: runsDirectory.appendingPathComponent(name))
            try write(#"{"status": "done"}"#, "meta.json", in: runsDirectory.appendingPathComponent(name))
        }
        let cancelled = runsDirectory.appendingPathComponent("planet_x")
        try write(#"{"status": "interrupted", "play_build": "planet"}"#, "meta.json", in: cancelled)
        try write(#"{"world": {"dims": 3}}"#, "config.json", in: cancelled)
        let model = make(runsDirectory, launches: Launches())
        model.refresh()
        #expect(model.catalog.unfinished == ["planet_x"])
        model.selectSource("64k3d_s0")
        model.planet.name = "eval_s0_B"
        #expect(model.planetErrors == ["'runs/eval_s0_B' exists."], "a name the engine would refuse is refused here")
    }
}
