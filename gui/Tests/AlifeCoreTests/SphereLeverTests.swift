import Foundation
import Testing

@testable import AlifeCore

/// The sphere's levers, presets, launch lines and places in the GUI (DESIGN §46.8, §46.9): what the Lab passes for a planet, what a fork may
/// not change, how the launcher picks the weather overlay, and the longitude and latitude of a power.
@Suite struct SphereLeverTests {
    private func tiny3S() throws -> World3DLevers { World3DLevers(preset: PresetFile(text: try Repo.preset("tiny3s"))) }

    private func newRun(_ world3D: World3DLevers, _ edit: (inout RunSettings) -> Void = { _ in }) -> RunSettings {
        var s = RunSettings()
        s.mode = .new
        s.name = "t"
        s.seed = "1"
        s.selectPreset(.tiny3s, world3D: world3D)
        edit(&s)
        return s
    }

    private static let run = ["run", "--preset", "tiny3s", "--seed", "1", "--out", "runs/t"]

    // MARK: The preset

    @Test func theSphereValuesAreReadFromThePresetFile() throws {
        let w = try tiny3S()
        #expect(w.sphere.isSphere && w.sphere.geometry == "sphere" && w.dims == 3)
        #expect(w.sphere.radius == "36" && w.sphere.nside == "8" && w.sphere.layers == "8" && w.sphere.ceilingBand == "0" && w.height == "16")
        #expect(w.validate().isEmpty)
        #expect(World3DLevers.preset("tiny3s", repo: Repo.root) == w)
        #expect(Preset(rawValue: "tiny3s") == .tiny3s && Preset.allCases.contains(.tiny3s))
    }

    /// The three planets of the engine's `config.PRESETS` (DESIGN §46.9): each is a sphere preset of its own shape, in the picker and on the command line.
    @Test(arguments: [
        ("tiny3s", Preset.tiny3s, "36", "16", "8", "8"),
        ("mini3s", Preset.mini3s, "48", "24", "11", "12"),
        ("64k3s", Preset.k64_3s, "90", "40", "21", "18"),
    ])
    func everySpherePresetIsAPlanetOfItsShape(name: String, preset: Preset, radius: String, height: String, nside: String, layers: String) throws {
        let w = World3DLevers(preset: PresetFile(text: try Repo.preset(name)))
        #expect(w.sphere.isSphere && w.dims == 3 && w.height == height)
        #expect(w.sphere.radius == radius && w.sphere.nside == nside && w.sphere.layers == layers && w.sphere.ceilingBand == "0")
        #expect(w.validate().isEmpty)
        #expect(World3DLevers.preset(name, repo: Repo.root) == w)
        #expect(preset.rawValue == name && Preset(rawValue: name) == preset)
        var s = RunSettings()
        s.mode = .new
        s.name = "t"
        s.seed = "1"
        s.selectPreset(preset, world3D: w)
        #expect(s.validate(existingRuns: []).isEmpty)
        #expect(s.arguments(existingRuns: []) == ["run", "--preset", name, "--seed", "1", "--out", "runs/t"])
        #expect(s.worldDims == 3)
    }

    @Test(arguments: ["tiny3d", "64k3d", "mini3d", "tiny", "dev"])
    func aBoxPresetIsNotASphere(name: String) throws {
        let w = World3DLevers(preset: PresetFile(text: try Repo.preset(name)))
        #expect(!w.sphere.isSphere && w.sphere.geometry == "box" && w.sphere == SphereLevers())
        #expect(w.sphere.validate().isEmpty && w.sphere.setArguments(baseline: SphereLevers()).isEmpty)
        #expect(w.sphere.summary == "box")
    }

    @Test func anUnchangedPlanetPassesThePresetNameAlone() throws {
        let s = newRun(try tiny3S())
        #expect(s.validate(existingRuns: []).isEmpty)
        #expect(s.arguments(existingRuns: []) == Self.run)
        #expect(s.worldDims == 3)
    }

    @Test func aChangedLeverIsOneSetAndTheGeometryIsNeverPassed() throws {
        let base = try tiny3S()
        var w = base
        w.sphere.radius = "40"
        w.sphere.nside = "10"
        w.sphere.layers = "6"
        w.sphere.ceilingBand = "1.5"
        w.sphere.geometry = "box"  // never an input: even if it were changed, nothing is passed for it
        // (a box geometry makes the struct inert: nothing at all is passed)
        #expect(w.sphere.setArguments(baseline: base.sphere).isEmpty)
        w.sphere.geometry = "sphere"
        let args = newRun(base) { $0.levers.world3D = w }.arguments(existingRuns: [])
        #expect(args == Self.run + ["--set", "world.radius=40.0", "--set", "world.nside=10", "--set", "world.layers=6", "--set", "physics.ceiling_band=1.5"])
        #expect(!args.joined(separator: " ").contains("world.geometry"))
        var same = base
        same.sphere.radius = "36.0"
        #expect(newRun(base) { $0.levers.world3D = same }.arguments(existingRuns: []) == Self.run, "compared as numbers: 36.0 is 36")
    }

    // MARK: The rules a field can break

    @Test func eachSphereRuleHasItsMessage() throws {
        let base = try tiny3S()
        func errors(_ edit: (inout SphereLevers) -> Void) -> [String] {
            var w = base
            edit(&w.sphere)
            return w.validate()
        }
        #expect(errors { $0.radius = "19.9" }.contains { $0.contains("world.radius must be at least 20") })
        #expect(errors { $0.radius = "20" }.isEmpty)
        #expect(errors { $0.radius = "big" }.contains { $0.contains("world.radius must be a plain decimal") })
        #expect(errors { $0.nside = "1" }.contains { $0.contains("world.nside must be at least 2") })
        #expect(errors { $0.nside = "2" }.isEmpty)
        #expect(errors { $0.nside = "2.5" }.contains { $0.contains("world.nside must be an integer") })
        #expect(errors { $0.layers = "1" }.contains { $0.contains("world.layers must be at least 2") })
        #expect(errors { $0.layers = "" }.contains { $0.contains("world.layers must be an integer") })
        #expect(errors { $0.ceilingBand = "-1" }.contains { $0.contains("physics.ceiling_band") })
        #expect(errors { $0.ceilingBand = "0.5" }.isEmpty)
        // a box ignores them all
        var box = base
        box.sphere = SphereLevers()
        box.dims = 2
        #expect(box.validate().isEmpty)
        var s = newRun(base) { $0.levers.world3D.sphere.radius = "3" }
        #expect(!s.validate(existingRuns: []).isEmpty)
        s = newRun(base)
        #expect(s.validate(existingRuns: []).isEmpty)
    }

    // MARK: A fork and a resume

    private func configJSON(geometry: String? = "sphere", radius: Double = 36, nside: Int = 8, layers: Int = 8, band: Double? = nil) -> Data {
        var world: [String: Any] = ["dims": 3, "height": 16.0, "size": 1024.0]
        if let geometry {
            world["geometry"] = geometry
            world["radius"] = radius
            world["nside"] = nside
            world["layers"] = layers
        }
        // the rest of a 3D world, so that nothing but the sphere's rules can complain
        var physics: [String: Any] = ["gravity": 0.03, "buoyancy_max": 0.04]
        if let band { physics["ceiling_band"] = band }
        var root: [String: Any] = [
            "world": world, "physics": physics, "energy": ["buoyancy_cost": 16, "buoyancy_cap_cost": 8],
            "fields": ["benthic_lambda": 8.0, "photic_amp": 0.5, "photic_lambda": 8.0],
            "mutation": ["buoy_init": 0.5, "buoy_sigma": 0.02],
        ]
        root["world"] = world
        return try! JSONSerialization.data(withJSONObject: root)
    }

    @Test func aForkReadsTheSourcesPlanetAndCannotChangeItsShape() throws {
        let levers = try Levers(configJSON: configJSON(radius: 48, nside: 11, layers: 12, band: 2))
        #expect(levers.world3D.sphere == SphereLevers(geometry: "sphere", radius: "48", nside: "11", layers: "12", ceilingBand: "2"))
        var s = RunSettings()
        s.mode = .fork
        s.name = "f"
        s.sourceRun = "src"
        s.seed = "2"
        s.leverBaseline = levers
        s.levers = levers
        #expect(s.validate(existingRuns: ["src"]).isEmpty)
        #expect(s.arguments(existingRuns: ["src"]) == ["run", "--init-from", "runs/src", "--seed", "2", "--out", "runs/f"], "unchanged: nothing passed")
        // the shape is the state's: an edit of the radius, the Nside or the layers does not reach the line; the band does
        s.levers.world3D.sphere.radius = "99"
        s.levers.world3D.sphere.nside = "3"
        s.levers.world3D.sphere.layers = "3"
        s.levers.world3D.sphere.geometry = "box"
        s.levers.world3D.sphere.ceilingBand = "4"
        #expect(s.effectiveLevers.world3D.sphere == SphereLevers(geometry: "sphere", radius: "48", nside: "11", layers: "12", ceilingBand: "4"))
        #expect(s.arguments(existingRuns: ["src"]) == ["run", "--init-from", "runs/src", "--seed", "2", "--out", "runs/f", "--set", "physics.ceiling_band=4.0"])
    }

    @Test func aBoxRunsConfigIsABoxAndItsLineIsUnchanged() throws {
        let levers = try Levers(configJSON: configJSON(geometry: nil))
        #expect(levers.world3D.sphere == SphereLevers() && !levers.world3D.sphere.isSphere)
        var s = RunSettings()
        s.mode = .fork
        s.name = "f"
        s.sourceRun = "src"
        s.seed = "1"
        s.leverBaseline = levers
        s.levers = levers
        #expect(s.arguments(existingRuns: ["src"]) == ["run", "--init-from", "runs/src", "--seed", "1", "--out", "runs/f"])
    }

    @Test func aWrongTypeInAConfigIsRefusedAsTheEngineWould() {
        for (what, text) in [("geometry", #"{"world": {"geometry": 3}}"#), ("radius", #"{"world": {"geometry": "sphere", "radius": "big"}}"#),
                             ("nside", #"{"world": {"nside": 8.5}}"#), ("band", #"{"physics": {"ceiling_band": true}}"#)] {
            #expect(throws: LeverConfigError.self, "\(what)") { try Levers(configJSON: Data(text.utf8)) }
        }
    }

    // MARK: The launcher

    @Test func theLauncherPicksTheWeatherOverlayByTheSourcesGeometry() {
        #expect(PlanetSpec.overlay(forGeometry: "sphere", worldSize: 1024) == "stormsph")
        #expect(PlanetSpec.overlay(forGeometry: "sphere", worldSize: 160) == "stormsph", "the size does not matter on a planet")
        #expect(PlanetSpec.overlay(forGeometry: "box", worldSize: 160) == "storm160")
        #expect(PlanetSpec.overlay(forGeometry: "box", worldSize: 320) == "storm")
        #expect(PlanetSpec.overlay(forGeometry: nil, worldSize: nil) == "storm", "an unknown source is a box")
        var planet = PlanetSpec(source: "s", sourceStep: 100, name: "p", sourceWorldSize: 1024, sourceDims: 3, sourceGeometry: "sphere")
        #expect(planet.validate(existing: []).isEmpty)
        let args = PlayLaunch.arguments(planet: planet)
        #expect(args == ["play", "--init-from", "runs/s:100", "--overlay", "stormsph", "--set", "director.enabled=true", "--weather-spinup", "20000",
                         "--houses", "auto", "--out", "runs/p", "--rate", "1"])
        planet.sourceGeometry = nil
        planet.sourceWorldSize = 320
        #expect(PlayLaunch.arguments(planet: planet).contains("storm") && !PlayLaunch.arguments(planet: planet).contains("stormsph"))
        planet.storm = false
        planet.weatherSpinup = 0
        #expect(!PlayLaunch.arguments(planet: planet).contains("--overlay"), "a world whose weather is on is forked as it is")
    }

    @Test func theCatalogShowsEachWorldsGeometry() throws {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        func make(_ name: String, geometry: String?, radius: Double = 36) throws {
            let dir = scratch.url.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: dir.appendingPathComponent("checkpoints"), withIntermediateDirectories: true)
            FileManager.default.createFile(atPath: dir.appendingPathComponent("checkpoints/step_100.safetensors").path, contents: Data([0]))
            var world: [String: Any] = ["dims": 3, "size": 1024.0]
            if let geometry {
                world["geometry"] = geometry
                world["radius"] = radius
            }
            try JSONSerialization.data(withJSONObject: ["world": world, "weather": ["enabled": false]]).write(to: dir.appendingPathComponent("config.json"))
            let play = #"{"play_version":1,"base_rate":14.4,"cadence":{"metrics_every":10,"heavy_metrics_every":10,"checkpoint_every":50,"keep_checkpoints":3},"tick":25}"#
            try Data(play.utf8).write(to: dir.appendingPathComponent("play.json"))
        }
        try make("planet", geometry: "sphere", radius: 48)
        try make("flat", geometry: nil)
        try make("boxed", geometry: "box")
        let catalog = PlayCatalogScanner.scan(runsDirectory: scratch.url)
        let planet = try #require(catalog.sources.first { $0.name == "planet" })
        #expect(planet.geometry == "sphere" && planet.planetRadius == 48)
        #expect(catalog.sources.first { $0.name == "flat" }?.geometry == "box" && catalog.sources.first { $0.name == "flat" }?.planetRadius == nil)
        #expect(catalog.sources.first { $0.name == "boxed" }?.geometry == "box")
        #expect(catalog.worlds.first { $0.name == "planet" }?.geometry == "sphere" && catalog.worlds.first { $0.name == "planet" }?.planetRadius == 48)
        #expect(catalog.worlds.first { $0.name == "flat" }?.geometry == "box")
    }

    // MARK: Places

    private let patches: [PlayEnvironment.Patch] = [
        PlayEnvironment.Patch(i: 0, x: 1, y: 0, z: 0, lon: 0, lat: 0, gain: 1, sigma: 12),
        PlayEnvironment.Patch(i: 1, x: 0, y: 0, z: -1, lon: nil, lat: nil, gain: 1, sigma: 12),  // placed by the unit vector (the south pole)
        PlayEnvironment.Patch(i: 2, x: 0, y: 1, z: 0, lon: 90, lat: 0, gain: 1, sigma: 12),
    ]

    @Test func theNearestPatchIsByGreatCircle() {
        #expect(PowersModel.nearestPatch(lon: 10, lat: 5, in: patches) == 0)
        #expect(PowersModel.nearestPatch(lon: 80, lat: -10, in: patches) == 2)
        #expect(PowersModel.nearestPatch(lon: 170, lat: -80, in: patches) == 1, "near the south pole, whatever the longitude")
        let across = [PlayEnvironment.Patch(i: 0, x: 0, y: 0, z: 1, lon: 175, lat: 0, gain: 1, sigma: 1),
                      PlayEnvironment.Patch(i: 1, x: 0, y: 0, z: 1, lon: -120, lat: 0, gain: 1, sigma: 1)]
        #expect(PowersModel.nearestPatch(lon: -179, lat: 0, in: across) == 0, "across the antimeridian the arc is the short way")
        #expect(PowersModel.nearestPatch(lon: 0, lat: 0, in: []) == nil)
        // a box patch (no z) is no patch of a planet
        #expect(PowersModel.nearestPatch(lon: 0, lat: 0, in: [PlayEnvironment.Patch(i: 0, x: 1, y: 2, gain: 1, sigma: 1)]) == nil)
    }

    @Test func aStrikeNamesALongitudeAndALatitudeAndASignedLineIsSent() throws {
        let offer = try #require(PowerOffer.offers(header: try powersHeader(), states: [:]).first { $0.name == .strike })
        var model = PowersModel()
        _ = model.choose(offer)
        let aimed = model.aim(lon: -30.5, lat: -60, patches: patches)
        let command = try #require(aimed, "a column power takes the place as it is")
        let line = try command.line()
        #expect(line.hasPrefix("power preview strike id=") && line.contains("x=-30.5") && line.contains("y=-60"))
        #expect(model.target == .point(x: -30.5, y: -60))
    }

    @Test func aBloomGoesToTheNearestPatch() throws {
        let offer = try #require(PowerOffer.offers(header: try powersHeader(), states: [:]).first { $0.name == .bloom })
        var model = PowersModel()
        _ = model.choose(offer)
        let aimed = model.aim(lon: 85, lat: 3, patches: patches)
        let command = try #require(aimed)
        #expect(try command.line().contains("patch=2"))
        let none = model.aim(lon: 85, lat: 3, patches: [])
        #expect(none == nil)
    }

    private func powersHeader() throws -> PlayHeader {
        var play = try #require(WorldFixtures.store("3d_g3b").header?.play)
        play.powers = [
            "bloom": .init(action: "fertility_shift", label: "Bloom", cost: 1, cooldown: 3000, scope: "patch", duration: [3000, 12000],
                           params: ["gain": .array([.number(1.5), .number(2.0)])]),
            "strike": .init(action: "catastrophe", label: "Strike", cost: 1, cooldown: 3000, scope: "column", duration: nil,
                            params: ["radius": .array([.number(6.25), .number(25)]), "kill_p": .array([.number(0.3), .number(0.6)])]),
        ]
        return play
    }

    @Test func onlyAPlaceMayBeSignedInAControlLine() throws {
        let strike = PlayCommand.power(PowerCommand(.apply, .strike, id: 4, [.init(.x, -120.25), .init(.y, -45), .init(.radius, 10), .init(.kill, 0.5)]))
        #expect(try strike.line() == "power strike id=4 x=-120.25 y=-45 radius=10 kill=0.5")
        for key in [PowerCommand.Key.radius, .kill, .gain, .mult, .steps, .patch] {
            let bad = PlayCommand.power(PowerCommand(.apply, .strike, id: 1, [.init(key, -1)]))
            #expect(throws: PlayCommand.Invalid.self, "\(key) is not signed") { try bad.line() }
        }
    }
}

extension SphereLevers {
    /// A sphere's values by name (a test's baseline).
    init(geometry: String, radius: String, nside: String, layers: String, ceilingBand: String) {
        self.init()
        self.geometry = geometry
        self.radius = radius
        self.nside = nside
        self.layers = layers
        self.ceilingBand = ceilingBand
    }
}
