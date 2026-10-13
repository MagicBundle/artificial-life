import Foundation
import Testing

@testable import AlifeCore

/// The 3D levers and their argument rules: a new run diffs them against its preset, a fork against its source with
/// the world's dims and height fixed, a resume has none, and a 2D world ignores them.
@Suite struct World3DLeverTests {
    /// The `tiny3d` values as `World3DLevers(preset:)` reads them from the repository's preset file.
    private func tiny3D() throws -> World3DLevers {
        World3DLevers(preset: PresetFile(text: try Repo.preset("tiny3d")))
    }

    private func newRun(preset: Preset = .tiny3d, world3D: World3DLevers, _ edit: (inout RunSettings) -> Void = { _ in })
        -> RunSettings
    {
        var s = RunSettings()
        s.mode = .new
        s.name = "t"
        s.seed = "1"
        s.selectPreset(preset, world3D: world3D)
        edit(&s)
        return s
    }

    private static let tiny3DRun = ["run", "--preset", "tiny3d", "--seed", "1", "--out", "runs/t"]

    // MARK: Preset values

    @Test func tiny3DIsReadFromItsPresetFile() throws {
        let w = try tiny3D()
        #expect(w.dims == 3 && w.height == "24")
        #expect(w.gravity == "0.03" && w.buoyancyMax == "0.04")
        #expect(w.buoyancyCost == "16" && w.buoyancyCapCost == "8")
        #expect(w.benthicLambda == "8" && w.photicAmp == "0.5" && w.photicLambda == "8")
        #expect(w.buoyInit == "0.5" && w.buoySigma == "0.02")
        #expect(w.validate().isEmpty)
        #expect(World3DLevers.preset("tiny3d", repo: Repo.root) == w)
        #expect(World3DLevers.preset("no-such-preset", repo: Repo.root) == nil)
    }

    @Test(arguments: ["tiny", "dev", "64k", "256k"])
    func aTwoDPresetIsInert(name: String) throws {
        let w = World3DLevers(preset: PresetFile(text: try Repo.preset(name)))
        #expect(w == World3DLevers(), "a 2D preset sets none of the 3D keys")
        #expect(w.dims == 2 && w.setArguments(baseline: World3DLevers()) == [])
    }

    // MARK: New run

    @Test func anUnchangedThreeDPresetPassesNoLeverArguments() throws {
        let s = newRun(world3D: try tiny3D())
        #expect(s.validate(existingRuns: []).isEmpty)
        #expect(s.arguments(existingRuns: []) == Self.tiny3DRun)
        #expect(s.worldDims == 3)
    }

    @Test func changedLeversPassOnlyTheirDifferenceFromThePreset() throws {
        let s = newRun(world3D: try tiny3D()) {
            $0.levers.world3D.gravity = "0.02"
            $0.levers.world3D.height = "32"
            $0.levers.world3D.photicAmp = "0.25"
        }
        #expect(s.arguments(existingRuns: []) == Self.tiny3DRun + [
            "--set", "world.height=32.0", "--set", "physics.gravity=0.02", "--set", "fields.photic_amp=0.25",
        ])
    }

    @Test func aValueEqualToTheEngineDefaultIsPassedWhenThePresetDiffers() throws {
        // The reason the 3D levers diff against the preset: tiny3d's gravity is 0.03, so 0 is a change to pass.
        let s = newRun(world3D: try tiny3D()) { $0.levers.world3D.gravity = "0"; $0.levers.world3D.photicAmp = "0.0" }
        #expect(s.arguments(existingRuns: []) == Self.tiny3DRun + [
            "--set", "physics.gravity=0.0", "--set", "fields.photic_amp=0.0",
        ])
        #expect(s.validate(existingRuns: []).isEmpty)
    }

    @Test func aValueEqualToThePresetIsNotPassedWhateverItsSpelling() throws {
        let s = newRun(world3D: try tiny3D()) {
            $0.levers.world3D.gravity = "0.030"
            $0.levers.world3D.height = "24.0"
            $0.levers.world3D.buoyancyCost = "016"
            $0.levers.world3D.buoyancyMax = "4e-2"
        }
        #expect(s.arguments(existingRuns: []) == Self.tiny3DRun)
    }

    @Test func everyLeverHasItsKeyInOrder() throws {
        let s = newRun(world3D: try tiny3D()) {
            $0.levers.rateMin = "0.01"  // the 2D levers come first
            $0.levers.world3D.height = "48"
            $0.levers.world3D.gravity = "0.01"
            $0.levers.world3D.buoyancyMax = "0.05"
            $0.levers.world3D.buoyancyCost = "20"
            $0.levers.world3D.buoyancyCapCost = "6"
            $0.levers.world3D.benthicLambda = "6"
            $0.levers.world3D.photicAmp = "0.8"
            $0.levers.world3D.photicLambda = "12"
        }
        #expect(s.validate(existingRuns: []).isEmpty)
        #expect(s.arguments(existingRuns: []) == Self.tiny3DRun + [
            "--set", "mutation.rate_min=0.01",
            "--set", "world.height=48.0",
            "--set", "physics.gravity=0.01",
            "--set", "physics.buoyancy_max=0.05",
            "--set", "energy.buoyancy_cost=20",
            "--set", "energy.buoyancy_cap_cost=6",
            "--set", "fields.benthic_lambda=6.0",
            "--set", "fields.photic_amp=0.8",
            "--set", "fields.photic_lambda=12.0",
        ])
    }

    @Test func switchingTheActuatorOffZeroesTheKeysTheEngineTiesToIt() throws {
        let off = newRun(world3D: try tiny3D()) {
            $0.levers.world3D.buoyancyMax = "0"
            $0.levers.world3D.buoyancyCost = "0"
            $0.levers.world3D.buoyancyCapCost = "0"
        }
        #expect(off.validate(existingRuns: []).isEmpty)
        #expect(off.arguments(existingRuns: []) == Self.tiny3DRun + [
            "--set", "physics.buoyancy_max=0.0", "--set", "energy.buoyancy_cost=0", "--set", "energy.buoyancy_cap_cost=0",
            "--set", "mutation.buoy_init=0.0", "--set", "mutation.buoy_sigma=0.0",
        ])

        // The costs are levers the user must set too: the engine rejects a cost without the actuator.
        let costly = newRun(world3D: try tiny3D()) { $0.levers.world3D.buoyancyMax = "0" }
        #expect(costly.validate(existingRuns: []) == [
            "energy.buoyancy_cost and energy.buoyancy_cap_cost must be 0 when physics.buoyancy_max is 0."
        ])

        // A baseline without the actuator has nothing to zero.
        var plain = try tiny3D()
        plain.buoyancyMax = "0"; plain.buoyancyCost = "0"; plain.buoyancyCapCost = "0"; plain.buoyInit = "0"; plain.buoySigma = "0"
        let none = newRun(world3D: plain) { $0.levers.world3D.gravity = "0.02" }
        #expect(none.arguments(existingRuns: []) == Self.tiny3DRun + ["--set", "physics.gravity=0.02"])
    }

    @Test func theDimsOfANewRunAreThePresetsWhateverWasTyped() throws {
        var s = newRun(world3D: try tiny3D())
        s.levers.world3D.dims = 2
        #expect(s.effectiveLevers.world3D.dims == 3 && s.arguments(existingRuns: []) == Self.tiny3DRun)
        // On a 2D preset the 3D levers are inert, even with values typed into them.
        var flat = newRun(preset: .dev, world3D: World3DLevers())
        flat.levers.world3D.dims = 3
        flat.levers.world3D.gravity = "garbage"
        flat.levers.world3D.height = "99"
        #expect(flat.worldDims == 2 && flat.validate(existingRuns: []).isEmpty)
        #expect(flat.arguments(existingRuns: []) == ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t"])
    }

    @Test func choosingAPresetResetsTheThreeDLeversAndKeepsTheOthers() throws {
        var s = newRun(preset: .dev, world3D: World3DLevers())
        s.levers.rateMin = "0.01"
        s.levers.world3D.gravity = "junk"
        s.selectPreset(.tiny3d, world3D: try tiny3D())
        #expect(s.preset == .tiny3d && s.presetWorld3D.dims == 3 && s.levers.world3D == s.presetWorld3D)
        #expect(s.world3DBaseline == s.presetWorld3D)
        #expect(s.levers.rateMin == "0.01")
        s.selectPreset(.dev, world3D: World3DLevers())
        #expect(s.worldDims == 2 && s.levers.world3D == World3DLevers() && s.levers.rateMin == "0.01")
    }

    @Test func resettingTheLeversKeepsAThreeDWorldValid() throws {
        var levers = Levers(world3D: try tiny3D())
        levers.rateMin = "0.02"
        levers.world3D.gravity = "0.01"
        levers.reset(world3D: try tiny3D())
        #expect(levers == Levers(world3D: try tiny3D()) && levers.validate().isEmpty)
        levers.reset()
        #expect(levers == Levers() && levers.world3D.dims == 2)
    }

    // MARK: Validation

    @Test(arguments: [
        ("height", "0", "world.height must be > 0."),
        ("height", "abc", "world.height must be a plain decimal number > 0."),
        ("gravity", "-0.1", "physics.gravity must be a plain decimal number >= 0."),
        ("gravity", "", "physics.gravity must be a plain decimal number >= 0."),
        ("buoyancyMax", "x", "physics.buoyancy_max must be a plain decimal number >= 0."),
        ("benthicLambda", "0", "fields.benthic_lambda must be > 0."),
        ("benthicLambda", "", "fields.benthic_lambda must be a plain decimal number > 0."),
        ("photicAmp", "1.5", "fields.photic_amp must be in [0, 1]."),
        ("photicAmp", "-1", "fields.photic_amp must be a plain decimal number in [0, 1]."),
        ("photicLambda", "0", "fields.photic_lambda must be > 0 when fields.photic_amp > 0."),
        ("photicLambda", "x", "fields.photic_lambda must be a plain decimal number (> 0 when photic_amp > 0)."),
        ("buoyancyCost", "65536", "energy.buoyancy_cost must be an integer in [0, 65536)."),
        ("buoyancyCost", "1.5", "energy.buoyancy_cost must be an integer in [0, 65536)."),
        ("buoyancyCapCost", "-2", "energy.buoyancy_cap_cost must be an integer in [0, 65536)."),
    ])
    func eachLeverIsChecked(field: String, value: String, message: String) throws {
        var w = try tiny3D()
        switch field {
        case "height": w.height = value
        case "gravity": w.gravity = value
        case "buoyancyMax": w.buoyancyMax = value
        case "benthicLambda": w.benthicLambda = value
        case "photicAmp": w.photicAmp = value
        case "photicLambda": w.photicLambda = value
        case "buoyancyCost": w.buoyancyCost = value
        case "buoyancyCapCost": w.buoyancyCapCost = value
        default: Issue.record("unknown field \(field)")
        }
        #expect(w.validate().contains(message), "\(field) = '\(value)': \(w.validate())")
        // The run refuses to start on it, and an invalid field is not passed.
        let s = newRun(world3D: try tiny3D()) { $0.levers.world3D = w }
        #expect(s.validate(existingRuns: []).contains(message))
    }

    @Test func photicLambdaMayBeZeroWithoutThePhoticBand() throws {
        var w = try tiny3D()
        w.photicAmp = "0"
        w.photicLambda = "0"
        #expect(w.validate().isEmpty)
    }

    @Test func aTwoDWorldAcceptsAnythingInTheThreeDFields() {
        var w = World3DLevers()
        w.height = "junk"; w.gravity = "-1"; w.benthicLambda = "0"
        #expect(w.dims == 2 && w.validate().isEmpty && w.setArguments(baseline: World3DLevers()).isEmpty)
    }

    // MARK: Pre-fill from a run's config

    static let tiny3DConfig = """
    {"world": {"dims": 3, "size": 128.0, "height": 24.0},
     "physics": {"gravity": 0.03, "floor_eps": 0.1, "buoyancy_max": 0.04},
     "energy": {"buoyancy_cost": 16, "buoyancy_cap_cost": 8},
     "mutation": {"rate_min": 0.002, "buoy_init": 0.5, "buoy_sigma": 0.02},
     "fields": {"benthic_lambda": 8.0, "photic_amp": 0.5, "photic_lambda": 8.0, "patch_drift": 0.01}}
    """

    @Test func theLeversOfAThreeDRunAreReadFromItsConfig() throws {
        let levers = try Levers(configJSON: Data(Self.tiny3DConfig.utf8))
        #expect(levers.world3D == (try tiny3D()), "the config of a tiny3d run gives the preset's values")
        #expect(levers.rateMin == "0.002" && levers.patchDrift == "0.01")
    }

    @Test func aRunFromBeforeTheThreeDKeysIsATwoDWorld() throws {
        let levers = try Levers(configJSON: Data(#"{"mutation": {"rate_min": 0.01}, "world": {"size": 128.0}}"#.utf8))
        #expect(levers.world3D == World3DLevers() && levers.rateMin == "0.01")
        #expect(try Levers(configJSON: Data("{}".utf8)) == Levers())
    }

    @Test(arguments: [
        (#"{"physics": {"gravity": "heavy"}}"#, "physics.gravity is not a number"),
        (#"{"world": {"dims": 2.5}}"#, "world.dims is not an integer"),
        (#"{"world": {"height": true}}"#, "world.height is not a number"),
        (#"{"energy": {"buoyancy_cost": 1.5}}"#, "energy.buoyancy_cost is not an integer"),
        (#"{"fields": 3}"#, "fields is not an object"),
    ])
    func aWrongTypeInTheThreeDKeysIsRefused(json: String, message: String) {
        #expect(throws: LeverConfigError(message: message)) { try Levers(configJSON: Data(json.utf8)) }
    }

    // MARK: Fork and resume

    private func fork(config: String = tiny3DConfig, _ edit: (inout RunSettings) -> Void = { _ in }) throws -> RunSettings {
        let scratch = try ScratchDirectory()
        defer { scratch.cleanup() }
        let dir = scratch.url.appendingPathComponent("src3d", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try config.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        var s = RunSettings()
        s.mode = .fork
        s.name = "child"
        s.seed = ""
        s.sourceRun = "src3d"
        let base = Levers.forkBaseline(runDirectory: dir)
        #expect(base.warning == nil)
        s.levers = base.levers
        s.leverBaseline = base.levers
        let context = DirectorContext.forRun(at: dir, mode: .fork)
        s.directorContext = context
        s.director = context.baseline
        edit(&s)
        return s
    }

    private static let forkRun = ["run", "--init-from", "runs/src3d", "--out", "runs/child"]

    @Test func anUntouchedForkOfAThreeDRunPassesNothing() throws {
        let s = try fork()
        #expect(s.worldDims == 3 && s.validate(existingRuns: ["src3d"]).isEmpty)
        let preset = try tiny3D()
        #expect(s.world3DBaseline == s.leverBaseline.world3D && s.world3DBaseline == preset)
        #expect(s.arguments(existingRuns: ["src3d"]) == Self.forkRun)
    }

    @Test func aForkKeepsItsSourcesDimsAndHeight() throws {
        let s = try fork {
            $0.levers.world3D.height = "48"
            $0.levers.world3D.dims = 2
            $0.levers.world3D.gravity = "0.02"
            $0.levers.world3D.photicAmp = "0"
        }
        #expect(s.effectiveLevers.world3D.height == "24" && s.effectiveLevers.world3D.dims == 3)
        #expect(s.arguments(existingRuns: ["src3d"]) == Self.forkRun + [
            "--set", "physics.gravity=0.02", "--set", "fields.photic_amp=0.0",
        ])
        // A height nobody can change does not fail validation either.
        let bad = try fork { $0.levers.world3D.height = "junk" }
        #expect(bad.validate(existingRuns: ["src3d"]).isEmpty)
        #expect(bad.arguments(existingRuns: ["src3d"]) == Self.forkRun)
    }

    @Test func aForkOfATwoDRunIgnoresTheThreeDLevers() throws {
        let s = try fork(config: #"{"mutation": {"rate_min": 0.002}}"#) {
            $0.levers.world3D.dims = 3
            $0.levers.world3D.gravity = "0.02"
            $0.levers.rateMin = "0.003"
        }
        #expect(s.worldDims == 2)
        #expect(s.arguments(existingRuns: ["src3d"]) == Self.forkRun + ["--set", "mutation.rate_min=0.003"])
    }

    @Test func aForkOfARunThatFlippedTheActuatorOffNeedsNoZeroing() throws {
        let config = Self.tiny3DConfig
            .replacingOccurrences(of: "\"buoyancy_max\": 0.04", with: "\"buoyancy_max\": 0.0")
            .replacingOccurrences(of: "\"buoyancy_cost\": 16, \"buoyancy_cap_cost\": 8", with: "\"buoyancy_cost\": 0, \"buoyancy_cap_cost\": 0")
            .replacingOccurrences(of: "\"buoy_init\": 0.5, \"buoy_sigma\": 0.02", with: "\"buoy_init\": 0.0, \"buoy_sigma\": 0.0")
        let s = try fork(config: config) { $0.levers.world3D.gravity = "0.02" }
        #expect(s.leverBaseline.world3D.buoyancyMax == "0")
        #expect(s.arguments(existingRuns: ["src3d"]) == Self.forkRun + ["--set", "physics.gravity=0.02"])
    }

    private static var actuatorOffConfig: String {
        tiny3DConfig
            .replacingOccurrences(of: "\"buoyancy_max\": 0.04", with: "\"buoyancy_max\": 0.0")
            .replacingOccurrences(of: "\"buoyancy_cost\": 16, \"buoyancy_cap_cost\": 8", with: "\"buoyancy_cost\": 0, \"buoyancy_cap_cost\": 0")
            .replacingOccurrences(of: "\"buoy_init\": 0.5, \"buoy_sigma\": 0.02", with: "\"buoy_init\": 0.0, \"buoy_sigma\": 0.0")
    }

    @Test func switchingTheActuatorBackOnInAForkOfAnActuatorlessRunIsRefused() throws {
        // The cap starts at buoy_init and drifts by buoy_sigma, both 0 and both CLI-only: nothing would ever lift.
        let s = try fork(config: Self.actuatorOffConfig) {
            $0.levers.world3D.buoyancyMax = "0.04"
            $0.levers.world3D.buoyancyCost = "16"
            $0.levers.world3D.buoyancyCapCost = "8"
        }
        let errors = s.validate(existingRuns: ["src3d"])
        #expect(errors.count == 1 && errors[0].contains("physics.buoyancy_max") && errors[0].contains("buoy_init"), "\(errors)")
        // Untouched, the same fork is fine: the refusal is about the switch, not about the source.
        let untouched = try fork(config: Self.actuatorOffConfig)
        #expect(untouched.validate(existingRuns: ["src3d"]).isEmpty)
        // So is a gravity edit, and so is an actuator that is still off.
        let gravity = try fork(config: Self.actuatorOffConfig) { $0.levers.world3D.gravity = "0.02" }
        #expect(gravity.validate(existingRuns: ["src3d"]).isEmpty)
    }

    @Test func aCapThatCanEvolveLetsTheActuatorSwitchOn() throws {
        // A source with the actuator off but a noisy cap (a preset or a command line can make one).
        let config = Self.actuatorOffConfig.replacingOccurrences(of: "\"buoy_sigma\": 0.0", with: "\"buoy_sigma\": 0.02")
        let s = try fork(config: config) { $0.levers.world3D.buoyancyMax = "0.04" }
        #expect(s.validate(existingRuns: ["src3d"]).isEmpty)
        #expect(s.arguments(existingRuns: ["src3d"]) == Self.forkRun + ["--set", "physics.buoyancy_max=0.04"])
    }

    @Test func aForkOfARunWithTheActuatorOnMaySwitchItOffAndOnAgain() throws {
        var s = try fork { $0.levers.world3D.buoyancyMax = "0" }
        #expect(!s.validate(existingRuns: ["src3d"]).isEmpty, "the costs must be 0 with no actuator (the engine's rule)")
        s.levers.world3D.buoyancyCost = "0"
        s.levers.world3D.buoyancyCapCost = "0"
        #expect(s.validate(existingRuns: ["src3d"]).isEmpty)
        s.levers.world3D.buoyancyMax = "0.04"  // back to the source's value: the baseline has an actuator
        #expect(s.world3DBaseline.buoyancyMax == "0.04")
        #expect(s.effectiveLevers.world3D.actuatorErrors(baseline: s.world3DBaseline).isEmpty)
    }

    @Test func aNewRunMayNotSwitchOnAnActuatorThePresetLacks() throws {
        var preset = try tiny3D()
        (preset.buoyancyMax, preset.buoyancyCost, preset.buoyancyCapCost) = ("0", "0", "0")
        (preset.buoyInit, preset.buoySigma) = ("0", "0")
        let s = newRun(world3D: preset) { $0.levers.world3D.buoyancyMax = "0.04" }
        #expect(s.validate(existingRuns: []).contains { $0.contains("physics.buoyancy_max cannot be switched on") })
        #expect(newRun(world3D: preset).validate(existingRuns: []).isEmpty)
    }

    @Test func aResumeHasNoLevers() throws {
        var s = try fork()
        s.mode = .resume
        s.levers.world3D.gravity = "junk"
        s.levers.rateMin = "junk"
        s.steps = "500"
        #expect(s.validate(existingRuns: ["src3d"]).isEmpty)
        #expect(s.arguments(existingRuns: ["src3d"]) == ["resume", "runs/src3d", "--steps", "500"])
        #expect(s.worldDims == 3, "a resume reports what leverBaseline holds; the app loads it for a fork only")
    }
}
