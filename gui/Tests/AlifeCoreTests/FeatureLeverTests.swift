import Foundation
import Testing

@testable import AlifeCore

/// The levers of the feature layers (arms, structures, bonds and roles, comm): their tables, the gates that keep a closed feature's keys
/// at their defaults, the overlays a new run starts from, the pre-fill and the layout lock of a fork (DESIGN §45).
@Suite struct FeatureLeverTests {
    private func newRun(_ edit: (inout RunSettings) -> Void = { _ in }) -> RunSettings {
        var s = RunSettings()
        s.mode = .new
        s.name = "t"
        s.seed = "1"
        s.preset = .tiny
        edit(&s)
        return s
    }

    private func args(_ s: RunSettings) -> [String] { s.arguments(existingRuns: []) }

    private let base = ["run", "--preset", "tiny", "--seed", "1", "--out", "runs/t"]

    // MARK: The tables

    @Test func everyKeyIsUniqueAndItsDefaultIsValid() {
        let keys = FeatureLevers.allSpecs.map(\.key)
        #expect(Set(keys).count == keys.count && keys.count == 48)
        for spec in FeatureLevers.allSpecs {
            #expect(spec.problem(spec.defaultText) == nil, "\(spec.key) default \(spec.defaultText)")
            if case .flag(let gate)? = spec.gate { #expect(FeatureLevers.spec(gate) != nil, "\(spec.key) gate") }
            if case .nonzero(let gate)? = spec.gate { #expect(FeatureLevers.spec(gate) != nil, "\(spec.key) gate") }
        }
        #expect(FeatureLevers.groups.map(\.feature) == ["arms", "structures", "bonds", "comm", "niche"])
        for group in FeatureLevers.groups {
            #expect(FeatureLevers.spec(group.switchKey) != nil && group.specs.first?.key == group.switchKey)
            #expect(group.specs.allSatisfy { $0.section == group.specs[0].section || group.feature == "bonds" })
        }
    }

    @Test func theDefaultsPassNothingAndValidate() {
        let features = FeatureLevers()
        #expect(features.validate().isEmpty && features.setArguments(baseline: FeatureLevers()).isEmpty)
        #expect(!FeatureLevers.groups.contains { features.isOn($0.feature) })
        #expect(args(newRun()) == base, "a run with every feature off keeps its command line")
    }

    // MARK: Passing levers

    @Test func switchingAFeatureOnPassesItsSwitchAndOnlyTheChangedLevers() {
        let s = newRun {
            $0.levers.features["arms.enabled"] = "true"
            $0.levers.features["arms.kappa"] = "3"
            $0.levers.features["arms.init_atk"] = "0.5"
        }
        #expect(s.validate(existingRuns: []).isEmpty)
        #expect(args(s) == base + ["--set", "arms.enabled=true", "--set", "arms.kappa=3.0", "--set", "arms.init_atk=0.5"])
    }

    @Test func aLeverOfAFeatureThatIsOffIsNotPassed() {
        // the engine requires a closed feature's keys at their defaults: the GUI passes none of them
        let s = newRun {
            $0.levers.features["arms.kappa"] = "3"
            $0.levers.features["structures.cap"] = "999"
            $0.levers.features["roles.forage_gain"] = "0.5"
            $0.levers.features["bonds.share_kappa"] = "0.1"
            $0.levers.features["comm.width"] = "2"
        }
        #expect(s.validate(existingRuns: []).isEmpty)
        #expect(args(s) == base)
    }

    @Test func theBondsGateIsTheModeOrTheRoles() {
        var f = FeatureLevers()
        f["bonds.share_kappa"] = "0.1"
        f["roles.forage_gain"] = "0.5"
        #expect(f.setArguments(baseline: FeatureLevers()).isEmpty)
        f["bonds.share_mode"] = "2"
        #expect(f.setArguments(baseline: FeatureLevers()) == ["--set", "bonds.share_mode=2", "--set", "bonds.share_kappa=0.1"])
        f["roles.enabled"] = "true"
        #expect(f.setArguments(baseline: FeatureLevers()) == [
            "--set", "bonds.share_mode=2", "--set", "bonds.share_kappa=0.1", "--set", "roles.enabled=true",
            "--set", "roles.forage_gain=0.5",
        ])
        #expect(f.isOn("bonds"))
        f["bonds.share_mode"] = "0"
        #expect(f.isOn("bonds"), "the roles alone keep the bond block on")
    }

    @Test func valuesAreComparedAsNumbers() {
        var f = FeatureLevers()
        f["comm.enabled"] = "true"
        f["comm.cost"] = "4.0"
        f["comm.width"] = "04"
        f["comm.atten_pow"] = "1e0"
        #expect(f.setArguments(baseline: FeatureLevers()) == ["--set", "comm.enabled=true"])
        f["comm.cost"] = "16"
        f["comm.width"] = "2"
        #expect(f.setArguments(baseline: FeatureLevers()) == [
            "--set", "comm.enabled=true", "--set", "comm.width=2", "--set", "comm.cost=16.0",
        ])
    }

    @Test func theArgumentsKeepTheTableOrder() {
        let s = newRun {
            $0.levers.rateMin = "0.01"  // the dynamics levers come first
            $0.levers.features["comm.enabled"] = "true"
            $0.levers.features["structures.enabled"] = "true"
            $0.levers.features["arms.enabled"] = "true"
        }
        #expect(args(s) == base + [
            "--set", "mutation.rate_min=0.01", "--set", "arms.enabled=true", "--set", "structures.enabled=true",
            "--set", "comm.enabled=true",
        ])
    }

    // MARK: Validation

    @Test func aBadValueIsNamedAndSkipped() {
        var f = FeatureLevers()
        f["structures.enabled"] = "true"
        f["structures.cap"] = "100"
        f["structures.decay_q12"] = "0"
        f["structures.shelter_gain"] = "1.5"
        f["structures.barrier_gain"] = "abc"
        f["structures.open_frac"] = "0"
        let errors = f.validate()
        #expect(errors.count == 5)
        for key in ["cap", "decay_q12", "shelter_gain", "barrier_gain", "open_frac"] {
            #expect(errors.contains { $0.hasPrefix("structures.\(key) must") }, Comment(rawValue: key))
        }
        let passed = f.setArguments(baseline: FeatureLevers())
        #expect(passed.contains("structures.enabled=true") && !passed.contains { $0.hasPrefix("structures.barrier_gain") },
                "a value that does not parse is skipped; the rest is the validation's")
    }

    @Test func theRangesAreTheEnginesAtTheirEdges() {
        func problem(_ key: String, _ text: String) -> String? { FeatureLevers.spec(key)!.problem(text) }
        #expect(problem("arms.key_floor", "1") == nil && problem("arms.key_floor", "1.01") != nil)
        #expect(problem("arms.speed_slow", "0.9") == nil && problem("arms.speed_slow", "0.91") != nil)
        #expect(problem("arms.a0", "0") != nil && problem("arms.a0", "0.001") == nil)
        #expect(problem("arms.sigma", "0.5") == nil && problem("arms.sigma_phase", "0.51") != nil)
        #expect(problem("structures.cap", "255") != nil && problem("structures.cap", "256") == nil)
        #expect(problem("structures.cap", "32767") == nil && problem("structures.cap", "32768") != nil)
        #expect(problem("structures.barrier_gain", "3") == nil && problem("structures.barrier_gain", "3.1") != nil)
        #expect(problem("structures.open_frac", "1") == nil && problem("structures.open_frac", "0") != nil)
        #expect(problem("structures.decay_q12", "4096") == nil && problem("structures.decay_q12", "4097") != nil)
        #expect(problem("bonds.share_kappa", "0.125") == nil && problem("bonds.share_kappa", "0.13") != nil)
        #expect(problem("bonds.share_kappa", "0") != nil)
        #expect(problem("bonds.share_mode", "3") != nil && problem("bonds.share_mode", "1") == nil)
        #expect(problem("roles.forage_defend_pen", "1") != nil && problem("roles.forage_defend_pen", "0.99") == nil)
        #expect(problem("roles.guard_shield", "0.9") == nil && problem("roles.guard_shield", "0.91") != nil)
        #expect(problem("roles.ramp_steps", "0") != nil && problem("roles.ramp_steps", "32767") == nil)
        #expect(problem("roles.threshold", "0") != nil && problem("roles.threshold", "1") != nil && problem("roles.threshold", "0.5") == nil)
        #expect(problem("comm.width", "0") != nil && problem("comm.width", "8") == nil && problem("comm.width", "9") != nil)
        #expect(problem("comm.atten_pow", "0") != nil && problem("comm.kin_exp", "0") == nil)
        #expect(problem("comm.enabled", "yes") != nil && problem("comm.enabled", "true") == nil)
    }

    @Test func theRulesAcrossKeysAreChecked() {
        var arms = FeatureLevers()
        arms["arms.enabled"] = "true"
        arms["arms.kappa"] = "9"
        #expect(arms.validate() == [], "kappa / (kappa + a0) = 0.9 is the edge")
        arms["arms.kappa"] = "9.5"
        #expect(arms.validate().contains { $0.contains("0.9") })
        arms["arms.kappa"] = "2"
        arms["arms.key_floor"] = "0"
        #expect(arms.validate().contains { $0.contains("protect_floor") })

        var structures = FeatureLevers()
        structures["structures.enabled"] = "true"
        structures["structures.build_max"] = "1024"
        structures["structures.build_cost"] = "1024"
        #expect(structures.validate().contains { $0.contains("1048576") })
        structures["structures.build_cost"] = "1023"
        #expect(structures.validate().isEmpty)
        structures["structures.shield_gain"] = "1"
        #expect(structures.validate().contains { $0.contains("protect_floor") })

        var niche = FeatureLevers()
        niche["niche.farm"] = "true"
        niche["niche.farm_cost"] = "32768"
        niche["niche.farm_rate"] = "64"
        #expect(niche.validate().contains { $0.contains("1048576") })
        niche["niche.farm_cost"] = "150"
        niche["niche.farm_cap"] = "32767"
        niche["niche.farm_decay_q16"] = "40000"
        #expect(niche.validate().contains { $0.contains("2^30") })
        niche["niche.farm_decay_q16"] = "16"
        #expect(niche.validate().isEmpty)

        var comm = FeatureLevers()
        comm["comm.enabled"] = "true"
        comm["comm.cost"] = "8192"
        comm["comm.width"] = "4"
        #expect(comm.validate().contains { $0.contains("32768") })
    }

    @Test func aClosedFeaturesBadValueIsNotAnError() {
        var f = FeatureLevers()
        f["arms.kappa"] = "abc"
        #expect(f.validate().isEmpty && f.setArguments(baseline: FeatureLevers()).isEmpty)
    }

    // MARK: Overlays

    private func fromOverlays(_ names: Set<String>, dims: Int = 2) -> FeatureLevers {
        FeatureLevers.baseline(overlays: names, dims: dims, repo: Repo.root)
    }

    @Test func theOverlayFilesAreReadIntoTheLevers() {
        let build = fromOverlays(["structures"])
        #expect(build.values.isOn("structures.enabled") && build["structures.cap"] == "4096" && !build.values.isOn("roles.enabled"))
        let bonds = fromOverlays(["bonds"])
        #expect(bonds["bonds.share_mode"] == "2" && bonds["bonds.share_kin_floor"] == "0.5" && bonds.values.isOn("roles.enabled"))
        #expect(bonds["roles.forage_gain"] == "0.75" && bonds["roles.synergy"] == "0.25" && bonds["roles.switch_cost"] == "100")
        let comm = fromOverlays(["comm"])
        #expect(comm.values.isOn("comm.enabled") && comm["comm.width"] == "4" && comm["comm.cost"] == "4")
        for f in [build, bonds, comm] { #expect(f.validate().isEmpty) }
        let niche = fromOverlays(["niche"])
        #expect(niche.values.isOn("niche.farm") && niche["niche.farm_cost"] == "30" && niche["niche.farm_rate"] == "8" && niche.validate().isEmpty)
        let niche3D = fromOverlays(["niche"], dims: 3)
        #expect(niche3D.values.isOn("niche.farm") && niche3D["niche.farm_cost"] == "150" && niche3D.validate().isEmpty,
                "a 3D world takes the overlay niche3d, whose cost fits the larger regrowth")
        #expect(fromOverlays(["arms"]) == FeatureLevers(), "the arms layer has no overlay file")
        #expect(fromOverlays([]) == FeatureLevers())
    }

    @Test func aNewRunWithAnOverlayPassesTheOverlayAndOnlyTheDifferenceFromIt() {
        func run(_ names: Set<String>, _ edit: (inout RunSettings) -> Void = { _ in }) -> RunSettings {
            newRun {
                $0.selectFeatureOverlays(names, baseline: fromOverlays(names))
                edit(&$0)
            }
        }
        #expect(args(run(["bonds"])) == ["run", "--preset", "tiny", "--overlay", "bonds", "--seed", "1", "--out", "runs/t"])
        #expect(run(["bonds"]).validate(existingRuns: []).isEmpty)
        let two = run(["structures", "comm"])
        #expect(args(two) == ["run", "--preset", "tiny", "--overlay", "build", "--overlay", "comm", "--seed", "1", "--out", "runs/t"])
        let edited = run(["bonds"]) {
            $0.levers.features["roles.guard_shield"] = "0.6"
            $0.levers.features["bonds.share_mode"] = "1"
        }
        #expect(args(edited) == ["run", "--preset", "tiny", "--overlay", "bonds", "--seed", "1", "--out", "runs/t",
                                 "--set", "bonds.share_mode=1", "--set", "roles.guard_shield=0.6"])
        // an overlay's switches turned off take every key of the feature back to its default, whatever the overlay set
        let off = run(["bonds"]) {
            $0.levers.features["bonds.share_mode"] = "0"
            $0.levers.features["roles.enabled"] = "false"
        }
        #expect(off.validate(existingRuns: []).isEmpty)
        #expect(args(off) == ["run", "--preset", "tiny", "--overlay", "bonds", "--seed", "1", "--out", "runs/t",
                              "--set", "bonds.share_mode=0", "--set", "bonds.share_kin_floor=1.0", "--set", "roles.enabled=false",
                              "--set", "roles.forage_gain=0.0", "--set", "roles.forage_defend_pen=0.0",
                              "--set", "roles.guard_harvest_pen=0.0", "--set", "roles.guard_shield=0.0",
                              "--set", "roles.synergy=0.0", "--set", "roles.switch_cost=0"])
        // the overlays are the new run's: a fork passes none
        let fork = run(["bonds"]) { $0.mode = .fork; $0.sourceRun = "src" }
        #expect(fork.overlayArguments.isEmpty)
    }

    @Test func aThreeDWorldPassesTheThreeDFarmingOverlay() {
        var s = newRun { $0.preset = .tiny3d }
        s.selectPreset(.tiny3d, world3D: World3DLevers(preset: PresetFile(text: "[world]\ndims = 3\nheight = 24\n")))
        s.selectFeatureOverlays(["niche", "comm"], baseline: fromOverlays(["niche", "comm"], dims: 3))
        #expect(s.worldDims == 3 && s.overlayArguments == ["--overlay", "comm", "--overlay", "niche3d"])
        #expect(args(s) == ["run", "--preset", "tiny3d", "--overlay", "comm", "--overlay", "niche3d", "--seed", "1", "--out", "runs/t"])
        let flat = newRun { $0.selectFeatureOverlays(["niche"], baseline: fromOverlays(["niche"])) }
        #expect(flat.overlayArguments == ["--overlay", "niche"])
    }

    @Test func choosingOverlaysResetsTheFeatureLevers() {
        var s = newRun { $0.levers.features["arms.enabled"] = "true" }
        s.selectFeatureOverlays(["structures"], baseline: fromOverlays(["structures"]))
        #expect(s.levers.features == fromOverlays(["structures"]) && s.featureLeverBaseline == fromOverlays(["structures"]))
        s.selectFeatureOverlays([], baseline: FeatureLevers())
        #expect(s.levers.features == FeatureLevers() && s.overlayArguments.isEmpty)
    }

    // MARK: A fork

    private let sourceConfig = """
    {"mutation": {"rate_min": 0.001},
     "arms": {"enabled": true, "kappa": 3.0, "init_atk": 0.25},
     "structures": {"enabled": true, "cap": 2048, "decay_q12": 32, "shelter_gain": 0.4},
     "bonds": {"share_mode": 1, "share_kappa": 0.1},
     "roles": {"enabled": true, "forage_gain": 0.5, "inherit_role": true, "ramp_steps": 32},
     "comm": {"enabled": true, "width": 2, "cost": 8.0, "control": "deaf"},
     "niche": {"farm": true, "farm_rate": 4, "farm_cost": 60, "farm_bonus_frac": 0.25, "control": "mute"}}
    """

    private func forkSettings(_ edit: (inout RunSettings) -> Void = { _ in }) throws -> RunSettings {
        let levers = try Levers(configJSON: Data(sourceConfig.utf8))
        var s = RunSettings()
        s.mode = .fork
        s.name = "child"
        s.sourceRun = "src"
        s.seed = ""
        s.levers = levers
        s.leverBaseline = levers
        edit(&s)
        return s
    }

    @Test func aConfigPrefillsEveryFeatureLever() throws {
        let f = try Levers(configJSON: Data(sourceConfig.utf8)).features
        #expect(f.values.isOn("arms.enabled") && f["arms.kappa"] == "3" && f["arms.init_atk"] == "0.25" && f["arms.a0"] == "1")
        #expect(f["structures.cap"] == "2048" && f["structures.decay_q12"] == "32" && f["structures.shelter_gain"] == "0.4")
        #expect(f["bonds.share_mode"] == "1" && f["bonds.share_kappa"] == "0.1" && f["roles.ramp_steps"] == "32")
        #expect(f.values.isOn("roles.inherit_role") && f["roles.forage_gain"] == "0.5")
        #expect(f["comm.width"] == "2" && f["comm.cost"] == "8")
        #expect(f.values.isOn("niche.farm") && f["niche.farm_rate"] == "4" && f["niche.farm_cost"] == "60" && f["niche.farm_cap"] == "4095")
        #expect(f["niche.farm_bonus_frac"] == "0.25")
        #expect(f.validate().isEmpty)
    }

    @Test func aRunWithoutTheKeysIsAllDefaults() throws {
        let plain = try Levers(configJSON: Data(#"{"mutation": {"rate_min": 0.001}, "run": {"metrics_every": 100}}"#.utf8))
        #expect(plain.features == FeatureLevers())
    }

    @Test func aFeatureKeyOfTheWrongTypeIsRefused() {
        for json in [#"{"arms": {"enabled": 1}}"#, #"{"arms": {"kappa": true}}"#, #"{"comm": {"width": 2.5}}"#,
                     #"{"roles": {"enabled": "yes"}}"#, #"{"structures": {"cap": "big"}}"#] {
            #expect(throws: LeverConfigError.self, Comment(rawValue: json)) { try Levers(configJSON: Data(json.utf8)) }
        }
    }

    @Test func anUnchangedForkPassesNothingAndAChangedLeverPassesItsDifference() throws {
        #expect(try forkSettings().validate(existingRuns: ["src"]).isEmpty)
        #expect(args(try forkSettings()) == ["run", "--init-from", "runs/src", "--out", "runs/child"])
        let tuned = try forkSettings {
            $0.levers.features["arms.kappa"] = "2"
            $0.levers.features["comm.cost"] = "16"
            $0.levers.features["roles.forage_gain"] = "0.75"
            $0.levers.features["niche.farm_cost"] = "90"
        }
        #expect(tuned.validate(existingRuns: ["src"]).isEmpty)
        #expect(args(tuned) == ["run", "--init-from", "runs/src", "--out", "runs/child",
                                "--set", "arms.kappa=2.0", "--set", "roles.forage_gain=0.75", "--set", "comm.cost=16.0",
                                "--set", "niche.farm_cost=90"])
        // 1 and 2 are both the bond block: the mode can move between them
        let obligate = try forkSettings { $0.levers.features["bonds.share_mode"] = "2" }
        #expect(obligate.validate(existingRuns: ["src"]).isEmpty && args(obligate).suffix(2) == ["--set", "bonds.share_mode=2"])
    }

    @Test func aForkCannotChangeTheLayout() throws {
        for (key, value) in [("arms.enabled", "false"), ("structures.enabled", "false"), ("roles.enabled", "false"),
                             ("comm.enabled", "false"), ("comm.width", "3"), ("niche.farm", "false")] {
            let s = try forkSettings { $0.levers.features[key] = value }
            let errors = s.validate(existingRuns: ["src"])
            #expect(errors.contains { $0.contains("fixes the brain layout") || $0.contains("brain layout") }, Comment(rawValue: key))
        }
        // the bond block off by both its switches
        let off = try forkSettings {
            $0.levers.features["bonds.share_mode"] = "0"
            $0.levers.features["roles.enabled"] = "false"
        }
        #expect(!off.validate(existingRuns: ["src"]).isEmpty)
        // turning the roles off while the sharing mode keeps the block is a layout the same
        let rolesOnlyOff = try forkSettings { $0.levers.features["roles.enabled"] = "false" }
        #expect(rolesOnlyOff.validate(existingRuns: ["src"]).contains { $0.contains("roles.enabled") })
        // a feature the source lacks cannot be turned on either
        var plain = RunSettings()
        plain.mode = .fork
        plain.name = "c"
        plain.sourceRun = "src"
        plain.levers.features["arms.enabled"] = "true"
        #expect(plain.validate(existingRuns: ["src"]).contains { $0.contains("arms.enabled") })
    }

    @Test func aResumeHasNoLevers() {
        var s = RunSettings()
        s.mode = .resume
        s.sourceRun = "src"
        s.levers.features["arms.enabled"] = "true"
        #expect(s.arguments(existingRuns: ["src"]) == ["resume", "runs/src"])
        #expect(s.validate(existingRuns: ["src"]).isEmpty)
    }

    @Test func resettingTheLeversGoesToTheBaseline() {
        var l = Levers()
        l.features["arms.enabled"] = "true"
        l.reset(world3D: World3DLevers(), features: fromOverlays(["comm"]))
        #expect(l.features == fromOverlays(["comm"]))
        l.reset()
        #expect(l.features == FeatureLevers())
    }
}
