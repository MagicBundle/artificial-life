import Foundation

// MARK: - The 3D world's levers

/// The levers of a 3D world (`world.dims = 3`, docs/3D.md, DESIGN §27): its height, gravity, the buoyancy actuator and
/// the vertical food profile. Values are kept as text, like `Levers`, and only values that differ from a baseline are
/// passed to the engine, each as one `--set section.key=value`.
///
/// The baseline differs from the 2D levers'. A new run diffs these against the chosen *preset*
/// (`RunSettings.presetWorld3D`), not the engine defaults: `tiny3d` sets a gravity, a buoyancy actuator and a food
/// profile that the engine defaults (all 0) lack, so against the defaults an unchanged 3D preset would pass every key
/// and the GUI could never express "set back to 0". The 2D levers keep the engine defaults, so that a 2D run's command
/// line does not change. A fork diffs against its source's `config.json`, as the 2D levers do.
///
/// Three keys of the 3D world are CLI-only, and not levers:
/// - `physics.floor_eps` is the grounding tolerance. It is checked against the field cell (`radius + floor_eps <
///   field_cell`), so it is a numerical detail of the lattice, not something to explore.
/// - `mutation.buoy_init` and `mutation.buoy_sigma` are the start and the noise of the evolvable buoyancy cap, and
///   the engine requires both to be 0 when `physics.buoyancy_max` is 0. The baseline carries their values
///   (`buoyInit`, `buoySigma`; no field edits them) and `setArguments` passes both as 0 whenever the buoyancy lever
///   is 0, so that switching the actuator off is one edit, not an engine rejection the GUI cannot fix. The other
///   direction cannot be done from here: a baseline whose actuator is off has them at 0, and switching it on would
///   leave every cap at 0 (`actuatorErrors(baseline:)` says so).
package struct World3DLevers: Equatable, Sendable {
    /// `world.dims` of the preset or source run: 3 turns every rule below on, 2 makes the struct inert (nothing is
    /// validated or passed). It is never an input: the preset or the source fixes it, and a fork or resume cannot
    /// change it (`RunSettings.effectiveLevers`). The GUI offers no 2D to 3D conversion, because a 3D world needs a
    /// preset's vertical sizes, diffusion rates and food profile together (DESIGN §27.8).
    package var dims = 2
    /// `world.height`: the vertical size. A new run may change it; a fork may not (the lattice shape is fixed).
    package var height = "0"
    package var gravity = "0"
    package var buoyancyMax = "0"
    package var buoyancyCost = "0"
    package var buoyancyCapCost = "0"
    package var benthicLambda = "0"
    package var photicAmp = "0"
    package var photicLambda = "0"
    /// `mutation.buoy_init` and `mutation.buoy_sigma` of the baseline (CLI-only; see above).
    package var buoyInit = "0"
    package var buoySigma = "0"

    package init() {}

    /// The 3D values of a preset file; a key it does not set is the engine default, and a file without
    /// `world.dims = 3` is a 2D preset.
    package init(preset file: PresetFile) {
        dims = file.int("world", "dims") ?? 2
        if let v = file.double("world", "height") { height = Self.text(v) }
        if let v = file.double("physics", "gravity") { gravity = Self.text(v) }
        if let v = file.double("physics", "buoyancy_max") { buoyancyMax = Self.text(v) }
        if let n = file.int("energy", "buoyancy_cost") { buoyancyCost = String(n) }
        if let n = file.int("energy", "buoyancy_cap_cost") { buoyancyCapCost = String(n) }
        if let v = file.double("fields", "benthic_lambda") { benthicLambda = Self.text(v) }
        if let v = file.double("fields", "photic_amp") { photicAmp = Self.text(v) }
        if let v = file.double("fields", "photic_lambda") { photicLambda = Self.text(v) }
        if let v = file.double("mutation", "buoy_init") { buoyInit = Self.text(v) }
        if let v = file.double("mutation", "buoy_sigma") { buoySigma = Self.text(v) }
    }

    /// The 3D values of the named preset in the repository, or nil if its file cannot be read.
    package static func preset(_ name: String, repo: URL) -> World3DLevers? {
        PresetFile.preset(name, repo: repo).map(World3DLevers.init(preset:))
    }

    /// A number as lever text: integral values without a fraction, like `Levers(configJSON:)`.
    package static func text(_ v: Double) -> String {
        v == v.rounded() && abs(v) < 1e15 ? String(Int64(v)) : "\(v)"
    }

    /// Whether the actuator is off, as the engine will see it (`physics.buoyancy_max` = 0).
    private var buoyancyOff: Bool { Parse.decimal(buoyancyMax) == 0 }

    /// Human-readable problems; ranges mirror the `dims == 3` rules of `validate()` in src/alife/config.py that a
    /// field can break on its own. The engine checks the rest (the height against the world size, the field cell and
    /// the agent radius; gravity and buoyancy against the thrust) and a rejection (exit code 2) appears in the log.
    package func validate() -> [String] {
        guard dims == 3 else { return [] }
        var errors: [String] = []
        if let v = Parse.decimal(height) {
            if !(v > 0) { errors.append("world.height must be > 0.") }
        } else {
            errors.append("world.height must be a plain decimal number > 0.")
        }
        if Parse.decimal(gravity) == nil { errors.append("physics.gravity must be a plain decimal number >= 0.") }
        if Parse.decimal(buoyancyMax) == nil {
            errors.append("physics.buoyancy_max must be a plain decimal number >= 0.")
        }
        if let v = Parse.decimal(benthicLambda) {
            if !(v > 0) { errors.append("fields.benthic_lambda must be > 0.") }
        } else {
            errors.append("fields.benthic_lambda must be a plain decimal number > 0.")
        }
        if let a = Parse.decimal(photicAmp) {
            if a > 1 { errors.append("fields.photic_amp must be in [0, 1].") }
            if a > 0, let l = Parse.decimal(photicLambda), !(l > 0) {
                errors.append("fields.photic_lambda must be > 0 when fields.photic_amp > 0.")
            }
        } else {
            errors.append("fields.photic_amp must be a plain decimal number in [0, 1].")
        }
        if Parse.decimal(photicLambda) == nil {
            errors.append("fields.photic_lambda must be a plain decimal number (> 0 when photic_amp > 0).")
        }
        var costs: [Int] = []
        for (key, text) in [("energy.buoyancy_cost", buoyancyCost), ("energy.buoyancy_cap_cost", buoyancyCapCost)] {
            if let n = Parse.integer(text), n < 1 << 16 {
                costs.append(n)
            } else {
                errors.append("\(key) must be an integer in [0, 65536).")
            }
        }
        if buoyancyOff, costs.contains(where: { $0 != 0 }) {
            errors.append("energy.buoyancy_cost and energy.buoyancy_cap_cost must be 0 when physics.buoyancy_max is 0.")
        }
        return errors
    }

    /// Problems of switching the actuator on that no field of this struct can fix. The cap every agent carries is what
    /// `physics.buoyancy_max` scales, and it starts at `mutation.buoy_init` and drifts by `mutation.buoy_sigma` per
    /// birth; the engine requires both to be 0 while the actuator is off, so a baseline with the actuator off has them
    /// at 0, and nothing here sets them (they are CLI-only). Switching the actuator on from there would be accepted by
    /// the engine and do nothing: every cap stays 0, so there is never any lift and the costs are never paid. That is an
    /// error here, so that the run is not wasted; a baseline with the actuator off and a cap to evolve (its `buoy_init`
    /// or `buoy_sigma` above 0, which only a preset or a run made on the command line can have) can switch it on.
    package func actuatorErrors(baseline: World3DLevers) -> [String] {
        guard dims == 3, let wanted = Parse.decimal(buoyancyMax), wanted > 0, Parse.decimal(baseline.buoyancyMax) == 0,
              Parse.decimal(baseline.buoyInit) == 0, Parse.decimal(baseline.buoySigma) == 0 else { return [] }
        return ["physics.buoyancy_max cannot be switched on here: the source has mutation.buoy_init and "
                + "mutation.buoy_sigma at 0, so every agent's buoyancy cap would stay 0 and nothing would lift. "
                + "Set them with --set on the command line (alife run), or start from a preset with an actuator."]
    }

    /// `--set` arguments for the values that differ from `baseline` (compared as numbers, like `Levers`). Nothing
    /// unless the world is 3D. `world.dims` is never passed; `world.height` is passed when it differs, so a fork must
    /// set it to the baseline's first (`RunSettings.effectiveLevers`). Invalid fields are skipped. With the actuator
    /// off, the CLI-only cap keys the engine ties to it are passed as 0 when the baseline has them above 0.
    package func setArguments(baseline: World3DLevers) -> [String] {
        guard dims == 3 else { return [] }
        var args: [String] = []
        func decimal(_ key: String, _ now: String, _ base: String) {
            guard let v = Parse.decimal(now), v != Parse.decimal(base) else { return }
            args += ["--set", "\(key)=\(v)"]
        }
        func integer(_ key: String, _ now: String, _ base: String) {
            guard let v = Parse.integer(now), v != Parse.integer(base) else { return }
            args += ["--set", "\(key)=\(v)"]
        }
        decimal("world.height", height, baseline.height)
        decimal("physics.gravity", gravity, baseline.gravity)
        decimal("physics.buoyancy_max", buoyancyMax, baseline.buoyancyMax)
        integer("energy.buoyancy_cost", buoyancyCost, baseline.buoyancyCost)
        integer("energy.buoyancy_cap_cost", buoyancyCapCost, baseline.buoyancyCapCost)
        decimal("fields.benthic_lambda", benthicLambda, baseline.benthicLambda)
        decimal("fields.photic_amp", photicAmp, baseline.photicAmp)
        decimal("fields.photic_lambda", photicLambda, baseline.photicLambda)
        // Not editable: only the baseline's values count, and only when the actuator goes off.
        decimal("mutation.buoy_init", buoyancyOff ? "0" : baseline.buoyInit, baseline.buoyInit)
        decimal("mutation.buoy_sigma", buoyancyOff ? "0" : baseline.buoySigma, baseline.buoySigma)
        return args
    }
}
