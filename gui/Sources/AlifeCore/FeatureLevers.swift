import Foundation

// MARK: - The feature layers' levers

// The five feature layers of the engine (arms, structures, bonds and roles, comm, niche/farming; DESIGN §43 and §45) each add a
// config section whose keys are levers of a new run and of a fork. Their tables live in `<Feature>Levers.swift` (one file per feature,
// the file `tests/test_gui_sync.py` compares with `config.FEATURE_KEYS`); this file is the shared kit: one table row per key
// (`FeatureLeverSpec`), the values (`FeatureValues`), and the rules that are the same for every feature.
//
// Rules the engine states and this mirrors (`ext/*.py` `_validate`):
// - A key of a feature that is off is inert and must keep its default (the config would not hash otherwise), so the GUI passes
//   nothing for a gated key while its gate is closed, and passes the default when a baseline had it open and the gate is closed now.
// - A few keys are start-of-run (they fix the brain layout or the state shapes): a checkpoint never crosses them, so a fork cannot
//   change them (`alife migrate` does); `layoutErrors(baseline:)` says so, and the UI disables them in a fork.

/// One lever of a feature: its config key, default, range, gate and help text.
package struct FeatureLeverSpec: Equatable, Sendable, Identifiable {
    package struct Option: Equatable, Sendable {
        package var value: String
        package var label: String
    }

    package enum Kind: Equatable, Sendable {
        case flag
        /// A plain non-negative decimal inside `[low, high]`; an open end excludes it.
        case decimal(low: Double, high: Double, lowOpen: Bool, highOpen: Bool)
        case integer(low: Int, high: Int)
        case options([Option])
    }

    /// When the lever is live: the key of a flag that must be on, or of a number that must not be 0.
    package enum Gate: Equatable, Sendable {
        case flag(String)
        case nonzero(String)
    }

    /// `section.key`, as `--set` takes it.
    package let key: String
    package let defaultText: String
    package let title: String
    package let help: String
    package let kind: Kind
    package let gate: Gate?
    /// A start-of-run key: a fork cannot change it.
    package let layout: Bool

    package var id: String { key }
    package var section: String { String(key.prefix { $0 != "." }) }
    package var name: String { String(key.drop { $0 != "." }.dropFirst()) }

    package init(key: String, defaultText: String, title: String, help: String, kind: Kind, gate: Gate?, layout: Bool) {
        self.key = key
        self.defaultText = defaultText
        self.title = title
        self.help = help
        self.kind = kind
        self.gate = gate
        self.layout = layout
    }

    // The constructors are named for their kind and take the key first: `tests/test_gui_sync.py` finds the keys of a feature file by
    // these calls (`decimal("arms.kappa", ...)`).

    package static func flag(_ key: String, _ defaultText: String, _ title: String, _ help: String, gate: Gate? = nil,
                             layout: Bool = false) -> Self {
        Self(key: key, defaultText: defaultText, title: title, help: help, kind: .flag, gate: gate, layout: layout)
    }

    package static func decimal(_ key: String, _ defaultText: String, _ title: String, _ help: String,
                                low: Double = 0, high: Double = .infinity, lowOpen: Bool = false, highOpen: Bool = false,
                                gate: Gate? = nil) -> Self {
        Self(key: key, defaultText: defaultText, title: title, help: help,
             kind: .decimal(low: low, high: high, lowOpen: lowOpen, highOpen: highOpen), gate: gate, layout: false)
    }

    package static func integer(_ key: String, _ defaultText: String, _ title: String, _ help: String, low: Int = 0,
                                high: Int = 999_999_999, gate: Gate? = nil, layout: Bool = false) -> Self {
        Self(key: key, defaultText: defaultText, title: title, help: help, kind: .integer(low: low, high: high), gate: gate,
             layout: layout)
    }

    package static func options(_ key: String, _ defaultText: String, _ title: String, _ help: String, _ options: [Option],
                                gate: Gate? = nil, layout: Bool = false) -> Self {
        Self(key: key, defaultText: defaultText, title: title, help: help, kind: .options(options), gate: gate, layout: layout)
    }

    // MARK: Checking and passing a value

    /// What is wrong with `text` as this lever's value, nil when it is fine.
    package func problem(_ text: String) -> String? {
        switch kind {
        case .flag:
            return Self.flagValue(text) == nil ? "\(key) must be on or off." : nil
        case .decimal(let low, let high, let lowOpen, let highOpen):
            let range = "\(lowOpen ? "(" : "[")\(Self.number(low)), \(high.isFinite ? Self.number(high) : "inf")\(highOpen ? ")" : "]")"
            guard let v = Parse.decimal(text) else {
                return "\(key) must be a plain decimal number in \(range)."
            }
            let aboveLow = lowOpen ? v > low : v >= low
            let belowHigh = highOpen ? v < high : v <= high
            return aboveLow && belowHigh ? nil : "\(key) must be in \(range)."
        case .integer(let low, let high):
            guard let n = Parse.integer(text), n >= low, n <= high else {
                return "\(key) must be an integer in [\(low), \(high)]."
            }
            return nil
        case .options(let options):
            return options.contains { $0.value == text } ? nil : "\(key) must be one of \(options.map(\.value).joined(separator: ", "))."
        }
    }

    /// The `--set` value of `text` (nil for an invalid one): `true`/`false`, a decimal as the engine's float, an integer.
    package func argument(_ text: String) -> String? {
        switch kind {
        case .flag: return Self.flagValue(text).map { $0 ? "true" : "false" }
        case .decimal: return Parse.decimal(text).map { "\($0)" }
        case .integer: return Parse.integer(text).map { "\($0)" }
        case .options: return problem(text) == nil ? text : nil
        }
    }

    /// Whether two texts are the same value (numbers as numbers, so 5e-05 equals 0.00005 and 1 equals 1.0).
    package func same(_ a: String, _ b: String) -> Bool {
        switch kind {
        case .flag: return Self.flagValue(a) == Self.flagValue(b)
        case .decimal: return Parse.decimal(a) == Parse.decimal(b)
        case .integer: return Parse.integer(a) == Parse.integer(b)
        case .options: return a == b
        }
    }

    package static func flagValue(_ text: String) -> Bool? {
        switch text.trimmingCharacters(in: .whitespaces).lowercased() {
        case "true": return true
        case "false": return false
        default: return nil
        }
    }

    private static func number(_ v: Double) -> String { v == v.rounded() && abs(v) < 1e15 ? String(Int64(v)) : "\(v)" }
}

/// The texts of every feature lever, by `section.key`; a key not set holds its default.
package struct FeatureValues: Equatable, Sendable {
    private var texts: [String: String] = [:]

    package init() {}

    package subscript(key: String) -> String {
        get { texts[key] ?? FeatureLevers.spec(key)?.defaultText ?? "" }
        set {
            if let spec = FeatureLevers.spec(key), spec.defaultText == newValue {
                texts[key] = nil
            } else {
                texts[key] = newValue
            }
        }
    }

    /// The flag at `key`: on only for the text `true`.
    package func isOn(_ key: String) -> Bool { FeatureLeverSpec.flagValue(self[key]) == true }
}

/// A feature's group of levers: its title, its table and the rules across keys.
package struct FeatureLeverGroup: Sendable {
    /// The feature's name in `ext.EXTENSIONS` (`arms`, `structures`, `bonds`, `comm`, `niche`).
    package let feature: String
    package let title: String
    package let summary: String
    /// The key of the lever that switches the feature on (`arms.enabled`, `niche.farm`; for bonds, `bonds.share_mode`).
    package let switchKey: String
    package let specs: [FeatureLeverSpec]
    /// Problems that involve several keys; called only while the feature is on.
    package let crossErrors: @Sendable (FeatureValues) -> [String]
    /// The overlay file (`presets/NAME.toml`) that sets the feature's recommended values, if the engine has one.
    package let overlay: String?
    /// The overlay to use instead in a 3D world (farming's cost scales with the regrowth), if it differs.
    package let overlay3D: String?
    /// The keys whose change moves the brain layout or the state shapes (`FeatureLevers.layoutErrors`).
    package var layoutKeys: [String] { specs.filter(\.layout).map(\.key) }

    package init(feature: String, title: String, summary: String, switchKey: String, specs: [FeatureLeverSpec],
                 crossErrors: @escaping @Sendable (FeatureValues) -> [String], overlay: String?, overlay3D: String? = nil) {
        self.feature = feature
        self.title = title
        self.summary = summary
        self.switchKey = switchKey
        self.specs = specs
        self.crossErrors = crossErrors
        self.overlay = overlay
        self.overlay3D = overlay3D
    }

    /// The overlay file for a world of `dims` dimensions; nil when the feature has none.
    package func overlayName(dims: Int) -> String? { dims == 3 ? (overlay3D ?? overlay) : overlay }
}

/// What a fork or a new run passes for the five feature layers, and the rules that keep the engine from rejecting it.
package struct FeatureLevers: Equatable, Sendable {
    package var values = FeatureValues()

    package init() {}

    /// Every group, in `ext.EXTENSIONS` order.
    package static let groups: [FeatureLeverGroup] = [
        ArmsLevers.group, StructuresLevers.group, BondsLevers.group, CommLevers.group, NicheLevers.group,
    ]

    package static let allSpecs: [FeatureLeverSpec] = groups.flatMap(\.specs)

    private static let specsByKey: [String: FeatureLeverSpec] = Dictionary(uniqueKeysWithValues: allSpecs.map { ($0.key, $0) })

    package static func spec(_ key: String) -> FeatureLeverSpec? { specsByKey[key] }

    package static func group(_ feature: String) -> FeatureLeverGroup? { groups.first { $0.feature == feature } }

    package subscript(key: String) -> String {
        get { values[key] }
        set { values[key] = newValue }
    }

    // MARK: Gates

    /// Whether the lever's gate is open (a lever with no gate always is).
    package func isLive(_ spec: FeatureLeverSpec) -> Bool {
        switch spec.gate {
        case nil: return true
        case .flag(let key)?: return values.isOn(key)
        case .nonzero(let key)?: return Parse.integer(values[key]).map { $0 != 0 } ?? Parse.decimal(values[key]).map { $0 != 0 } ?? false
        }
    }

    /// Whether the feature is on: its switch is on (a flag), or not 0 (the bonds' mode), or the roles are on.
    package func isOn(_ feature: String) -> Bool {
        guard let group = Self.group(feature) else { return false }
        if feature == "bonds" { return values.isOn("roles.enabled") || (Parse.integer(values[group.switchKey]) ?? 0) != 0 }
        return values.isOn(group.switchKey)
    }

    /// The text that is passed or checked for `spec`: its own while the gate is open, else the default (the engine ties a closed
    /// feature's keys to their defaults).
    package func effective(_ spec: FeatureLeverSpec) -> String { isLive(spec) ? values[spec.key] : spec.defaultText }

    // MARK: Validation

    /// Human-readable problems; ranges mirror the `_validate` of `src/alife/ext/*.py`. A key whose gate is closed is not checked.
    package func validate() -> [String] {
        var errors: [String] = []
        for spec in Self.allSpecs where isLive(spec) {
            if let problem = spec.problem(values[spec.key]) { errors.append(problem) }
        }
        for group in Self.groups where isOn(group.feature) && errors.isEmpty {
            errors += group.crossErrors(values)
        }
        return errors
    }

    /// A fork's start-of-run keys that differ from its source: the layout (brain tail, state shapes) cannot change in a fork, and a
    /// checkpoint never crosses it (`alife migrate` is the way to a new layout). Empty for a layout that is the same.
    package func layoutErrors(baseline: FeatureLevers) -> [String] {
        var errors: [String] = []
        for group in Self.groups {
            for spec in group.specs where spec.layout {
                if !spec.same(effective(spec), baseline.effective(spec)) {
                    errors.append("\(spec.key) fixes the brain layout and cannot change in a fork; migrate the source with `alife migrate`.")
                }
            }
        }
        if values.isOn("roles.enabled") != baseline.values.isOn("roles.enabled")
            || isOn("bonds") != baseline.isOn("bonds") {
            errors.append("bonds.share_mode and roles.enabled fix the brain layout; a fork cannot turn the bond block on or off.")
        }
        return errors
    }

    // MARK: Arguments

    /// `--set` arguments for the levers whose effective value differs from `baseline`'s (the engine defaults, plus the chosen overlays,
    /// for a new run; the source's values for a fork). Invalid fields are skipped. In the order of the tables.
    package func setArguments(baseline: FeatureLevers) -> [String] {
        var args: [String] = []
        for spec in Self.allSpecs {
            let now = effective(spec)
            let was = baseline.effective(spec)
            guard !spec.same(now, was), let value = spec.argument(now) else { continue }
            args += ["--set", "\(spec.key)=\(value)"]
        }
        return args
    }

    /// The values of an overlay file over the defaults: what a new run with that overlay starts from. A key the file does not set
    /// keeps the value `self` has.
    package mutating func apply(overlay file: PresetFile) {
        for spec in Self.allSpecs {
            let (table, key) = (spec.section, spec.name)
            switch spec.kind {
            case .flag:
                if let b = file.bool(table, key) { values[spec.key] = b ? "true" : "false" }
            case .decimal:
                if let v = file.double(table, key) { values[spec.key] = World3DLevers.text(v) }
            case .integer:
                if let n = file.int(table, key) { values[spec.key] = String(n) }
            case .options:
                if let n = file.int(table, key) { values[spec.key] = String(n) }
            }
        }
    }

    /// The feature levers a new run starts from with `features` (names from `ext.EXTENSIONS`) as overlays in a world of `dims`
    /// dimensions: the defaults with each overlay file's values over them. A file that cannot be read adds nothing (the engine will
    /// say so when it is asked for the overlay).
    package static func baseline(overlays features: Set<String>, dims: Int, repo: URL) -> FeatureLevers {
        var result = FeatureLevers()
        for group in groups where features.contains(group.feature) {
            guard let name = group.overlayName(dims: dims), let file = PresetFile.preset(name, repo: repo) else { continue }
            result.apply(overlay: file)
        }
        return result
    }

    // MARK: Reading a config

    /// Sets `key` from a `config.json` value: a boolean, a number or an integer as the lever's kind needs; false when the value is
    /// of the wrong type (the engine would reject it too).
    package mutating func set(_ key: String, fromJSON raw: Any) -> Bool {
        guard let spec = Self.spec(key) else { return false }
        switch spec.kind {
        case .flag:
            guard let n = raw as? NSNumber, CFGetTypeID(n) == CFBooleanGetTypeID() else { return false }
            values[key] = n.boolValue ? "true" : "false"
        case .decimal:
            guard let n = raw as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else { return false }
            values[key] = World3DLevers.text(n.doubleValue)
        case .integer, .options:
            guard let n = raw as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(), let v = Int(exactly: n.doubleValue) else {
                return false
            }
            values[key] = String(v)
        }
        return true
    }
}
