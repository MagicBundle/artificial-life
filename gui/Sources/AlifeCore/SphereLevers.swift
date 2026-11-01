import Foundation

// MARK: - The sphere world's levers

/// The levers of a sphere world (`world.geometry = "sphere"`, DESIGN §46.9; the engine's `config.SPHERE_KEYS`): the planet's radius, the
/// field's Nside, the number of equal-volume layers and the soft lid band. Values are kept as text, like `World3DLevers`, which holds one
/// of these (a planet is a 3D world with a shape): the baseline is the chosen preset's or the source run's, and only values that differ
/// from it are passed to the engine, each as one `--set section.key=value`.
///
/// **`world.geometry` is never an input.** The preset or the source fixes it, and no fork or resume can change it (the engine refuses a
/// cross-geometry fork and resume, DESIGN §46.9), so the GUI shows it and does not offer it. The radius, the Nside and the layers are the
/// shape of the state: a new run may change them, a fork may not (`RunSettings.effectiveLevers`, as for `world.height`). The ceiling
/// band is a dynamics key: it can be changed in a fork.
///
/// A place in this world is a longitude and a latitude in degrees (`x` and `y` of a power); the radius is an arc length at the floor.
package struct SphereLevers: Equatable, Sendable {
    /// `world.geometry` of the preset or the source: `"box"` makes the struct inert (nothing is validated or passed), `"sphere"` turns
    /// every rule below on.
    package var geometry = "box"
    /// `world.radius`: the planet's radius `Rp`. 0 in a box world.
    package var radius = "0"
    /// `world.nside`: the field's HEALPix Nside.
    package var nside = "0"
    /// `world.layers`: the number of equal-volume radial layers.
    package var layers = "0"
    /// `physics.ceiling_band`: the soft band under the lid, 0 = off.
    package var ceilingBand = "0"

    package init() {}

    package var isSphere: Bool { geometry == "sphere" }

    /// The sphere values of a preset file; a preset without `world.geometry` is a box.
    package init(preset file: PresetFile) {
        geometry = file.string("world", "geometry") ?? "box"
        if let v = file.double("world", "radius") { radius = World3DLevers.text(v) }
        if let n = file.int("world", "nside") { nside = String(n) }
        if let n = file.int("world", "layers") { layers = String(n) }
        if let v = file.double("physics", "ceiling_band") { ceilingBand = World3DLevers.text(v) }
    }

    /// The sphere values of a run's `config.json` (the nested config): a key it lacks is the engine default (the box).
    package init(config: [String: Any]) throws {
        func value(_ section: String, _ key: String) throws -> Any? {
            guard let raw = config[section] else { return nil }
            guard let table = raw as? [String: Any] else { throw LeverConfigError(message: "\(section) is not an object") }
            return table[key]
        }
        func number(_ section: String, _ key: String) throws -> NSNumber? {
            guard let raw = try value(section, key) else { return nil }
            guard let n = raw as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else {
                throw LeverConfigError(message: "\(section).\(key) is not a number")
            }
            return n
        }
        func integer(_ section: String, _ key: String) throws -> String? {
            guard let n = try number(section, key) else { return nil }
            guard let v = Int(exactly: n.doubleValue) else { throw LeverConfigError(message: "\(section).\(key) is not an integer") }
            return String(v)
        }
        func string(_ section: String, _ key: String) throws -> String? {
            guard let raw = try value(section, key) else { return nil }
            guard let s = raw as? String else { throw LeverConfigError(message: "\(section).\(key) is not a string") }
            return s
        }
        if let g = try string("world", "geometry") { geometry = g }
        if let n = try number("world", "radius") { radius = World3DLevers.text(n.doubleValue) }
        if let t = try integer("world", "nside") { nside = t }
        if let t = try integer("world", "layers") { layers = t }
        if let n = try number("physics", "ceiling_band") { ceilingBand = World3DLevers.text(n.doubleValue) }
    }

    /// Human-readable problems; the rules a field can break on its own (the engine's `validate()` for `geometry = "sphere"`). The rest
    /// (the shell against the radius, the cell volume, `layers x 12 nside^2` against the exact-sum limit, the weather's Courant number)
    /// the engine checks, and a rejection (exit code 2) appears in the log.
    package func validate() -> [String] {
        guard isSphere else { return [] }
        var errors: [String] = []
        if let v = Parse.decimal(radius) {
            if !(v >= 20) { errors.append("world.radius must be at least 20.") }
        } else {
            errors.append("world.radius must be a plain decimal number of at least 20.")
        }
        if let n = Parse.integer(nside) {
            if n < 2 { errors.append("world.nside must be at least 2 (HEALPix at Nside 1 has cells that touch six others).") }
        } else {
            errors.append("world.nside must be an integer of at least 2.")
        }
        if let n = Parse.integer(layers) {
            if n < 2 { errors.append("world.layers must be at least 2.") }
        } else {
            errors.append("world.layers must be an integer of at least 2.")
        }
        if Parse.decimal(ceilingBand) == nil { errors.append("physics.ceiling_band must be a plain decimal number >= 0 (0 = off).") }
        return errors
    }

    /// `--set` arguments for the values that differ from `baseline` (compared as numbers). Nothing unless the world is a sphere; the
    /// geometry is never passed. Invalid fields are skipped.
    package func setArguments(baseline: SphereLevers) -> [String] {
        guard isSphere else { return [] }
        var args: [String] = []
        func decimal(_ key: String, _ now: String, _ base: String) {
            guard let v = Parse.decimal(now), v != Parse.decimal(base) else { return }
            args += ["--set", "\(key)=\(v)"]
        }
        func integer(_ key: String, _ now: String, _ base: String) {
            guard let v = Parse.integer(now), v != Parse.integer(base) else { return }
            args += ["--set", "\(key)=\(v)"]
        }
        decimal("world.radius", radius, baseline.radius)
        integer("world.nside", nside, baseline.nside)
        integer("world.layers", layers, baseline.layers)
        decimal("physics.ceiling_band", ceilingBand, baseline.ceilingBand)
        return args
    }

    /// A line for the launcher: "sphere, radius 36, Nside 8, 8 layers", or "box".
    package var summary: String {
        isSphere ? "sphere, radius \(radius), Nside \(nside), \(layers) layers" : "box"
    }
}
