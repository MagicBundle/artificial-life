import Foundation

/// The scalar settings of a preset file (`src/alife/presets/NAME.toml`): the integer, floating-point and boolean values
/// of its tables. Presets are sparse TOML overrides of the engine defaults written one `key = value` per line,
/// so this reads exactly that and nothing more: strings and arrays are skipped, as are inline tables
/// and multi-line values. A key the file does not set is nil, and the caller applies the engine default.
///
/// It is not a TOML parser, and agrees with one on the files `tomllib` accepts. A key set twice is the last value (a
/// parser refuses the file; the cadence reader this replaced kept the last value that was an integer of at least 1),
/// and a table header that is not closed names no table, as in that reader, so its keys are ignored.
package struct PresetFile: Equatable, Sendable {
    package enum Value: Equatable, Sendable {
        case int(Int)
        case double(Double)
        /// `true` or `false` (the overlays of the feature layers set their switches this way).
        case bool(Bool)
    }

    /// Table name (`run`, `world`, ...; "" before the first table) to its scalar keys.
    private var tables: [String: [String: Value]] = [:]

    package init() {}

    package init(text: String) {
        var table = ""
        for raw in text.split(whereSeparator: \.isNewline) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("[") {
                // `[run]` and `[run] # comment` name the table `run`; the name is not trimmed, so `[ run ]` names
                // another table (the cadence reader this replaced matched `[run]` literally, and so does this). A
                // header without its `]` names no table: the NUL cannot be a table name in a file.
                let name = line.dropFirst().prefix { $0 != "]" }
                table = name.endIndex == line.endIndex ? "\u{0}" : String(name)
                continue
            }
            guard let eq = line.firstIndex(of: "=") else { continue }
            let key = line[..<eq].trimmingCharacters(in: .whitespaces)
            // Drop a trailing comment and TOML digit separators.
            let value = line[line.index(after: eq)...].split(separator: "#", maxSplits: 1,
                                                              omittingEmptySubsequences: false)[0]
                .trimmingCharacters(in: .whitespaces).replacingOccurrences(of: "_", with: "")
            if let scalar = Self.scalar(value) { tables[table, default: [:]][key] = scalar }
        }
    }

    /// An integer (`Int(value)`), else a finite decimal or exponent number; nil for anything else (`inf`, `nan`
    /// and hexadecimal are not presets' business).
    private static func scalar(_ text: String) -> Value? {
        if text == "true" { return .bool(true) }
        if text == "false" { return .bool(false) }
        if let n = Int(text) { return .int(n) }
        guard text.range(of: #"^[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$"#, options: .regularExpression) != nil,
              let v = Double(text), v.isFinite else { return nil }
        return .double(v)
    }

    package func value(_ table: String, _ key: String) -> Value? { tables[table]?[key] }

    /// The integer at `table.key`; nil if absent or not an integer (`1.0` is a float in TOML).
    package func int(_ table: String, _ key: String) -> Int? {
        if case .int(let n) = value(table, key) { return n }
        return nil
    }

    /// The number at `table.key`; an integer counts (`height = 24` and `height = 24.0` are the same setting).
    package func double(_ table: String, _ key: String) -> Double? {
        switch value(table, key) {
        case .int(let n): return Double(n)
        case .double(let v): return v
        case .bool, nil: return nil
        }
    }

    /// The boolean at `table.key`; nil if absent or not `true` or `false`.
    package func bool(_ table: String, _ key: String) -> Bool? {
        if case .bool(let b) = value(table, key) { return b }
        return nil
    }

    /// The preset `name` in the repository, or nil if its file cannot be read.
    package static func preset(_ name: String, repo: URL) -> PresetFile? {
        let file = repo.appendingPathComponent("src/alife/presets/\(name).toml")
        guard let text = try? String(contentsOf: file, encoding: .utf8) else { return nil }
        return PresetFile(text: text)
    }
}
