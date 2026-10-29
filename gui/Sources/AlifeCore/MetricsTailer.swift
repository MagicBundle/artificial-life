import Foundation

/// The latest row of a run's metrics.jsonl (one JSON object per metrics interval).
package struct MetricsSnapshot: Equatable {
    package var step: Int
    package var nAlive: Double?
    package var stepsPerSecond: Double?
    package var clades: Double?
    package var generationMax: Double?
    package var ledgerError: Double?
    // The keys of a dims = 3 run (DESIGN §27.9, in every row); nil in a 2D run, and in a 3D row with nobody alive.
    package var heightMean: Double?
    package var groundedFrac: Double?
    package var surfaceFrac: Double?
    /// The numeric keys of the feature layers' rows (`build_`, `arms_`, `bond_`, `comm_`, `farm_`; DESIGN §45), by key. A feature that is off
    /// writes none, so the dictionary is empty for a world without them.
    package var features: [String: Double] = [:]
}

/// The metrics keys of the feature layers, grouped for the status readout. Any numeric key with a feature's prefix is shown, so a key the
/// engine adds later needs no change here; a list-valued key (`bond_flow_by_role`) and a null (a mean over no attacks) are left out.
package enum FeatureMetrics {
    package struct Row: Equatable, Sendable {
        package var key: String
        package var label: String
        package var value: Double
    }

    package struct Group: Equatable, Sendable {
        package var title: String
        package var rows: [Row]
    }

    /// Prefix and title, in the order of the feature table.
    package static let prefixes: [(prefix: String, title: String)] = [
        ("arms_", "Arms race"), ("build_", "Structures"), ("bond_", "Bonds and roles"), ("comm_", "Communication"),
        ("farm_", "Farming"),
    ]

    package static func groups(_ features: [String: Double]) -> [Group] {
        prefixes.compactMap { prefix, title in
            let rows = features.filter { $0.key.hasPrefix(prefix) }.keys.sorted().map { key in
                Row(key: key, label: String(key.dropFirst(prefix.count)).replacingOccurrences(of: "_", with: " "), value: features[key]!)
            }
            return rows.isEmpty ? nil : Group(title: title, rows: rows)
        }
    }

    /// A value for the readout: an integer as it is, a fraction to three significant digits.
    package static func text(_ value: Double) -> String {
        if value == value.rounded(), abs(value) < 1e12 { return Numbers.grouped(Int(value)) }
        return String(format: abs(value) >= 100 ? "%.0f" : abs(value) >= 1 ? "%.2f" : "%.3f", value)
    }
}

package enum MetricsTailer {
    /// Reads only the tail of the file (64 KiB) and returns the newest line that parses as a complete JSON
    /// object, so a torn last line (the engine is mid-append) falls back to the previous row.
    package static func readLast(at url: URL) -> MetricsSnapshot? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let size = try? handle.seekToEnd(), size > 0 else { return nil }
        let window: UInt64 = 64 * 1024
        let start = size > window ? size - window : 0
        guard (try? handle.seek(toOffset: start)) != nil,
              let data = try? handle.read(upToCount: Int(size - start)) else { return nil }

        for piece in data.split(separator: 0x0A, omittingEmptySubsequences: true).reversed() {
            guard let snapshot = parse(Data(piece)) else { continue }
            return snapshot
        }
        return nil
    }

    package static func parse(_ line: Data) -> MetricsSnapshot? {
        // Python's json module may emit bare NaN / Infinity, which JSONSerialization rejects.
        var text = String(decoding: line, as: UTF8.self)
        if text.contains("NaN") || text.contains("Infinity") {
            text = text.replacingOccurrences(of: "-Infinity", with: "null")
                .replacingOccurrences(of: "Infinity", with: "null")
                .replacingOccurrences(of: "NaN", with: "null")
        }
        guard let object = try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any],
              let step = (object["step"] as? NSNumber)?.intValue else { return nil }
        func number(_ key: String) -> Double? { (object[key] as? NSNumber)?.doubleValue }
        var features: [String: Double] = [:]
        for (key, raw) in object where FeatureMetrics.prefixes.contains(where: { key.hasPrefix($0.prefix) }) {
            if let n = raw as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(), n.doubleValue.isFinite { features[key] = n.doubleValue }
        }
        return MetricsSnapshot(
            step: step,
            nAlive: number("n_alive"),
            stepsPerSecond: number("steps_per_s"),
            clades: number("clades_alive"),
            generationMax: number("generation_max"),
            ledgerError: number("ledger_error"),
            heightMean: number("height_mean"),
            groundedFrac: number("grounded_frac"),
            surfaceFrac: number("surface_frac"),
            features: features
        )
    }
}
