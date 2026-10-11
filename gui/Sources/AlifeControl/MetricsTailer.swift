import Foundation

/// The latest row of a run's metrics.jsonl (one JSON object per metrics interval).
struct MetricsSnapshot: Equatable {
    var step: Int
    var nAlive: Double?
    var stepsPerSecond: Double?
    var clades: Double?
    var generationMax: Double?
    var ledgerError: Double?
}

enum MetricsTailer {
    /// Reads only the tail of the file (64 KiB) and returns the newest line that parses as a complete JSON
    /// object, so a torn last line (the engine is mid-append) falls back to the previous row.
    static func readLast(at url: URL) -> MetricsSnapshot? {
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

    private static func parse(_ line: Data) -> MetricsSnapshot? {
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
        return MetricsSnapshot(
            step: step,
            nAlive: number("n_alive"),
            stepsPerSecond: number("steps_per_s"),
            clades: number("clades_alive"),
            generationMax: number("generation_max"),
            ledgerError: number("ledger_error")
        )
    }
}
