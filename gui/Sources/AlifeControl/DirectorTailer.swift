import Foundation

// MARK: - Rows

/// One intervention: an entry of a run's `director.jsonl` (the header lines are not entries).
struct InterventionRow: Identifiable, Equatable, Sendable {
    /// Byte offset of the line in the file, which stays the same while the file grows.
    let id: UInt64
    let step: Int
    /// `rule`, `llm` or `revert`; anything else the engine may add later is shown as it is.
    let source: String
    let action: String
    /// Compact "key=value" summary, floats at three significant digits.
    let params: String
    let reason: String
}

/// A one-line record of `journal.jsonl`: what a rule did, or a rule that was blocked.
struct JournalNote: Identifiable, Equatable, Sendable {
    let id: UInt64
    let step: Int
    let text: String
}

/// A `journal.jsonl` record of one LLM call.
struct LLMCall: Identifiable, Equatable, Sendable {
    let id: UInt64
    let step: Int
    let latency: Double?
    /// ok, none, rejected, failed or invalid.
    let outcome: String
    let model: String?
    /// The model's own account (its `journal` text).
    let journal: String?
    let accepted: Int
    let rejected: Int
    let error: String?
    /// The rule-tier actions that decided instead (a failed, invalid or fully rejected call); nil if there was
    /// no fallback.
    let fallback: [String]?
    /// The request messages as one text, and the reply as it arrived. Bounded for display.
    let prompt: String
    let response: String
}

enum JournalRow: Identifiable, Equatable, Sendable {
    case note(JournalNote)
    case call(LLMCall)

    var id: UInt64 {
        switch self {
        case .note(let n): return n.id
        case .call(let c): return c.id
        }
    }
}

/// What the Director views show for one run.
struct DirectorSnapshot: Equatable, Sendable {
    /// False when the run has no `director.jsonl` (it ran without a director).
    var hasLog: Bool
    /// The newest first.
    var interventions: [InterventionRow]
    /// The newest first.
    var journal: [JournalRow]
}

// MARK: - Reading the tail of a JSON-lines file

/// One line of a file with the byte offset where it starts.
struct TailLine: Sendable {
    let offset: UInt64
    let data: Data
}

enum JSONLTail {
    /// How much of the end of a file is read. The journal's LLM records hold the raw prompt (a few KB each).
    static let window: UInt64 = 256 * 1024

    /// The lines in the last `window` bytes of the file, or nil if it cannot be opened. A line cut by the start
    /// of the window is dropped.
    static func lines(at url: URL, window: UInt64 = JSONLTail.window) -> [TailLine]? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let size = try? handle.seekToEnd() else { return nil }
        guard size > 0 else { return [] }
        // Start one byte early, to see whether the window begins exactly at a line start.
        let start = size > window ? size - window - 1 : 0
        guard (try? handle.seek(toOffset: start)) != nil,
              let data = try? handle.read(upToCount: Int(size - start)) else { return nil }
        return lines(in: data, startOffset: start)
    }

    /// The lines of `data`, which begins at byte `startOffset` of the file. When it does not begin at the file's
    /// start, the first line is dropped: it may be cut, and the caller reads one byte early so that a line which
    /// starts right at the window edge is kept.
    static func lines(in data: Data, startOffset: UInt64) -> [TailLine] {
        var result: [TailLine] = []
        var lineStart = data.startIndex
        var first = true
        func add(_ range: Range<Data.Index>) {
            defer { first = false }
            guard !range.isEmpty else { return }
            if first && startOffset > 0 { return }  // the cut line
            result.append(TailLine(offset: startOffset + UInt64(range.lowerBound - data.startIndex),
                                   data: Data(data[range])))
        }
        for index in data.indices where data[index] == 0x0A {
            add(lineStart..<index)
            lineStart = index + 1
        }
        add(lineStart..<data.endIndex)  // a last line without a newline (it may still be partial: the parse skips it)
        return result
    }

    /// The line as a JSON object, or nil if it is not one (a torn or malformed line). Python's `json` may emit
    /// bare NaN / Infinity, which `JSONSerialization` rejects; those become null if the plain parse fails.
    static func object(_ line: Data) -> [String: Any]? {
        if let object = try? JSONSerialization.jsonObject(with: line) as? [String: Any] { return object }
        var text = String(decoding: line, as: UTF8.self)
        guard text.contains("NaN") || text.contains("Infinity") else { return nil }
        text = text.replacingOccurrences(of: "-Infinity", with: "null")
            .replacingOccurrences(of: "Infinity", with: "null")
            .replacingOccurrences(of: "NaN", with: "null")
        return try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]
    }
}

// MARK: - The two views

enum DirectorTailer {
    static let interventionLimit = 10
    static let journalLimit = 20
    /// Raw prompt and response text are cut here (a reply may be up to 1 MiB; the views stay responsive).
    static let textLimit = 20_000

    /// The snapshot of the run directory `runDirectory`. Missing files, a torn last line and malformed lines
    /// are tolerated; unknown keys are ignored.
    static func snapshot(runDirectory: URL) -> DirectorSnapshot {
        guard let log = JSONLTail.lines(at: runDirectory.appendingPathComponent("director.jsonl")) else {
            return DirectorSnapshot(hasLog: false, interventions: [], journal: [])
        }
        let journal = JSONLTail.lines(at: runDirectory.appendingPathComponent("journal.jsonl")) ?? []
        return DirectorSnapshot(hasLog: true, interventions: interventions(log), journal: journalRows(journal))
    }

    /// The last `limit` interventions of a log's lines, newest first.
    static func interventions(_ lines: [TailLine], limit: Int = interventionLimit) -> [InterventionRow] {
        var rows: [InterventionRow] = []
        for line in lines.reversed() {
            guard rows.count < limit, let object = JSONLTail.object(line.data),
                  object["kind"] == nil,  // header lines carry a kind; entries do not
                  let step = int(object["step_applied"]), let action = object["action"] as? String else { continue }
            rows.append(InterventionRow(
                id: line.offset, step: step, source: object["source"] as? String ?? "?", action: action,
                params: summary(object["params"]), reason: object["reason"] as? String ?? ""))
        }
        return rows
    }

    /// The last `limit` journal records, newest first.
    static func journalRows(_ lines: [TailLine], limit: Int = journalLimit) -> [JournalRow] {
        var rows: [JournalRow] = []
        for line in lines.reversed() {
            guard rows.count < limit, let object = JSONLTail.object(line.data), let step = int(object["step"]) else {
                continue
            }
            if object["kind"] as? String == "llm_call" {
                rows.append(.call(call(object, id: line.offset, step: step)))
            } else if object["kind"] == nil, let text = object["text"] as? String {
                rows.append(.note(JournalNote(id: line.offset, step: step, text: text)))
            }
        }
        return rows
    }

    private static func call(_ object: [String: Any], id: UInt64, step: Int) -> LLMCall {
        var fallback: [String]?
        if let f = object["fallback"] as? [String: Any] { fallback = (f["actions"] as? [String]) ?? [] }
        let messages = (object["messages"] as? [[String: Any]] ?? []).map {
            "[\($0["role"] as? String ?? "?")]\n\($0["content"] as? String ?? "")"
        }
        return LLMCall(
            id: id, step: step, latency: (object["latency_s"] as? NSNumber)?.doubleValue,
            outcome: object["outcome"] as? String ?? "?", model: object["model"] as? String,
            journal: object["journal"] as? String, accepted: count(object["accepted"]),
            rejected: count(object["rejected"]), error: object["error"] as? String, fallback: fallback,
            prompt: bounded(messages.joined(separator: "\n\n")),
            response: bounded(object["response"] as? String ?? "(no reply)"))
    }

    private static func bounded(_ text: String) -> String {
        guard text.count > textLimit else { return text }
        return String(text.prefix(textLimit)) + "\n... (\(text.count - textLimit) more characters not shown)"
    }

    private static func count(_ value: Any?) -> Int {
        (value as? [Any])?.count ?? 0
    }

    /// An integer-valued JSON number (not a boolean).
    private static func int(_ value: Any?) -> Int? {
        guard let n = value as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else { return nil }
        return Int(exactly: n.doubleValue)
    }

    // MARK: Display formatting

    /// "key=value" pairs of an action's params, sorted by key; floats at three significant digits.
    static func summary(_ params: Any?) -> String {
        guard let table = params as? [String: Any], !table.isEmpty else { return "" }
        return table.keys.sorted().map { "\($0)=\(format(table[$0]))" }.joined(separator: " ")
    }

    static func format(_ value: Any?) -> String {
        switch value {
        case let s as String: return s
        case let n as NSNumber:
            if CFGetTypeID(n) == CFBooleanGetTypeID() { return n.boolValue ? "true" : "false" }
            return CFNumberIsFloatType(n) ? significant(n.doubleValue) : n.stringValue
        case let a as [Any]: return "[" + a.map { format($0) }.joined(separator: ", ") + "]"
        case let d as [String: Any]: return "{" + summary(d).replacingOccurrences(of: " ", with: ", ") + "}"
        case nil, is NSNull: return "null"
        default: return "\(value!)"
        }
    }

    /// `value` rounded to three significant digits, without an exponent where that reads naturally.
    static func significant(_ value: Double) -> String {
        guard value.isFinite else { return "null" }
        if value == 0 { return "0" }
        let magnitude = abs(value)
        if magnitude >= 1000 {
            let digits = Int(log10(magnitude).rounded(.down)) + 1
            let unit = pow(10.0, Double(digits - 3))
            let rounded = (value / unit).rounded() * unit
            return rounded.magnitude < 1e15 ? String(Int64(rounded)) : String(format: "%.2e", rounded)
        }
        return String(format: "%.3g", value)
    }
}
