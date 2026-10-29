import CryptoKit
import Foundation
import Testing

@testable import AlifeCore

/// A committed play stream of `tests/data` (`play_tiny3d`, `play_tiny2d`, `play_tiny3d_n70` and G3b's `play_tiny3d_g3b`: `.bin` and `.json`) and its
/// sidecar, which holds the expected decoding produced by the independent Python decoder (`tests/play_decode.py`).
struct PlayFixture {
    let kind: String
    let bytes: [UInt8]
    let sidecar: [String: Any]

    static let kinds = ["3d", "2d", "3d_n70", "3d_g3b"]

    init(kind: String) throws {
        self.kind = kind
        bytes = [UInt8](try Data(contentsOf: Repo.streamData.appendingPathComponent("play_tiny\(kind).bin")))
        let json = try Data(contentsOf: Repo.streamData.appendingPathComponent("play_tiny\(kind).json"))
        sidecar = try #require(try JSONSerialization.jsonObject(with: json) as? [String: Any])
    }

    var data: Data { Data(bytes) }
    var header: [String: Any] { sidecar["header"] as! [String: Any] }
    var records: [[String: Any]] { sidecar["records"] as! [[String: Any]] }
    var skipped: [[String: Any]] { sidecar["skipped"] as! [[String: Any]] }

    /// `(offset of the tag, payload length)` of a decoded record.
    func span(_ record: [String: Any]) -> (offset: Int, length: Int) { (record["offset"] as! Int, record["length"] as! Int) }

    /// The offset of the first record: the end of the header.
    var headerEnd: Int {
        let offsets = records.map { $0["offset"] as! Int } + skipped.map { $0["offset"] as! Int }
        return offsets.min()!
    }

    /// `(offset, length)` of every record, known or skipped, in stream order.
    var allSpans: [(offset: Int, length: Int)] {
        let known = records.map { span($0) }
        let unknown = skipped.map { entry -> (offset: Int, length: Int) in
            let offset = entry["offset"] as! Int
            return (offset, Int(le32(at: offset + 4)))
        }
        return (known + unknown).sorted { $0.offset < $1.offset }
    }

    /// The offsets at which the stream is complete: after the header and after each record.
    var boundaries: [Int] { [headerEnd] + allSpans.map { $0.offset + 8 + $0.length } }

    func le32(at offset: Int) -> UInt32 {
        (0..<4).reduce(UInt32(0)) { $0 | UInt32(bytes[offset + $1]) << (8 * UInt32($1)) }
    }

    /// The bytes of the whole record at `offset` (tag, length and payload).
    func raw(at offset: Int) -> [UInt8] {
        let length = Int(le32(at: offset + 4))
        return Array(bytes[offset..<(offset + 8 + length)])
    }

    /// All events of the stream, decoded in one piece.
    func events() throws -> [StreamEvent] {
        var decoder = FrameDecoder()
        let events = try decoder.feed(data)
        try decoder.finish()
        return events
    }

    /// The offsets of the records of `tag` (known), in stream order.
    func offsets(of tag: String) -> [Int] { records.filter { $0["tag"] as? String == tag }.map { $0["offset"] as! Int } }

    /// The offset of the `index`th record of `tag` (known).
    func offset(of tag: String, index: Int = 0) throws -> Int {
        let matches = offsets(of: tag)
        return try #require(matches.indices.contains(index) ? matches[index] : nil)
    }
}

// MARK: - The bytes of a decoded record, for the sidecar's SHA-256 digests

private func raw<T>(_ values: [T]) -> Data { values.withUnsafeBytes { Data($0) } }

private func le<T: FixedWidthInteger>(_ v: T) -> [UInt8] {
    (0..<MemoryLayout<T>.size).map { UInt8(truncatingIfNeeded: v >> (8 * T(truncatingIfNeeded: $0))) }
}

private func leFloat(_ v: Float) -> [UInt8] { le(v.bitPattern) }

/// The sections of a decoded record that are arrays or fixed fields, as the stream carried them (padding included).
/// The JSON sections are checked against the sidecar by value instead (`PlayFixture.jsonSection`).
func playSectionBytes(_ record: StreamRecord) -> [String: Data] {
    switch record {
    case .frma(let f):
        return [
            "bitmap": raw(f.bitmap), "pos": raw(f.pos), "orient": raw(f.orient), "rgb": raw(f.rgb),
            "uid": raw(f.uid), "bonds": raw(f.bonds),
        ]
    case .idnt(let r):
        var rows: [UInt8] = []
        for row in r.rows {
            rows += le(row.uid) + le(UInt32(bitPattern: row.clade)) + [row.house, row.kin, row.colony, row.flags]
        }
        return ["rows": Data(rows)]
    case .intn(let r):
        var rows: [UInt8] = []
        for row in r.rows { rows += le(row.uid) + [row.archPrimary, row.archSecondary, row.intent, row.flags] }
        return ["rows": Data(rows)]
    case .terr(let t):
        return ["owner": Data(t.owner), "purity": Data(t.purity)]
    case .map1(let m):
        return ["P": Data(m.p), "Dc": Data(m.dc), "zeta": raw(m.zeta), "Ts": Data(m.ts), "winds": raw(m.winds)]
    case .clck(let c):
        var b: [UInt8] = leFloat(c.rate) + leFloat(c.achieved) + leFloat(c.stepMs) + leFloat(c.frameMs)
        b += le(c.lateSteps) + le(c.saveStallSteps) + [c.reason.rawValue, c.saving ? 1 : 0] + le(c.flags)
        b += le(c.fe) + le(c.chapter) + le(c.inChapter) + le(c.chapterLen)
        return ["body": Data(b)]
    case .focs(let f):
        var b: [UInt8] = le(f.uid) + le(f.row) + le(f.flags)
        b += leFloat(f.x) + leFloat(f.y) + leFloat(f.z) + leFloat(f.vx) + leFloat(f.vy) + leFloat(f.vz)
        b += le(UInt32(bitPattern: f.energy)) + le(f.age) + le(f.generation) + le(f.parentUid)
        b += [f.nBonds, f.house, f.kin, f.colony, f.archPrimary, f.archSecondary, f.intent, f.labelFlags]
        return ["body": Data(b)]
    case .strc(let c), .fsol(let c):
        return ["idx": raw(c.indices), "level": Data(c.levels)]
    case .fsig(let v):
        return ["says": Data(v.says), "hears": Data(v.hears)]
    case .strm, .evnt, .envs, .alrt, .rprt, .plan:
        return [:]
    }
}

/// `object` as JSON data and back, without `null` members (a Swift `nil` is encoded as absent, the writer's `null`).
func normalisedJSON<T: Encodable>(_ value: T) throws -> Any {
    let data = try JSONEncoder().encode(value)
    return strippingNulls(try JSONSerialization.jsonObject(with: data))
}

func strippingNulls(_ any: Any) -> Any {
    if let dict = any as? [String: Any] { return dict.compactMapValues { $0 is NSNull ? nil : strippingNulls($0) } }
    if let array = any as? [Any] { return array.map(strippingNulls) }
    return any
}

func sameJSON(_ a: Any, _ b: Any) -> Bool { (a as AnyObject).isEqual(b) }

/// Feeds `bytes` to a new decoder in pieces of `chunk` bytes (the whole input when nil), like `decode(_:chunk:)`, and also
/// returns the decoder for its counters.
func decodeWithDecoder(_ bytes: [UInt8], chunk: Int? = nil) -> (events: [StreamEvent], error: StreamError?, decoder: FrameDecoder) {
    var decoder = FrameDecoder()
    var events: [StreamEvent] = []
    let size = chunk ?? max(bytes.count, 1)
    var start = 0
    var failure: StreamError?
    do {
        repeat {
            let end = min(start + size, bytes.count)
            events += try decoder.feed(Data(bytes[start..<end]))
            start = end
        } while start < bytes.count
        try decoder.finish()
    } catch let error as StreamError {
        failure = error
    } catch {
        Issue.record("an error that is not a StreamError: \(error)")
    }
    return (events, failure, decoder)
}
