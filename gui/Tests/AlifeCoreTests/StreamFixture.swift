import CryptoKit
import Foundation
import Testing

@testable import AlifeCore

/// A committed stream of `tests/data` (`stream_tiny_2d.bin`, `stream_tiny_3d.bin`) and its JSON sidecar, which holds
/// the expected decoding produced by the independent Python decoder (`tests/stream_decode.py`).
struct StreamFixture {
    let kind: String
    let bytes: [UInt8]
    let sidecar: [String: Any]

    init(kind: String) throws {
        self.kind = kind
        bytes = [UInt8](try Data(contentsOf: Repo.streamData.appendingPathComponent("stream_tiny_\(kind).bin")))
        let json = try Data(contentsOf: Repo.streamData.appendingPathComponent("stream_tiny_\(kind).json"))
        sidecar = try #require(try JSONSerialization.jsonObject(with: json) as? [String: Any])
    }

    var data: Data { Data(bytes) }
    var header: [String: Any] { sidecar["header"] as! [String: Any] }
    var frames: [[String: Any]] { sidecar["frames"] as! [[String: Any]] }

    /// The offset of the first record: the end of the header.
    var headerEnd: Int { frames[0]["offset"] as! Int }

    /// `(offset of the tag, declared length)` of each frame record.
    var records: [(offset: Int, length: Int)] { frames.map { ($0["offset"] as! Int, $0["length"] as! Int) } }

    /// The offsets at which the stream is complete: after the header and after each record.
    var boundaries: [Int] { [headerEnd] + records.map { $0.offset + 8 + $0.length } }

    /// All events of the stream, decoded in one piece.
    func events() throws -> [StreamEvent] {
        var decoder = FrameDecoder()
        let events = try decoder.feed(data)
        try decoder.finish()
        return events
    }
}

/// Feeds `bytes` to a new decoder in pieces of `chunk` bytes (the whole input when nil) and ends the input. Returns
/// the events completed and the error that ended the stream, if any, however it was found.
func decode(_ bytes: [UInt8], chunk: Int? = nil) -> (events: [StreamEvent], error: StreamError?) {
    var decoder = FrameDecoder()
    var events: [StreamEvent] = []
    let size = chunk ?? max(bytes.count, 1)
    var start = 0
    do {
        repeat {
            let end = min(start + size, bytes.count)
            events += try decoder.feed(Data(bytes[start..<end]))
            start = end
        } while start < bytes.count
        try decoder.finish()
        return (events, nil)
    } catch let error as StreamError {
        return (events, error)
    } catch {
        Issue.record("an error that is not a StreamError: \(error)")
        return (events, nil)
    }
}

/// The bytes of the sections of a decoded frame as the stream carried them (padding included), for the sidecar's
/// SHA-256 digests.
func sectionBytes(_ frame: StreamFrame) -> [String: Data] {
    func raw<T>(_ values: [T]) -> Data { values.withUnsafeBytes { Data($0) } }
    var aligned = frame.alive
    aligned += [UInt8](repeating: 0, count: (4 - aligned.count % 4) % 4)
    var sections = [
        "alive": Data(aligned), "pos": raw(frame.pos), "orient": raw(frame.orient), "rgb": raw(frame.rgb),
        "bonds": raw(frame.bonds),
    ]
    if let field = frame.field { sections["field"] = raw(field) }
    return sections
}

func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func le32(_ v: Int) -> [UInt8] { (0..<4).map { UInt8((v >> (8 * $0)) & 0xFF) } }

/// `bytes` with `replacement` written over them at `offset`.
func patched(_ bytes: [UInt8], at offset: Int, _ replacement: [UInt8]) -> [UInt8] {
    var copy = bytes
    copy.replaceSubrange(offset..<(offset + replacement.count), with: replacement)
    return copy
}

/// A stream written straight from the text of DESIGN §27.11, for shapes the committed fixtures (N = 64, a multiple
/// of 4) lack. Row `r` has position (r, 2r, 3r + 0.5), the half-precision yaw `0x3C00 + r` and pitch `0xBC00 + r`, and the
/// colour `0x010203 + r`.
struct SyntheticStream {
    var rows: Int
    var fieldGrid = 2
    var fieldLayers = 3
    var dims = 3

    func preamble() -> [UInt8] {
        let json = """
        {"protocol":1,"dims":\(dims),"L":16.0,"height":\(dims == 3 ? "12.0" : "0.0"),"N":\(rows),"B":4,"radius":1.0,\
        "field_cell":4.0,"FG":\(fieldGrid),"FZ":\(fieldLayers),"NCH":3,"color_modes":["tag","energy","clade"],\
        "color":"tag","fields":false,"field_z":0,"step":0,"run":null,"config_hash":"h","mode":"simulate","every":1,\
        "source":"preset"}
        """
        var text = Array(json.utf8)
        while text.count % 4 != 0 { text.append(0x20) }
        return Array("ALSTRM".utf8) + [0, 1] + le32(text.count) + text
    }

    /// The `FRM1` record of a frame with the given alive rows and bonds, and a field section of layer `layer`.
    func frame(step: Int, alive: [Int], bonds: [(Int, Int)] = [], layer: Int? = nil, flags: Int = 0) -> [UInt8] {
        var payload: [UInt8] = (0..<8).map { UInt8((step >> (8 * $0)) & 0xFF) }
        payload += le32(alive.count) + le32(flags | (layer == nil ? 0 : 1 | layer! << 16)) + le32(bonds.count) + le32(2)
        payload += [UInt8](repeating: 0, count: 8)
        var aliveBytes = (0..<rows).map { alive.contains($0) ? UInt8(1) : 0 }
        aliveBytes += [UInt8](repeating: 0, count: (4 - rows % 4) % 4)
        payload += aliveBytes
        for r in 0..<rows { payload += [Float(r), Float(2 * r), Float(3 * r) + 0.5].flatMap { le32(Int($0.bitPattern)) } }
        for r in 0..<rows { payload += [UInt8(truncatingIfNeeded: r), 0x3C, UInt8(truncatingIfNeeded: r), 0xBC] }
        for r in 0..<rows { payload += le32(0x010203 + r) }
        for (i, j) in bonds { payload += le32(i) + le32(j) }
        if let layer { for c in 0..<(fieldGrid * fieldGrid) { payload += le32(0x100000 * (layer + 1) + c) } }
        return Array("FRM1".utf8) + le32(payload.count) + payload
    }
}
