import Foundation
import Testing

@testable import AlifeCore

/// `FrameDecoder` against DESIGN §27.11: the committed fixtures decoded and checked against the sidecars of the
/// independent Python decoder, chunking, and every refusal the protocol asks for.
@Suite struct StreamDecoderTests {
    static let kinds = ["2d", "3d"]
    /// Bytes per `feed`: one at a time, an odd size that cuts every field, a pipe-sized read, and the whole input.
    static let chunkings: [Int?] = [1, 7, 4096, nil]

    // MARK: Fixtures

    @Test(arguments: kinds)
    func fixtureMatchesItsSidecar(kind: String) throws {
        let fixture = try StreamFixture(kind: kind)
        let events = try fixture.events()
        #expect(events.count == 1 + fixture.frames.count)

        guard case .header(let header) = events[0] else { Issue.record("the first event is not the header"); return }
        let expected = fixture.header
        #expect(header.protocolVersion == expected["protocol"] as? Int)
        #expect(header.dims == expected["dims"] as? Int)
        #expect(header.size == expected["L"] as? Double)
        #expect(header.height == expected["height"] as? Double)
        #expect(header.capacity == expected["N"] as? Int)
        #expect(header.bondSlots == expected["B"] as? Int)
        #expect(header.radius == expected["radius"] as? Double)
        #expect(header.fieldCell == expected["field_cell"] as? Double)
        #expect(header.fieldGrid == expected["FG"] as? Int)
        #expect(header.fieldLayers == expected["FZ"] as? Int)
        #expect(header.channels == expected["NCH"] as? Int)
        #expect(header.colorModes == expected["color_modes"] as? [String])
        #expect(header.color == expected["color"] as? String)
        #expect(header.fields == expected["fields"] as? Bool)
        #expect(header.fieldZ == expected["field_z"] as? Int)
        #expect(header.step == expected["step"] as? Int)
        #expect(header.run == nil && expected["run"] is NSNull)
        #expect(header.configHash == expected["config_hash"] as? String)
        #expect(header.mode == expected["mode"] as? String)
        #expect(header.every == expected["every"] as? Int)
        #expect(header.source == expected["source"] as? String)
        #expect(header.dims == (kind == "3d" ? 3 : 2))
        #expect(fixture.bytes.count == fixture.sidecar["bytes"] as? Int, "the sidecar describes this file")
        #expect(fixture.sidecar["kind"] as? String == kind)

        for (index, sidecar) in fixture.frames.enumerated() {
            guard case .frame(let frame) = events[1 + index] else { Issue.record("event \(1 + index) is not a frame"); continue }
            try check(frame, against: sidecar, header: header, label: "\(kind) frame \(index)")
        }
    }

    private func check(_ frame: StreamFrame, against sidecar: [String: Any], header: StreamHeader, label: String) throws {
        #expect(frame.step == UInt64(sidecar["step"] as! Int), "\(label) step")
        #expect(frame.nAlive == sidecar["n_alive"] as? Int, "\(label) n_alive")
        #expect(frame.flags == UInt32(sidecar["flags"] as! Int), "\(label) flags")
        #expect(frame.nBonds == sidecar["n_bonds"] as? Int, "\(label) n_bonds")
        #expect(frame.colorIndex == sidecar["color_index"] as? Int, "\(label) color_index")
        #expect(header.colorMode(of: frame) == sidecar["color"] as? String, "\(label) colour mode")
        #expect(frame.hasDiagError == sidecar["diag_error"] as? Bool, "\(label) diag error")
        if let layer = sidecar["field_layer"] as? Int {
            #expect(frame.hasField && frame.fieldLayer == layer, "\(label) field layer")
        } else {
            #expect(!frame.hasField && frame.fieldLayer == nil && frame.field == nil, "\(label) has no field")
        }

        // Row counts: every per-agent section has N rows; the bond list has n_bonds pairs.
        let n = header.capacity
        #expect(frame.rows == n && frame.pos.count == 3 * n && frame.orient.count == 2 * n && frame.rgb.count == n)
        #expect(frame.bonds.count == 2 * frame.nBonds)

        #expect((0..<n).filter(frame.isAlive) == sidecar["alive_rows"] as? [Int], "\(label) alive rows")
        let bonds = sidecar["bonds"] as! [[Int]]
        #expect((0..<frame.nBonds).map { [frame.bond($0).i, frame.bond($0).j] } == bonds, "\(label) bonds")

        for agent in sidecar["agents"] as! [[String: Any]] {
            let row = agent["row"] as! Int
            let p = frame.position(row)
            #expect([Double(p.x), Double(p.y), Double(p.z)] == agent["pos"] as? [Double], "\(label) row \(row) position")
            #expect(Double(frame.yaw(row)) == agent["yaw"] as? Double, "\(label) row \(row) yaw")
            #expect(Double(frame.pitch(row)) == agent["pitch"] as? Double, "\(label) row \(row) pitch")
            #expect("#" + String(format: "%06x", frame.rgb[row]) == agent["rgb"] as? String, "\(label) row \(row) colour")
        }
        if let first = sidecar["field_first"] as? [String] {
            let field = try #require(frame.field)
            #expect(field.count == header.fieldGrid * header.fieldGrid)
            #expect(field.prefix(first.count).map { "#" + String(format: "%06x", $0) } == first, "\(label) field cells")
        }

        // Every section byte for byte, padding included.
        let digests = sidecar["sha256"] as! [String: String]
        let sections = sectionBytes(frame)
        #expect(Set(sections.keys) == Set(digests.keys), "\(label) sections")
        for (name, digest) in digests { #expect(sha256(sections[name] ?? Data()) == digest, "\(label) \(name) section") }
    }

    @Test(arguments: kinds)
    func twoDStreamsHaveFlatZAndNoPitch(kind: String) throws {
        guard kind == "2d" else { return }
        let events = try StreamFixture(kind: kind).events()
        for case .frame(let frame) in events {
            for row in 0..<frame.rows {
                #expect(frame.position(row).z == 0 && frame.pitch(row) == 0)
            }
        }
    }

    // MARK: Chunking

    @Test(arguments: kinds, chunkings)
    func anyChunkingGivesTheSameEvents(kind: String, chunk: Int?) throws {
        let fixture = try StreamFixture(kind: kind)
        let whole = try fixture.events()
        let result = decode(fixture.bytes, chunk: chunk)
        #expect(result.error == nil)
        #expect(result.events == whole, "chunk \(chunk.map(String.init) ?? "whole")")
    }

    @Test func eventsArriveWhenTheirLastByteDoes() throws {
        let fixture = try StreamFixture(kind: "3d")
        var decoder = FrameDecoder()
        var seen: [(event: StreamEvent, end: Int)] = []
        for (index, byte) in fixture.bytes.enumerated() {
            for event in try decoder.feed(Data([byte])) { seen.append((event, index + 1)) }
        }
        try decoder.finish()
        #expect(seen.map(\.end) == fixture.boundaries, "the header and each frame complete at the end of their bytes")
    }

    @Test func aLargeFrameDecodesInSmallChunks() throws {
        let rows = 65_536
        let grid = 80
        let json = """
        {"protocol": 1, "dims": 3, "L": 320.0, "height": 80.0, "N": \(rows), "B": 4, "radius": 1.0, \
        "field_cell": 4.0, "FG": \(grid), "FZ": 20, "NCH": 3, "color_modes": ["tag", "energy", "clade"], \
        "color": "tag", "fields": true, "field_z": 3, "step": 5, "run": "runs/big", "config_hash": "x", \
        "mode": "follow", "every": 0, "source": "run"}
        """
        let nBonds = 3000
        var bytes = Array("ALSTRM".utf8) + [0, 1]
        var text = Array(json.utf8)
        while text.count % 4 != 0 { text.append(0x20) }
        bytes += le32(text.count) + text
        var payload = [UInt8](repeating: 0, count: 8)
        payload[0] = 5
        payload += le32(rows) + le32(1 | 3 << 16) + le32(nBonds) + le32(1) + [UInt8](repeating: 0, count: 8)
        payload += [UInt8](repeating: 1, count: rows)  // 65536 is a multiple of 4: no padding
        payload += [UInt8](repeating: 0x3F, count: 12 * rows)
        payload += [UInt8](repeating: 0x11, count: 4 * rows)
        payload += [UInt8](repeating: 0x22, count: 4 * rows)
        for k in 0..<nBonds { payload += le32(k) + le32(k + 1) }
        payload += [UInt8](repeating: 0x33, count: 4 * grid * grid)
        bytes += Array("FRM1".utf8) + le32(payload.count) + payload

        let result = decode(bytes, chunk: 1000)
        #expect(result.error == nil)
        #expect(result.events.count == 2)
        guard case .frame(let frame) = result.events.last else { Issue.record("no frame"); return }
        #expect(frame.step == 5 && frame.nAlive == rows && frame.nBonds == nBonds)
        #expect(frame.bond(nBonds - 1).i == nBonds - 1 && frame.bond(nBonds - 1).j == nBonds)
        #expect(frame.fieldLayer == 3 && frame.field?.count == grid * grid && frame.field?.first == 0x3333_3333)
        #expect(frame.colorIndex == 1 && frame.pos.count == 3 * rows)
    }

    @Test(arguments: [1, 2, 3, 4, 5, 6, 7, 9, 10, 11], chunkings)
    func theAliveSectionIsPaddedToAMultipleOfFour(rows: Int, chunk: Int?) {
        let stream = SyntheticStream(rows: rows)
        let alive = Array(stride(from: 0, to: rows, by: 2))
        let bonds = rows >= 4 ? [(0, 2), (1, 3)] : []
        var bytes = stream.preamble()
        bytes += stream.frame(step: 7, alive: alive, bonds: bonds)
        bytes += stream.frame(step: 8, alive: Array(0..<rows), layer: 2)
        let result = decode(bytes, chunk: chunk)
        #expect(result.error == nil)
        #expect(result.events.count == 3)
        guard case .frame(let first) = result.events[1], case .frame(let second) = result.events[2] else {
            Issue.record("expected two frames")
            return
        }
        #expect(first.step == 7 && first.nAlive == alive.count && first.nBonds == bonds.count)
        let expectedAlive: [UInt8] = (0..<rows).map { alive.contains($0) ? 1 : 0 }
        #expect(first.alive == expectedAlive)
        #expect(first.field == nil && second.fieldLayer == 2)
        let expectedField: [UInt32] = (0..<4).map { UInt32(0x300000 + $0) }
        #expect(second.field == expectedField)
        var expectedPos: [Float] = [], expectedOrient: [UInt16] = [], expectedRGB: [UInt32] = []
        for r in 0..<rows {
            expectedPos += [Float(r), Float(2 * r), Float(3 * r) + 0.5]
            expectedOrient += [UInt16(0x3C00 + r), UInt16(0xBC00 + r)]
            expectedRGB.append(UInt32(0x010203 + r))
        }
        for frame in [first, second] {
            #expect(frame.pos == expectedPos && frame.orient == expectedOrient && frame.rgb == expectedRGB)
            #expect(frame.colorIndex == 2)
        }
        var expectedBonds: [UInt32] = []
        for (i, j) in bonds { expectedBonds += [UInt32(i), UInt32(j)] }
        #expect(first.bonds == expectedBonds)
    }

    // MARK: Unknown records

    @Test(arguments: chunkings)
    func unknownRecordsAreSkippedByLength(chunk: Int?) throws {
        let fixture = try StreamFixture(kind: "3d")
        let records = fixture.records
        func record(_ tag: String, _ length: Int) -> [UInt8] {
            Array(tag.utf8) + le32(length) + (0..<length).map { UInt8(truncatingIfNeeded: $0 &* 7 &+ 3) }
        }
        // One before the first frame, a zero-length one between frames, a large one before the last, one at the end.
        var bytes = Array(fixture.bytes[0..<fixture.headerEnd])
        bytes += record("INFO", 12)
        bytes += fixture.bytes[records[0].offset..<(records[0].offset + 8 + records[0].length)]
        bytes += record("NOPE", 0)
        bytes += fixture.bytes[records[1].offset..<(records[1].offset + 8 + records[1].length)]
        bytes += record("BIGG", 5000)
        bytes += fixture.bytes[records[2].offset..<(records[2].offset + 8 + records[2].length)]
        bytes += record("TAIL", 4)

        let plain = try fixture.events()
        let result = decode(bytes, chunk: chunk)
        #expect(result.error == nil)
        let expected: [StreamEvent] = [
            plain[0], .unknownRecord(tag: "INFO", length: 12), plain[1], .unknownRecord(tag: "NOPE", length: 0), plain[2],
            .unknownRecord(tag: "BIGG", length: 5000), plain[3], .unknownRecord(tag: "TAIL", length: 4),
        ]
        #expect(result.events == expected)
    }

    @Test func aStreamMayEndAfterOnlyTheHeader() throws {
        let fixture = try StreamFixture(kind: "2d")
        let result = decode(Array(fixture.bytes[0..<fixture.headerEnd]))
        #expect(result.error == nil)
        #expect(result.events.count == 1)
    }

    // MARK: Refusals

    @Test func badMagicIsRefusedAsSoonAsItShows() throws {
        var decoder = FrameDecoder()
        #expect(throws: StreamError.badMagic) { _ = try decoder.feed(Data("Traceback (most recent call last):".utf8)) }
        // The decoder stays failed.
        #expect(throws: StreamError.badMagic) { _ = try decoder.feed(Data([0])) }
        #expect(throws: StreamError.badMagic) { try decoder.finish() }

        // Six bytes that are not the magic are enough; the decoder does not wait for the rest.
        var early = FrameDecoder()
        #expect(throws: StreamError.badMagic) { _ = try early.feed(Data([0x41, 0x4C, 0x53, 0x54, 0x52, 0x4E])) }

        let fixture = try StreamFixture(kind: "3d")
        for index in 0..<7 {
            let result = decode(patched(fixture.bytes, at: index, [0x58]), chunk: 7)
            #expect(result.error == .badMagic, "byte \(index) of the magic")
            #expect(result.events.isEmpty)
        }
    }

    @Test(arguments: [0, 2, 9, 255])
    func anUnknownMajorVersionIsRefused(version: Int) throws {
        let fixture = try StreamFixture(kind: "3d")
        for chunk in Self.chunkings {
            let result = decode(patched(fixture.bytes, at: 7, [UInt8(version)]), chunk: chunk)
            #expect(result.error == .unsupportedProtocol(version))
            #expect(result.events.isEmpty)
        }
    }

    @Test func badMagicNamesTheVersionOnlyWhenTheRestMatches() {
        // The version byte is judged only when the seven bytes before it are the magic.
        let result = decode(Array("ALSTRM".utf8) + [0, 2] + le32(4) + Array("{} ".utf8) + [0x20])
        #expect(result.error == .unsupportedProtocol(2))
    }

    @Test func theHeaderIsValidated() throws {
        let fixture = try StreamFixture(kind: "3d")
        let jsonStart = 12
        let json = String(decoding: fixture.bytes[jsonStart..<fixture.headerEnd], as: UTF8.self)

        /// The stream with its JSON text replaced (padded to a multiple of 4 with spaces).
        func stream(json text: String, jsonLength: Int? = nil) -> [UInt8] {
            var body = Array(text.utf8)
            while body.count % 4 != 0 { body.append(0x20) }
            return Array(fixture.bytes[0..<8]) + le32(jsonLength ?? body.count) + body + Array(fixture.bytes[fixture.headerEnd...])
        }
        func malformed(_ result: (events: [StreamEvent], error: StreamError?)) -> Bool {
            if case .malformedHeader = result.error { return result.events.isEmpty }
            return false
        }

        let broken: [(String, String)] = [
            ("not JSON", "{\"protocol\": 1,"),
            ("a missing key", json.replacingOccurrences(of: "\"NCH\"", with: "\"NCHX\"")),
            ("a wrong type", json.replacingOccurrences(of: "\"dims\":3", with: "\"dims\":\"3\"")),
            ("dims 4", json.replacingOccurrences(of: "\"dims\":3", with: "\"dims\":4")),
            ("protocol 2 under version-1 magic", json.replacingOccurrences(of: "\"protocol\":1", with: "\"protocol\":2")),
            ("N 0", json.replacingOccurrences(of: "\"N\":64", with: "\"N\":0")),
            ("N past the cap", json.replacingOccurrences(of: "\"N\":64", with: "\"N\":99999999")),
            ("FG 0", json.replacingOccurrences(of: "\"FG\":12", with: "\"FG\":0")),
            ("no colour modes", json.replacingOccurrences(of: "[\"tag\",\"energy\",\"clade\"]", with: "[]")),
            ("a negative L", json.replacingOccurrences(of: "\"L\":48.0", with: "\"L\":-48.0")),
        ]
        for (what, text) in broken {
            #expect(text != json || what == "not JSON", "the edit for \(what) changed the JSON")
            for chunk in Self.chunkings {
                #expect(malformed(decode(stream(json: text), chunk: chunk)), "\(what), chunk \(chunk.map(String.init) ?? "whole")")
            }
        }
        // json_len: not a multiple of 4, and past the 1 MiB cap.
        #expect(malformed(decode(patched(fixture.bytes, at: 8, le32(json.utf8.count - 1)))))
        #expect(malformed(decode(patched(fixture.bytes, at: 8, le32(1 << 22)))))
        #expect(malformed(decode(stream(json: json, jsonLength: 0))))
        // A key the version does not know is ignored.
        #expect(decode(stream(json: "{\"future_key\":[1,2]," + json.dropFirst())).error == nil)
    }

    @Test(arguments: [("a shorter", -4), ("a longer", 4), ("a much longer", 40_000)])
    func aFrameWithTheWrongLengthIsRefusedBeforeItsPayloadArrives(name: String, delta: Int) throws {
        let fixture = try StreamFixture(kind: "3d")
        let (offset, length) = fixture.records[1]  // a frame with a field section
        let bytes = patched(fixture.bytes, at: offset + 4, le32(length + delta))
        for chunk in Self.chunkings {
            let result = decode(bytes, chunk: chunk)
            #expect(result.error == .lengthMismatch(tag: "FRM1", declared: length + delta, expected: length), "\(name), chunk \(chunk.map(String.init) ?? "whole")")
            #expect(result.events.count == 2, "the header and the frame before it arrive first")
        }
        // The refusal does not wait for the payload: the record prefix and the 32-byte head are enough.
        var decoder = FrameDecoder()
        _ = try decoder.feed(Data(bytes[0..<(offset + 40)]))
        #expect(throws: StreamError.lengthMismatch(tag: "FRM1", declared: length + delta, expected: length)) {
            try decoder.finish()
        }
    }

    @Test func aFrameShorterThanItsHeadIsRefused() throws {
        let fixture = try StreamFixture(kind: "2d")
        let result = decode(patched(fixture.bytes, at: fixture.records[0].offset + 4, le32(8)))
        #expect(result.error == .lengthMismatch(tag: "FRM1", declared: 8, expected: 32))
    }

    @Test func aFrameThatContradictsTheHeaderIsRefused() throws {
        let fixture = try StreamFixture(kind: "3d")
        let (offset, _) = fixture.records[1]
        let payload = offset + 8
        let section = (alive: payload + 32, bonds: payload + 32 + 64 + 768 + 256 + 256)
        func refused(_ name: String, _ bytes: [UInt8]) {
            for chunk in Self.chunkings {
                let result = decode(bytes, chunk: chunk)
                if case .malformedFrame = result.error {} else {
                    Issue.record("\(name): expected malformedFrame, got \(String(describing: result.error)), chunk \(chunk.map(String.init) ?? "whole")")
                }
                #expect(result.events.count == 2, "\(name): only the header and the frame before it")
            }
        }
        refused("n_alive too large", patched(fixture.bytes, at: payload + 8, le32(8)))
        refused("an alive byte of 2", patched(fixture.bytes, at: section.alive, [2]))
        refused("an alive row not counted", patched(fixture.bytes, at: section.alive + 5, [1]))
        refused("color_index 3", patched(fixture.bytes, at: payload + 20, le32(3)))
        refused("a bond with i = j", patched(fixture.bytes, at: section.bonds, le32(4) + le32(4)))
        refused("a bond with i > j", patched(fixture.bytes, at: section.bonds, le32(9) + le32(2)))
        refused("a bond past N", patched(fixture.bytes, at: section.bonds, le32(1) + le32(64)))
        refused("a field layer past FZ", patched(fixture.bytes, at: payload + 12, le32(1 | 6 << 16)))
        // n_bonds past what N rows can hold: refused before the length is compared (a corrupt count must not
        // make the decoder wait for gigabytes).
        refused("an absurd n_bonds", patched(fixture.bytes, at: payload + 16, le32(1_000_000)))
    }

    @Test func reservedBytesAndUnknownFlagBitsAreIgnored() throws {
        let fixture = try StreamFixture(kind: "3d")
        let (offset, _) = fixture.records[0]
        let payload = offset + 8
        var bytes = patched(fixture.bytes, at: payload + 24, [1, 2, 3, 4, 5, 6, 7, 8])
        bytes = patched(bytes, at: payload + 12, le32(0b1100))  // reserved flag bits 2 and 3 set
        let result = decode(bytes)
        #expect(result.error == nil)
        guard case .frame(let frame) = result.events[1] else { Issue.record("no frame"); return }
        #expect(frame.flags == 0b1100 && !frame.hasField && !frame.hasDiagError)
    }

    // MARK: Truncation

    @Test(arguments: kinds)
    func endOfInputInsideARecordIsTruncation(kind: String) throws {
        let fixture = try StreamFixture(kind: kind)
        let complete = Set(fixture.boundaries)
        for cut in 0..<fixture.bytes.count {
            let result = decode(Array(fixture.bytes[0..<cut]), chunk: nil)
            if complete.contains(cut) {
                #expect(result.error == nil, "a stream cut at the end of a record (\(cut)) is complete")
            } else if case .truncated = result.error {
                continue
            } else {
                Issue.record("cut at \(cut): expected truncated, got \(String(describing: result.error))")
            }
        }
        // An empty stream is truncated too (the process died before it wrote anything).
        #expect(decode([]).error == .truncated("the stream ended before its header"))
        #expect(decode(Array(fixture.bytes[0..<5])).error == .truncated("the stream ended inside its header"))
    }

    @Test func truncationInsideASkippedRecordIsDetected() throws {
        let fixture = try StreamFixture(kind: "2d")
        var bytes = Array(fixture.bytes[0..<fixture.headerEnd]) + Array("JUNK".utf8) + le32(100) + [UInt8](repeating: 0, count: 60)
        #expect(decode(bytes).error == .truncated("the stream ended inside a JUNK record, 40 of 100 bytes short"))
        bytes += [UInt8](repeating: 0, count: 40)
        #expect(decode(bytes).error == nil)
    }

    // MARK: Errors

    @Test func anErrorIsReportedAfterTheEventsThatPrecedeIt() throws {
        let fixture = try StreamFixture(kind: "3d")
        let (offset, length) = fixture.records[2]
        // The first two frames, then a third with a bad length.
        let bytes = patched(Array(fixture.bytes[0..<(offset + 8 + length)]), at: offset + 4, le32(length + 4))
        let baseline = decode(bytes, chunk: 1)
        #expect(baseline.events.count == 3)
        #expect(baseline.error == .lengthMismatch(tag: "FRM1", declared: length + 4, expected: length))
        for chunk in Self.chunkings {
            let result = decode(bytes, chunk: chunk)
            #expect(result.events == baseline.events && result.error == baseline.error, "chunk \(chunk.map(String.init) ?? "whole")")
        }

        // One feed that holds the events and the error returns the events, then the error comes from the next call.
        var decoder = FrameDecoder()
        let events = try decoder.feed(Data(bytes))
        #expect(events.count == 3)
        #expect(throws: StreamError.self) { _ = try decoder.feed(Data()) }
        #expect(throws: StreamError.self) { try decoder.finish() }
    }

    // MARK: Binary16

    @Test func halfPrecisionConvertsExactly() {
        #expect(Half.float(0x0000) == 0 && Half.float(0x8000).sign == .minus && Half.float(0x8000) == 0)
        #expect(Half.float(0x3C00) == 1 && Half.float(0xBC00) == -1)
        #expect(Half.float(0x4248) == 3.140625)
        #expect(Half.float(0x7BFF) == 65504)
        #expect(Half.float(0x0001) == Float(sign: .plus, exponent: -24, significand: 1))
        #expect(Half.float(0x7C00) == .infinity && Half.float(0xFC00) == -.infinity)
        #expect(Half.float(0x7E00).isNaN && Half.float(0x7C01).isNaN)
        #if arch(arm64)  // Float16 exists on Apple silicon only: check all 65536 patterns against it
        let wrong = (UInt16.min...UInt16.max).filter { bits in
            let reference = Float(Float16(bitPattern: bits)), ours = Half.float(bits)
            return ours.isNaN ? !reference.isNaN : ours.bitPattern != reference.bitPattern
        }
        #expect(wrong.isEmpty, "patterns converted differently: \(wrong.prefix(5))")
        #endif
    }
}
