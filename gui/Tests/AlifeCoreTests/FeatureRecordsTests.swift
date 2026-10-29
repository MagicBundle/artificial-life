import Foundation
import Testing

@testable import AlifeCore

/// The records and header keys of the five feature layers (DESIGN §45): `STRC` and `FSOL` (sparse cell levels), `FSIG` (the focused
/// organism's voice), the bond flow of `FRMA` (`FLAG_BONDFLOW`), the `role` colour mode and the optional `play.*` keys. Every record is
/// built here from the wire contract (`stream_contract.md`, written for the observer PR), byte by byte; the hex strings are what the
/// Python encoders (`streamrec.encode_strc`, `encode_fsig`, `encode_frma`) wrote for the same inputs, so the layouts are pinned from both
/// sides.
@Suite struct FeatureRecordsTests {
    // MARK: Building records by hand

    private func u16(_ v: Int) -> [UInt8] { [UInt8(v & 0xFF), UInt8((v >> 8) & 0xFF)] }
    private func u64(_ v: UInt64) -> [UInt8] { (0..<8).map { UInt8((v >> (8 * UInt64($0))) & 0xFF) } }

    private func pad4(_ bytes: [UInt8]) -> [UInt8] { bytes + [UInt8](repeating: 0, count: (4 - bytes.count % 4) % 4) }

    /// tag, length, the 16-byte head (`step, seq, version, flags`) and `body`.
    private func record(_ tag: String, step: UInt64 = 7, seq: Int = 0, version: Int = 1, flags: Int = 0, body: [UInt8]) -> [UInt8] {
        let payload = u64(step) + le32(seq) + u16(version) + u16(flags) + body
        return Array(tag.utf8) + le32(payload.count) + payload
    }

    /// The body of `STRC` and `FSOL`: `u16 nx, ny, nz, flags; u32 count; count x u32 idx; count x u8 level`, padded to 4.
    private func cellsBody(nx: Int, ny: Int, nz: Int, flags: Int = 0, indices: [Int], levels: [Int], count: Int? = nil,
                           padding: [UInt8] = []) -> [UInt8] {
        var body = u16(nx) + u16(ny) + u16(nz) + u16(flags) + le32(count ?? indices.count)
        for i in indices { body += le32(i) }
        body += levels.map { UInt8($0) }
        let padded = pad4(body)
        return padding.isEmpty ? padded : Array(padded.dropLast(padding.count)) + padding
    }

    private func strc(nx: Int = 12, ny: Int = 12, nz: Int = 1, step: UInt64 = 7, seq: Int = 0, flags: Int = 0, bodyFlags: Int? = nil,
                      indices: [Int], levels: [Int], count: Int? = nil, version: Int = 1, tag: String = "STRC",
                      padding: [UInt8] = []) -> [UInt8] {
        record(tag, step: step, seq: seq, version: version, flags: flags,
               body: cellsBody(nx: nx, ny: ny, nz: nz, flags: bodyFlags ?? flags, indices: indices, levels: levels, count: count,
                               padding: padding))
    }

    /// `i64 uid; u8 W; u8 flags; u16 audible; W x u8 says; W x u8 hears`, padded to 4.
    private func fsig(uid: Int64 = 5, w: Int? = nil, flags: Int = 1, audible: Int = 0, says: [Int], hears: [Int], flags16: Int = 0,
                      padding: [UInt8] = []) -> [UInt8] {
        var body = u64(UInt64(bitPattern: uid)) + [UInt8(w ?? says.count), UInt8(flags)] + u16(audible)
        body += says.map { UInt8($0) } + hears.map { UInt8($0) }
        let padded = pad4(body)
        return record("FSIG", flags: flags16, body: padding.isEmpty ? padded : Array(padded.dropLast(padding.count)) + padding)
    }

    private func hex(_ bytes: [UInt8]) -> String { bytes.map { String(format: "%02x", $0) }.joined() }

    /// A stream: the preamble and header of the committed fixture of `kind` with `edit` applied to its `play` object, then `records`.
    private func stream(kind: String = "2d", edit: (inout [String: Any]) -> Void = { _ in }, _ records: [UInt8]...) throws -> [UInt8] {
        let fixture = try PlayFixture(kind: kind)
        var header = fixture.header
        var play = header["play"] as! [String: Any]
        edit(&play)
        header["play"] = play
        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var text = [UInt8](json)
        while text.count % 4 != 0 { text.append(0x20) }
        return Array("ALSTRM".utf8) + [0, 1] + le32(text.count) + text + records.flatMap { $0 }
    }

    /// The `play` object of a world with all the feature layers on (2D: FG 12, FZ 1).
    private func allOn(_ play: inout [String: Any], fz: Int = 1) {
        var records = play["records"] as! [String]
        records += ["STRC", "FSIG", "FSOL"]
        play["records"] = records
        var versions = play["versions"] as! [String: Int]
        for tag in ["STRC", "FSIG", "FSOL"] { versions[tag] = 1 }
        play["versions"] = versions
        play["structures"] = ["enabled": true, "cap": 4096, "cells": [fz, 12, 12], "cell": 4.0]
        play["comm"] = ["enabled": true, "w": 3, "range": 8.0, "control": "none"]
        play["niche"] = ["enabled": true, "farm": true, "cap": 4095, "cells": [fz, 12, 12], "cell": 4.0]
        play["bonds"] = ["flow": true, "scale": 1000.0]
    }

    private func records(_ bytes: [UInt8], chunk: Int? = nil) -> (records: [StreamRecord], error: StreamError?, events: [StreamEvent]) {
        let result = decode(bytes, chunk: chunk)
        let records = result.events.compactMap { event -> StreamRecord? in
            if case .record(let record) = event { return record }
            return nil
        }
        return (records, result.error, result.events)
    }

    private func isMalformed(_ error: StreamError?, _ tag: String, containing text: String) -> Bool {
        guard case .malformedRecord(let t, let why)? = error else { return false }
        return t == tag && why.contains(text)
    }

    // MARK: The header keys

    @Test func theOptionalPlayKeysAreDecodedWhenWritten() throws {
        let result = decode(try stream(edit: { allOn(&$0) }))
        #expect(result.error == nil)
        guard case .header(let header)? = result.events.first, let play = header.play else { Issue.record("no header"); return }
        #expect(play.structures == .init(enabled: true, farm: nil, cap: 4096, cells: [1, 12, 12], cell: 4))
        #expect(play.niche == .init(enabled: true, farm: true, cap: 4095, cells: [1, 12, 12], cell: 4))
        #expect(play.comm == .init(enabled: true, w: 3, range: 8, control: "none"))
        #expect(play.bonds == .init(flow: true, scale: 1000))
        #expect(play.records.contains("STRC") && play.records.contains("FSIG") && play.records.contains("FSOL"))
    }

    @Test(arguments: ["2d", "3d"]) func aWorldWithoutTheLayersHasNoneOfTheKeys(kind: String) throws {
        let result = decode(try stream(kind: kind))
        guard case .header(let header)? = result.events.first, let play = header.play else { Issue.record("no header"); return }
        #expect(result.error == nil)
        #expect(play.structures == nil && play.comm == nil && play.niche == nil && play.bonds == nil)
    }

    @Test func aGridThatIsNotTheFieldLatticeIsAMalformedHeader() throws {
        let cases: [(String, (inout [String: Any]) -> Void)] = [
            ("a wrong FG", { $0["structures"] = ["enabled": true, "cap": 4096, "cells": [1, 13, 13], "cell": 4.0] }),
            ("a wrong FZ", { $0["structures"] = ["enabled": true, "cap": 4096, "cells": [2, 12, 12], "cell": 4.0] }),
            ("two numbers", { $0["niche"] = ["enabled": true, "farm": true, "cap": 4095, "cells": [12, 12], "cell": 4.0] }),
            ("a zero cap", { $0["structures"] = ["enabled": true, "cap": 0, "cells": [1, 12, 12], "cell": 4.0] }),
            ("a zero cell", { $0["structures"] = ["enabled": true, "cap": 4096, "cells": [1, 12, 12], "cell": 0.0] }),
            ("nine channels", { $0["comm"] = ["enabled": true, "w": 9, "range": 8.0, "control": "none"] }),
            ("no channels", { $0["comm"] = ["enabled": true, "w": 0, "range": 8.0, "control": "none"] }),
            ("a zero flow scale", { $0["bonds"] = ["flow": true, "scale": 0.0] }),
        ]
        for (name, edit) in cases {
            let result = decode(try stream(edit: edit))
            guard case .malformedHeader? = result.error else { Issue.record("\(name): \(String(describing: result.error))"); continue }
            #expect(result.events.isEmpty, Comment(rawValue: name))
        }
    }

    // MARK: STRC and FSOL

    /// `encode_strc(step=7, seq=3, shape=(1,4,4), mat=, cap=4096)` with 4096 units at (x 2, y 1) and 100 at (x 0, y 3).
    private let pythonSTRC = "535452432800000007000000000000000300000001000000040004000100000002000000060000000c000000ff070000"

    @Test func theCellRecordIsTheBytesThePythonEncoderWrote() {
        let mine = strc(nx: 4, ny: 4, nz: 1, step: 7, seq: 3, indices: [6, 12], levels: [255, 7])
        #expect(hex(mine) == pythonSTRC)
    }

    @Test func aMaterialRecordDecodesToItsCells() throws {
        let bytes = try stream(strc(step: 40, seq: 2, indices: [3, 14, 15, 30], levels: [1, 128, 255, 7]))
        let (found, error, _) = records(bytes)
        #expect(error == nil && found.count == 1)
        guard case .strc(let cells)? = found.first else { Issue.record("not an STRC: \(found)"); return }
        #expect(cells.head.step == 40 && cells.head.seq == 2 && cells.head.version == 1 && cells.head.flags == 0)
        #expect((cells.nx, cells.ny, cells.nz) == (12, 12, 1) && cells.count == 4 && !cells.isTruncated)
        #expect(cells.indices == [3, 14, 15, 30] && cells.levels == [1, 128, 255, 7])
        let at = (0..<4).map { cells.cell($0) }
        #expect(at.map(\.x) == [3, 2, 3, 6] && at.map(\.y) == [0, 1, 1, 2] && at.allSatisfy { $0.z == 0 })
        #expect(found.first?.tag == "STRC" && found.first?.head.step == 40)
    }

    @Test func theLevelsCarryTheMaterialBackToItsUnits() {
        // level = 1 + 254 m // cap, so m is recovered to within one step of the level: (level - 1) cap / 254
        #expect(PlayCells.units(255, cap: 4096) == 4096)
        #expect(PlayCells.units(1, cap: 4096) == 0)
        #expect(abs(PlayCells.units(7, cap: 4096) - 96.75) < 0.01)
        #expect(PlayCells.alpha(255) == 1 && PlayCells.alpha(0) == 0)
    }

    @Test func aSoilRecordIsTheSameLayoutUnderItsOwnTag() throws {
        let (found, error, _) = records(try stream(edit: { allOn(&$0) }, strc(indices: [0, 143], levels: [9, 200], tag: "FSOL")))
        #expect(error == nil)
        guard case .fsol(let soil)? = found.first else { Issue.record("not an FSOL: \(found)"); return }
        #expect(soil.indices == [0, 143] && soil.levels == [9, 200] && found.first?.tag == "FSOL")
        #expect(soil.cell(1) == (11, 11, 0))
    }

    @Test func anEmptyRecordClearsTheLayer() throws {
        let (found, error, _) = records(try stream(strc(indices: [], levels: [])))
        #expect(error == nil)
        guard case .strc(let cells)? = found.first else { Issue.record("not an STRC"); return }
        #expect(cells.count == 0 && cells.columnMaximum() == [UInt8](repeating: 0, count: 144))
    }

    @Test func a3DRecordIndexesZMajorAndTheColumnMaximumFlattensIt() throws {
        // FZ 6 layers of 12 x 12: idx = (z * 12 + y) * 12 + x
        let a = (0 * 12 + 4) * 12 + 5, b = (3 * 12 + 4) * 12 + 5, c = (5 * 12 + 11) * 12 + 0
        let bytes = try stream(kind: "3d", edit: { allOn(&$0, fz: 6) }, strc(nz: 6, indices: [a, b, c], levels: [40, 90, 12]))
        let (found, error, _) = records(bytes)
        #expect(error == nil)
        guard case .strc(let cells)? = found.first else { Issue.record("not an STRC"); return }
        #expect(cells.cell(1) == (5, 4, 3) && cells.cell(2) == (0, 11, 5))
        let image = cells.columnMaximum()
        #expect(image.count == 144 && image[4 * 12 + 5] == 90 && image[11 * 12 + 0] == 12 && image.filter { $0 > 0 }.count == 2)
    }

    @Test func aMalformedCellRecordIsRefusedWithItsReason() throws {
        let cases: [(String, [UInt8], String)] = [
            ("a grid that is not the header's", strc(nx: 13, indices: [1], levels: [1]), "grid"),
            ("indices that do not ascend", strc(indices: [5, 5], levels: [1, 1]), "ascending"),
            ("indices that go down", strc(indices: [9, 4], levels: [1, 1]), "ascending"),
            ("an index outside the grid", strc(indices: [144], levels: [1]), "ascending"),
            ("a level of 0", strc(indices: [1, 2], levels: [4, 0]), "level 0"),
            ("body flags unlike the head's", strc(flags: 0, bodyFlags: 1, indices: [1], levels: [1]), "flags"),
            ("a truncated flag on a short list", strc(flags: 1, indices: [1], levels: [1]), "truncated"),
            ("a padding byte that is not zero", strc(indices: [1], levels: [1], padding: [0, 0, 7]), "padding"),
        ]
        for (name, bytes, reason) in cases {
            let error = decode(try stream(bytes), chunk: 17).error
            #expect(isMalformed(error, "STRC", containing: reason), Comment(rawValue: "\(name): \(String(describing: error))"))
        }
    }

    @Test func aCountOverTheCapIsRefusedBeforeItsPayloadIsWaitedFor() throws {
        // 65 537 cells: the record is refused from its 12-byte prefix alone
        let head = Array("STRC".utf8) + le32(16 + 12 + 5 * 65537 + 3) + u64(1) + le32(0) + u16(1) + u16(0)
        let body = u16(12) + u16(12) + u16(1) + u16(0) + le32(65537)
        let error = decode(try stream(head + body)).error
        #expect(isMalformed(error, "STRC", containing: "65537"))
    }

    @Test func aLengthThatDisagreesWithTheCountIsRefused() throws {
        var bytes = strc(indices: [1, 2], levels: [1, 2])
        bytes.replaceSubrange(4..<8, with: le32(Int(word(bytes, at: 4)) + 4))
        bytes += [0, 0, 0, 0]
        guard case .lengthMismatch(let tag, _, _)? = decode(try stream(bytes)).error else { Issue.record("not refused"); return }
        #expect(tag == "STRC")
    }

    private func word(_ bytes: [UInt8], at offset: Int) -> UInt32 {
        (0..<4).reduce(UInt32(0)) { $0 | UInt32(bytes[offset + $1]) << (8 * UInt32($1)) }
    }

    // MARK: FSIG

    /// `encode_fsig(step=7, seq=2, uid=-5, says=[0, 255, 17], hears=[1, 2, 3], audible=9)`
    private let pythonFSIG = "465349472400000007000000000000000200000001000000fbffffffffffffff0301090000ff110102030000"
    /// `encode_fsig(step=7, seq=2, uid=5, says=[200], hears=None)`
    private let pythonFSIGSilent = "465349472000000007000000000000000200000001000000050000000000000001000000c8000000"

    @Test func theVoiceRecordIsTheBytesThePythonEncoderWrote() {
        let loud = record("FSIG", seq: 2, body: u64(UInt64(bitPattern: -5)) + [3, 1] + u16(9) + [0, 255, 17, 1, 2, 3, 0, 0])
        #expect(hex(loud) == pythonFSIG)
        let silent = record("FSIG", seq: 2, body: u64(5) + [1, 0] + u16(0) + [200, 0, 0, 0])
        #expect(hex(silent) == pythonFSIGSilent)
    }

    @Test func aVoiceRecordDecodesBothHalves() throws {
        let bytes = try stream(edit: { allOn(&$0) }, fsig(uid: -5, flags: 1, audible: 9, says: [0, 255, 17], hears: [1, 2, 3]))
        let (found, error, _) = records(bytes)
        #expect(error == nil)
        guard case .fsig(let voice)? = found.first else { Issue.record("not an FSIG: \(found)"); return }
        #expect(voice.uid == -5 && voice.w == 3 && voice.heardValid && voice.audible == 9)
        #expect(voice.says == [0, 255, 17] && voice.hears == [1, 2, 3] && found.first?.tag == "FSIG")
    }

    @Test func aVoiceWithoutTheHeardHalfHasZeros() throws {
        let (found, error, _) = records(try stream(fsig(flags: 0, says: [200], hears: [0])))
        #expect(error == nil)
        guard case .fsig(let voice)? = found.first else { Issue.record("not an FSIG"); return }
        #expect(!voice.heardValid && voice.audible == 0 && voice.hears == [0] && voice.says == [200])
    }

    @Test(arguments: 1...8) func everyWidthIsWhatItsLengthSays(w: Int) throws {
        let bytes = try stream(fsig(w: w, flags: 1, audible: w, says: Array(0..<w), hears: Array(repeating: 9, count: w)))
        let (found, error, _) = records(bytes, chunk: 5)
        #expect(error == nil)
        guard case .fsig(let voice)? = found.first else { Issue.record("not an FSIG"); return }
        #expect(voice.w == w && voice.says == Array(0..<w).map(UInt8.init) && voice.hears.count == w)
        // the payload is the 16-byte head and 12 + 2W rounded up to 4
        let record = fsig(w: w, flags: 1, audible: w, says: Array(0..<w), hears: Array(repeating: 9, count: w))
        #expect(Int(word(record, at: 4)) == 16 + (12 + 2 * w + 3) / 4 * 4)
    }

    @Test func aMalformedVoiceRecordIsRefused() throws {
        let cases: [(String, [UInt8], String)] = [
            ("no channels", record("FSIG", body: u64(5) + [0, 1] + u16(0) + [0, 0, 0, 0]), "W 0"),
            ("nine channels", record("FSIG", body: u64(5) + [9, 1] + u16(0) + [UInt8](repeating: 0, count: 20)), "W 9"),
            ("a width unlike the header's", fsig(flags: 1, says: [1, 2], hears: [1, 2]), "comm.w"),
            ("hears without the flag", fsig(flags: 0, says: [1, 2, 3], hears: [0, 4, 0]), "heard half"),
            ("audible without the flag", fsig(flags: 0, audible: 2, says: [1, 2, 3], hears: [0, 0, 0]), "heard half"),
            ("a padding byte that is not zero", fsig(says: [1, 2, 3], hears: [1, 2, 3], padding: [0, 0, 3]), "padding"),
        ]
        for (name, bytes, reason) in cases {
            let error = decode(try stream(edit: { allOn(&$0) }, bytes), chunk: 13).error
            #expect(isMalformed(error, "FSIG", containing: reason), Comment(rawValue: "\(name): \(String(describing: error))"))
        }
    }

    // MARK: The bond flow of FRMA

    private func frma(flow: [Int8]?, flaggedWithoutFlow: Bool = false, flowPadding: [UInt8]? = nil) -> [UInt8] {
        // N 8, rows 1, 2, 5 alive, bonds (listed) 0-1 and 1-2: the inputs of the Python hex below
        var body: [UInt8] = []
        for v in [3, 2, 7, 5, 8, 0b100110] { body += le32(v) }
        for p in [[1, 2, 3], [4, 5, 6], [7, 8, 9]] as [[Float]] { for v in p { body += le32(Int(v.bitPattern)) } }
        body += [0x00, 0x38, 0x00, 0x34, 0x00, 0xBC, 0x00, 0x00, 0x00, 0x40, 0x00, 0xB8]
        for color in [0x112233, 0xAABBCC, 0xFF0000] { body += le32(color) }
        for uid: UInt64 in [10, 20, 30] { body += u64(uid) }
        for listed in [0, 1, 1, 2] { body += le32(listed) }
        var flags = 0
        if let flow {
            flags = 2
            var bytes = flow.map { UInt8(bitPattern: $0) }
            while bytes.count % 4 != 0 { bytes.append(0) }
            if let flowPadding { bytes = Array(bytes.dropLast(flowPadding.count)) + flowPadding }
            body += bytes
        } else if flaggedWithoutFlow {
            flags = 2
        }
        return record("FRMA", step: 9, seq: 1, flags: flags, body: body)
    }

    /// `encode_frma(... N 8, rows 1 2 5, bonds (1,2) and (2,5), color_index 7, span 5, flow=[-127, 64])` and the same without a flow.
    private let pythonFlow = "46524d4190000000090000000000000001000000010002000300000002000000070000000500000008000000260000000000803f0000004000004040000080400000a0400000c0400000e04000000041000010410038003400bc0000004000b833221100ccbbaa000000ff000a0000000000000014000000000000001e000000000000000000000001000000010000000200000081400000"
    private let pythonNoFlow = "46524d418c000000090000000000000001000000010000000300000002000000070000000500000008000000260000000000803f0000004000004040000080400000a0400000c0400000e04000000041000010410038003400bc0000004000b833221100ccbbaa000000ff000a0000000000000014000000000000001e0000000000000000000000010000000100000002000000"

    @Test func theFrameRecordIsTheBytesThePythonEncoderWrote() {
        #expect(hex(frma(flow: [-127, 64])) == pythonFlow)
        #expect(hex(frma(flow: nil)) == pythonNoFlow)
    }

    @Test func aFrameWithTheFlagCarriesOneFlowPerBond() throws {
        // the fixture's N is 64: the hand-built frame (N 8) belongs to `smallStream`'s header
        #expect(decode(try stream(frma(flow: [-127, 64]))).error != nil, "N 8 is not the fixture's 64")
        let (found, error, _) = records(try smallStream(frma(flow: [-127, 64])))
        #expect(error == nil)
        guard case .frma(let frame)? = found.first else { Issue.record("not an FRMA: \(found)"); return }
        #expect(frame.n == 3 && frame.nBonds == 2 && frame.rows == [1, 2, 5] && frame.uid == [10, 20, 30])
        #expect(frame.hasBondFlow && frame.flow == [-127, 64] && frame.colorIndex == 7 && frame.span == 5)
        #expect(frame.bonds == [0, 1, 1, 2] && frame.hasDiagError == false)
    }

    @Test func aFrameWithoutTheFlagHasNoFlow() throws {
        let (found, error, _) = records(try smallStream(frma(flow: nil)))
        #expect(error == nil)
        guard case .frma(let frame)? = found.first else { Issue.record("not an FRMA"); return }
        #expect(!frame.hasBondFlow && frame.flow == nil && frame.nBonds == 2)
    }

    @Test func theDecoderUsesTheFlagAndNotAnExactLength() throws {
        // the same payload is two records: the flag decides which length is right
        #expect(decode(try smallStream(frma(flow: [1, 2]))).error == nil)
        #expect(decode(try smallStream(frma(flow: nil))).error == nil)
        // flagged but without the section
        guard case .lengthMismatch(let tag, let declared, let expected)? = decode(try smallStream(frma(flow: nil, flaggedWithoutFlow: true))).error
        else { Issue.record("a flagged frame without a flow was accepted"); return }
        #expect(tag == "FRMA" && expected == declared + 4)
        // the flag may be set with no bonds: the section is empty
        #expect(decode(try smallStream(frmaWithoutBonds(flagged: true))).error == nil)
    }

    private func frmaWithoutBonds(flagged: Bool) -> [UInt8] {
        var body: [UInt8] = []
        for v in [1, 0, 0, 0, 8, 0b10] { body += le32(v) }
        for v in [1, 2, 3] as [Float] { body += le32(Int(v.bitPattern)) }
        body += [0, 0, 0, 0] + le32(0) + u64(7)
        return record("FRMA", flags: flagged ? 2 : 0, body: body)
    }

    @Test func aFlowThatBreaksTheContractIsRefused() throws {
        #expect(isMalformed(decode(try smallStream(frma(flow: [-128, 3]))).error, "FRMA", containing: "-128"))
        #expect(isMalformed(decode(try smallStream(frma(flow: [3, 4], flowPadding: [0, 0, 1, 0]))).error, "FRMA", containing: "padding"))
        // the flow is 127 at most: a full-scale value is fine at both ends
        #expect(decode(try smallStream(frma(flow: [127, -127]))).error == nil)
    }

    /// A header whose N is 8 (the hand-built frame's capacity), the bond flow and the `role` colour mode.
    private func smallStream(_ records: [UInt8]...) throws -> [UInt8] {
        let fixture = try PlayFixture(kind: "2d")
        var header = fixture.header
        header["N"] = 8
        header["color_modes"] = (header["color_modes"] as! [String]) + ["role"]
        var play = header["play"] as! [String: Any]
        allOn(&play)
        header["play"] = play
        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var text = [UInt8](json)
        while text.count % 4 != 0 { text.append(0x20) }
        return Array("ALSTRM".utf8) + [0, 1] + le32(text.count) + text + records.flatMap { $0 }
    }

    // MARK: The role colour mode

    @Test func theRoleModeIsFoundByNameWhereverTheHeaderPutsIt() throws {
        let result = decode(try smallStream(frma(flow: nil)))
        guard case .header(let header)? = result.events.first, case .record(.frma(let frame))? = result.events.dropFirst().first else {
            Issue.record("no header or frame"); return
        }
        #expect(header.colorModes.last == "role" && header.colorModes.firstIndex(of: "role") == 7)
        #expect(header.colorMode(index: frame.colorIndex) == "role")
        #expect(RoleColours.isRoleMode(header.colorMode(index: frame.colorIndex)))
        // a world without the bonds feature does not list it; nothing breaks and the legend stays away
        let plain = decode(try stream())
        guard case .header(let bare)? = plain.events.first else { Issue.record("no header"); return }
        #expect(!bare.colorModes.contains("role") && !RoleColours.isRoleMode(bare.colorModes.last))
    }

    @Test func theRoleLegendIsTheServersColours() {
        #expect(RoleColours.entries.map(\.name) == ["Generalist", "Forager", "Guard", "Germ"])
        #expect(RoleColours.entries.map(\.rgb) == [0xBEBEBE, 0xF0E442, 0x0072B2, 0xCC79A7])
    }

    // MARK: Unknown tags and versions are still skipped

    @Test(arguments: [nil, 1, 3, 29]) func anUnknownTagIsSkippedByItsLength(chunk: Int?) throws {
        let bytes = try stream(strc(indices: [1], levels: [9], tag: "FARS"), strc(indices: [2], levels: [9], tag: "STRC"),
                               record("ZZZZ", body: [1, 2, 3, 4]))
        let result = decode(bytes, chunk: chunk)
        #expect(result.error == nil)
        let tags = result.events.compactMap { event -> String? in
            switch event {
            case .record(let r): "record " + r.tag
            case .unknownRecord(let tag, _): "skip " + tag
            default: nil
            }
        }
        #expect(tags == ["skip FARS", "record STRC", "skip ZZZZ"])
    }

    @Test(arguments: ["STRC", "FSOL", "FSIG"]) func aNewerVersionOfAFeatureTagIsSkippedAndCounted(tag: String) throws {
        var decoder = FrameDecoder()
        let bytes = try stream(strc(indices: [1], levels: [9], version: 2, tag: tag))
        let events = try decoder.feed(Data(bytes))
        try decoder.finish()
        #expect(events.contains { if case .unknownRecord(let t, _) = $0 { t == tag } else { false } })
        #expect(decoder.skippedVersions == 1)
    }

    // MARK: The store

    @Test func theStoreKeepsTheNewestOfEachLayerAndTheVoiceOfTheFollowedOne() throws {
        let fixture = try PlayFixture(kind: "2d")
        var store = WorldStore()
        store.reset(header: nil)
        let events = try fixture.events()
        guard case .header(let header)? = events.first else { Issue.record("no header"); return }
        store.reset(header: header)
        guard let focusRecord = events.compactMap({ event -> PlayFocus? in
            if case .record(.focs(let f)) = event, f.isAlive { return f }
            return nil
        }).first else { Issue.record("no live FOCS in the fixture"); return }
        #expect(store.apply(.focs(focusRecord)) == .focus)

        let head = RecordHead(step: 100, seq: 0, version: 1, flags: 0)
        let first = PlayCells(head: head, nx: 12, ny: 12, nz: 1, indices: [1], levels: [5])
        let second = PlayCells(head: head, nx: 12, ny: 12, nz: 1, indices: [], levels: [])
        #expect(store.apply(.strc(first)) == .material && store.material == first && store.materialSerial == 1)
        #expect(store.apply(.strc(second)) == .material && store.material?.count == 0 && store.materialSerial == 2)
        #expect(store.apply(.fsol(first)) == .soil && store.soil == first && store.soilSerial == 1)

        let mine = PlayVoice(head: head, uid: focusRecord.uid, w: 2, heardValid: true, audible: 1, says: [1, 2], hears: [3, 4])
        let other = PlayVoice(head: head, uid: focusRecord.uid &+ 1, w: 2, heardValid: false, audible: 0, says: [9, 9], hears: [0, 0])
        #expect(store.apply(.fsig(mine)) == .voice && store.focusVoice == mine)
        store.apply(.fsig(other))
        #expect(store.voice == other && store.focusVoice == nil, "the voice of another organism is not shown")
        store.apply(.fsig(mine))
        var dead = focusRecord
        dead.flags &= ~1
        store.apply(.focs(dead))
        #expect(store.focusVoice == nil, "an organism that is not alive has no voice")
        #expect(store.carries("STRC") == header.play?.records.contains("STRC"))
    }
}
