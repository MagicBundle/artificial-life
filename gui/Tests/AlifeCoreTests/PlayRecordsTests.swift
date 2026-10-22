import CryptoKit
import Foundation
import Testing

@testable import AlifeCore

/// The play records against DESIGN §33.1: the committed fixtures decoded and checked against the sidecars of the
/// independent Python decoder, chunking, truncation, and every refusal the format asks for.
@Suite struct PlayRecordsTests {
    static let chunkings: [Int?] = [1, 7, 4096, nil]

    // MARK: Fixtures

    @Test(arguments: PlayFixture.kinds)
    func fixtureMatchesItsSidecar(kind: String) throws {
        let fixture = try PlayFixture(kind: kind)
        let events = try fixture.events()
        #expect(fixture.bytes.count == fixture.sidecar["bytes"] as? Int, "the sidecar describes this file")
        #expect(fixture.sidecar["kind"] as? String == kind)

        guard case .header(let header) = events[0] else { Issue.record("the first event is not the header"); return }
        #expect(header.capacity == fixture.header["N"] as? Int)
        #expect(header.dims == (kind == "2d" ? 2 : 3))
        #expect(header.mode == "simulate" && header.source == "preset" && header.every == 1 && !header.fields)
        #expect(header.colorModes == ["tag", "energy", "clade", "house", "kin", "archetype", "intent"])
        let play = try #require(header.play)
        let expectedPlay = try #require(fixture.header["play"])
        #expect(sameJSON(try normalisedJSON(play), strippingNulls(expectedPlay)), "the play header")

        // The records and the skipped ones interleave in stream order.
        var expected: [(offset: Int, event: String)] = fixture.records.map { ($0["offset"] as! Int, $0["tag"] as! String) }
        expected += fixture.skipped.map { ($0["offset"] as! Int, "skip " + ($0["tag"] as! String)) }
        expected.sort { $0.offset < $1.offset }
        let got: [String] = events.dropFirst().map {
            switch $0 {
            case .record(let r): r.tag
            case .unknownRecord(let tag, _): "skip " + tag
            default: "?"
            }
        }
        #expect(got == expected.map(\.event))

        let records = events.compactMap { event -> StreamRecord? in if case .record(let r) = event { r } else { nil } }
        #expect(records.count == fixture.records.count)
        for (index, (record, sidecar)) in zip(records, fixture.records).enumerated() {
            try check(record, against: sidecar, bytes: fixture, label: "\(kind) record \(index) \(record.tag)")
        }
    }

    private func check(_ record: StreamRecord, against sidecar: [String: Any], bytes fixture: PlayFixture, label: String)
        throws
    {
        #expect(record.tag == sidecar["tag"] as? String, "\(label) tag")
        let head = record.head
        #expect(head.step == UInt64(sidecar["step"] as! Int), "\(label) step")
        #expect(head.seq == UInt32(sidecar["seq"] as! Int), "\(label) seq")
        #expect(head.version == UInt16(sidecar["version"] as! Int) && head.version == 1, "\(label) version")
        #expect(head.flags == UInt16(sidecar["flags"] as! Int), "\(label) flags")
        let f = sidecar["fields"] as! [String: Any]

        switch record {
        case .frma(let frame):
            #expect(frame.n == f["n"] as? Int && frame.nBonds == f["n_bonds"] as? Int, "\(label) counts")
            #expect(frame.colorIndex == f["color_index"] as? Int && frame.span == f["span"] as? Int)
            #expect(frame.capacity == f["N"] as? Int)
            #expect(frame.rows.map(Int.init) == f["rows"] as? [Int], "\(label) rows")
            let bonds = f["bonds"] as! [[Int]]
            #expect((0..<frame.nBonds).map { [frame.bond($0).i, frame.bond($0).j] } == bonds, "\(label) bonds")
            for agent in f["agents"] as! [[String: Any]] {
                let i = agent["index"] as! Int
                #expect(Int(frame.rows[i]) == agent["row"] as? Int)
                let p = frame.position(i)
                #expect([Double(p.x), Double(p.y), Double(p.z)] == agent["pos"] as? [Double], "\(label) \(i) position")
                #expect(Double(frame.yaw(i)) == agent["yaw"] as? Double, "\(label) \(i) yaw")
                #expect(Double(frame.pitch(i)) == agent["pitch"] as? Double, "\(label) \(i) pitch")
                #expect(Int(frame.rgb[i]) == agent["rgb"] as? Int, "\(label) \(i) colour")
                #expect(frame.uid[i] == Int64(agent["uid"] as! Int), "\(label) \(i) uid")
            }
            #expect(frame.hasDiagError == (head.flags & 1 != 0))
        case .idnt(let r):
            #expect(Int(r.snapshotStep) == f["snapshot_step"] as? Int && r.rows.count == f["n_rows"] as? Int)
            #expect(sameJSON(try normalisedJSON(r.tables), strippingNulls(f["json"]!)), "\(label) tables")
            let rows = f["rows"] as! [[String: Int]]
            for (row, expected) in zip(r.rows, rows) {
                #expect(row.uid == Int64(expected["uid"]!) && Int(row.clade) == expected["clade"])
                #expect(Int(row.house) == expected["house"] && Int(row.kin) == expected["kin"])
                #expect(Int(row.colony) == expected["colony"] && Int(row.flags) == expected["flags"])
            }
        case .intn(let r):
            #expect(Int(r.labelsStep) == f["labels_step"] as? Int && Int(r.intentVersion) == f["intent_version"] as? Int)
            let rows = f["rows"] as! [[String: Int]]
            #expect(r.rows.count == rows.count)
            for (row, expected) in zip(r.rows, rows) {
                #expect(row.uid == Int64(expected["uid"]!) && Int(row.archPrimary) == expected["arch_primary"])
                #expect(Int(row.archSecondary) == expected["arch_secondary"] && Int(row.intent) == expected["intent"])
                #expect(Int(row.flags) == expected["flags"])
            }
        case .terr(let t):
            #expect(t.nx == f["nx"] as? Int && t.ny == f["ny"] as? Int)
            #expect(Int(t.snapshotStep) == f["snapshot_step"] as? Int)
            #expect((f["json"] as! [String: Any])["ids"] as? [Int] == t.ids)
            #expect(t.owner.reduce(0) { $0 + Int($1) } == f["owner_sum"] as? Int)
            #expect(t.purity.reduce(0) { $0 + Int($1) } == f["purity_sum"] as? Int)
            #expect(Array(t.owner[(3 * t.nx)..<(4 * t.nx)]).map(Int.init) == f["owner_row3"] as? [Int])
        case .strm(let s):
            #expect(Int(s.tick) == f["tick"] as? Int && s.storms.storms.count == f["n_storms"] as? Int)
            #expect(sameJSON(try normalisedJSON(s.storms), strippingNulls(f["json"]!)), "\(label) storms")
        case .map1(let m):
            #expect(m.n == f["n"] as? Int && m.nw == f["nw"] as? Int)
            // The sidecar holds the float32 values as binary64 numbers; JSON parsing may differ in the last digit.
            func close(_ a: Float, _ key: String) -> Bool { abs(Double(a) - (f[key] as! Double)) < 1e-12 }
            #expect(close(m.f0, "f0") && close(m.pRef, "p_ref") && close(m.dcRef, "dc_ref"), "\(label) refs")
            #expect(close(m.uRef, "u_ref") && close(m.zetaRef, "zeta_ref"), "\(label) refs")
            let samples = f["samples"] as! [String: [Int]]
            let cy = [0, 1, 17, 159], cx = [0, 5, 100, 159]
            #expect(zip(cy, cx).map { Int(m.p[$0 * m.n + $1]) } == samples["P"], "\(label) P samples")
            #expect(zip(cy, cx).map { Int(m.dc[$0 * m.n + $1]) } == samples["Dc"], "\(label) Dc samples")
            #expect(zip(cy, cx).map { Int(m.zeta[$0 * m.n + $1]) } == samples["zeta"], "\(label) zeta samples")
            #expect(zip(cy, cx).map { Int(m.ts[$0 * m.n + $1]) } == samples["Ts"], "\(label) Ts samples")
            #expect((0..<4).map { Int(m.winds[$0 * m.nw * m.nw + 3 * m.nw + 7]) } == samples["winds"], "\(label) winds")
        case .evnt(let e):
            #expect(e.events.count == f["count"] as? Int && e.isBacklog == f["backlog"] as? Bool)
            #expect(sameJSON(try normalisedJSON(e.events), strippingNulls(f["json"]!)), "\(label) events")
        case .envs(let e):
            #expect(sameJSON(try normalisedJSON(e.environment), strippingNulls(f["json"]!)), "\(label) environment")
        case .alrt(let b):
            #expect(b.alerts.count == f["count"] as? Int && b.isBacklog == f["backlog"] as? Bool)
            #expect(sameJSON(try normalisedJSON(b.alerts), strippingNulls(f["json"]!)), "\(label) alerts")
        case .rprt(let r):
            #expect(r.chapter == f["chapter"] as? Int && r.mode == f["mode"] as? Int, "\(label) head")
            #expect(r.isBacklog == f["backlog"] as? Bool && r.isFinal == f["final"] as? Bool, "\(label) flags")
            #expect(sameJSON(try normalisedJSON(r.report), strippingNulls(f["json"]!)), "\(label) report")
        case .plan(let p):
            #expect(p.isReply == f["reply"] as? Bool, "\(label) reply flag")
            #expect(sameJSON(try normalisedJSON(p.plan), strippingNulls(f["json"]!)), "\(label) plan")
        case .clck(let c):
            #expect(Double(c.rate) == f["rate"] as? Double && Double(c.achieved) == f["achieved"] as? Double)
            #expect(Double(c.stepMs) == f["step_ms"] as? Double && Double(c.frameMs) == f["frame_ms"] as? Double)
            #expect(Int(c.lateSteps) == f["late_steps"] as? Int && Int(c.saveStallSteps) == f["save_stall_steps"] as? Int)
            #expect(Int(c.reason.rawValue) == f["reason"] as? Int && (c.saving ? 1 : 0) == f["saving"] as? Int)
            #expect(Int(c.flags) == f["flags"] as? Int && Int(c.fe) == f["fe"] as? Int)
            #expect(Int(c.chapter) == f["chapter"] as? Int && Int(c.inChapter) == f["in_chapter"] as? Int)
            #expect(Int(c.chapterLen) == f["chapter_len"] as? Int)
        case .focs(let c):
            #expect(c.uid == Int64(f["uid"] as! Int) && Int(c.row) == f["row"] as? Int && Int(c.flags) == f["flags"] as? Int)
            #expect([c.x, c.y, c.z, c.vx, c.vy, c.vz].map(Double.init) == ["x", "y", "z", "vx", "vy", "vz"].map { f[$0] as! Double })
            #expect(Int(c.energy) == f["energy"] as? Int && Int(c.age) == f["age"] as? Int)
            #expect(Int(c.generation) == f["generation"] as? Int && c.parentUid == Int64(f["parent_uid"] as! Int))
            #expect([c.nBonds, c.house, c.kin, c.colony].map(Int.init) == ["n_bonds", "house", "kin", "colony"].map { f[$0] as! Int })
            #expect([c.archPrimary, c.archSecondary, c.intent, c.labelFlags].map(Int.init)
                == ["arch_primary", "arch_secondary", "intent", "label_flags"].map { f[$0] as! Int })
            #expect(c.isAlive && c.handedOff && c.isFollowed)
        }

        // Every array section byte for byte, padding included; the JSON sections by value above.
        let digests = sidecar["sha256"] as! [String: String]
        let sections = playSectionBytes(record)
        for (name, data) in sections {
            #expect(sha256(data) == digests[name], "\(label) \(name) section")
        }
        #expect(Set(digests.keys).subtracting(sections.keys).isSubset(of: ["json"]), "\(label) sections")
    }

    @Test func theDecodedFramesHaveTheirFixtureShape() throws {
        let fixture = try PlayFixture(kind: "3d")
        let frames = try fixture.events().compactMap { event -> PlayFrame? in
            if case .record(.frma(let f)) = event { f } else { nil }
        }
        #expect(frames.map(\.step) == [100, 104] && frames.map(\.head.seq) == [0, 1])
        #expect(frames[0].rows == [0, 1, 2, 3, 4, 10, 21] && frames[1].rows == [0, 1, 2, 3, 4, 5, 21])
        #expect(frames[0].uid[4] != frames[1].uid[4], "row 4 is reused by a newborn")
        #expect(frames[0].uid.max()! > Int64(UInt32.max), "uids are 64-bit")
        #expect(frames[0].bonds == [0, 1, 2, 5, 2, 3], "row 2's bonds stay in slot order, as listed-row indices")
        #expect(abs(frames[0].position(0).x - frames[0].position(1).x) > 40, "bond (0, 1) crosses the seam")
        #expect(frames[1].hasDiagError && !frames[0].hasDiagError)
        #expect(frames[0].rgb[2] == 0xFF0000, "the hurt agent is red")
    }

    @Test(arguments: PlayFixture.kinds)
    func twoDStreamsHaveFlatZAndNoPitch(kind: String) throws {
        guard kind == "2d" else { return }
        for case .record(.frma(let frame)) in try PlayFixture(kind: kind).events() {
            for i in 0..<frame.n { #expect(frame.position(i).z == 0 && frame.pitch(i) == 0) }
        }
    }

    @Test func aWorldWithoutWeatherSendsNoMapOrStorms() throws {
        let fixture = try PlayFixture(kind: "2d")
        let header = try #require(try fixture.events().compactMap { event -> StreamHeader? in
            if case .header(let h) = event { h } else { nil }
        }.first)
        #expect(header.play?.weather.enabled == false)
        #expect(header.play?.records == ["FRMA", "IDNT", "INTN", "TERR", "EVNT", "ENVS", "CLCK", "FOCS"])
        #expect(!(try fixture.events().contains { if case .record(let r) = $0 { r.tag == "MAP1" || r.tag == "STRM" } else { false } }))
    }

    @Test func theN70FixtureHasThreeBitmapWordsAndTheLastIsPartial() throws {
        let fixture = try PlayFixture(kind: "3d_n70")
        let frames = try fixture.events().compactMap { event -> PlayFrame? in
            if case .record(.frma(let f)) = event { f } else { nil }
        }
        #expect(frames.count == 2 && frames.allSatisfy { $0.capacity == 70 && $0.bitmap.count == 3 })
        #expect(frames[0].rows == [0, 31, 32, 63, 64, 69] && frames[1].n == 70)
        let first: [UInt32] = [0x8000_0001, 0x8000_0001, 0x21]  // rows 0, 31 | 32, 63 | 64, 69
        #expect(frames[0].bitmap == first)
        #expect(frames[1].bitmap == [UInt32.max, UInt32.max, 0x3F], "bits 70 to 95 stay zero")
        // Bonds across the word boundaries are listed-row indices.
        #expect(frames[0].bonds == [1, 2, 3, 4, 0, 5])
        for frame in frames {
            for row in 0..<70 {
                let expected = frame.rows.firstIndex(of: UInt32(row))
                #expect(frame.listedIndex(ofRow: row) == expected, "row \(row)")
                #expect(frame.isAlive(row: row) == (expected != nil))
            }
            #expect(frame.listedIndex(ofRow: 70) == nil && frame.listedIndex(ofRow: -1) == nil)
        }
    }

    // MARK: Chunking and truncation

    @Test(arguments: PlayFixture.kinds, chunkings)
    func anyChunkingGivesTheSameEvents(kind: String, chunk: Int?) throws {
        let fixture = try PlayFixture(kind: kind)
        let whole = try fixture.events()
        let result = decode(fixture.bytes, chunk: chunk)
        #expect(result.error == nil)
        #expect(result.events == whole, "chunk \(chunk.map(String.init) ?? "whole")")
    }

    @Test func eventsArriveWhenTheirLastByteDoes() throws {
        let fixture = try PlayFixture(kind: "2d")
        var decoder = FrameDecoder()
        var seen: [Int] = []
        for (index, byte) in fixture.bytes.enumerated() {
            for _ in try decoder.feed(Data([byte])) { seen.append(index + 1) }
        }
        try decoder.finish()
        #expect(seen == fixture.boundaries, "the header and each record complete at the end of their bytes")
    }

    @Test(arguments: ["2d", "3d_n70"])
    func aTruncationAtAnyByteIsRefusedExceptBetweenRecords(kind: String) throws {
        let fixture = try PlayFixture(kind: kind)
        let boundaries = Set(fixture.boundaries)
        for cut in 0..<fixture.bytes.count {
            var decoder = FrameDecoder()
            var failure: StreamError?
            do {
                _ = try decoder.feed(Data(fixture.bytes[0..<cut]))
                try decoder.finish()
            } catch let error as StreamError {
                failure = error
            }
            if boundaries.contains(cut) {
                #expect(failure == nil, "a stream cut after \(cut) bytes is complete")
            } else {
                guard case .truncated? = failure else { Issue.record("cut \(cut): \(String(describing: failure))"); return }
            }
        }
    }

    @Test func aTruncationInTheLargeFixtureIsRefusedAtSampledBytes() throws {
        let fixture = try PlayFixture(kind: "3d")
        let boundaries = Set(fixture.boundaries)
        var cuts = Set(stride(from: 1, to: fixture.bytes.count, by: 997))
        for b in boundaries { cuts.formUnion([b - 1, b + 1, b + 7, b + 8, b + 23, b + 24]) }
        for cut in cuts.sorted() where cut > 0 && cut < fixture.bytes.count {
            var decoder = FrameDecoder()
            var failure: StreamError?
            do {
                _ = try decoder.feed(Data(fixture.bytes[0..<cut]))
                try decoder.finish()
            } catch let error as StreamError {
                failure = error
            }
            if boundaries.contains(cut) {
                #expect(failure == nil, "cut \(cut)")
            } else {
                guard case .truncated? = failure else { Issue.record("cut \(cut): \(String(describing: failure))"); return }
            }
        }
    }

    // MARK: Unknown tags and versions

    @Test func theUnknownTagOfTheFixtureIsSkippedByLength() throws {
        let fixture = try PlayFixture(kind: "3d")
        let skipped = try #require(fixture.skipped.first)
        #expect(skipped["tag"] as? String == "ZZZZ")
        let events = try fixture.events()
        #expect(events.contains(.unknownRecord(tag: "ZZZZ", length: 28)), "12 bytes after the 16-byte head")
    }

    @Test(arguments: chunkings)
    func aReaderThatDoesNotKnowThePlayTagsPassesThemAllOn(chunk: Int?) throws {
        // The tags renamed to ones no decoder knows: the header with its `play` key is accepted, every record is skipped
        // by its length, and nothing else changes (what a client that predates the records does with a play stream).
        let fixture = try PlayFixture(kind: "3d")
        var bytes = fixture.bytes
        for record in fixture.records {
            bytes = patched(bytes, at: record["offset"] as! Int, Array("Q\((record["tag"] as! String).dropFirst())".utf8))
        }
        let result = decode(bytes, chunk: chunk)
        #expect(result.error == nil)
        guard case .header(let header)? = result.events.first else { Issue.record("no header"); return }
        #expect(header.play != nil)
        let skipped = result.events.dropFirst().map { event -> String in
            if case .unknownRecord(let tag, let length) = event { "\(tag):\(length)" } else { "?" }
        }
        let expected = fixture.allSpans.map { span -> String in
            let tag = String(decoding: bytes[span.offset..<(span.offset + 4)], as: UTF8.self)
            return "\(tag):\(span.length)"
        }
        #expect(skipped == expected)
    }

    @Test(arguments: ["FRMA", "STRM", "CLCK", "FOCS", "MAP1", "IDNT"], chunkings)
    func aKnownTagOfAnotherVersionIsSkippedAndCounted(tag: String, chunk: Int?) throws {
        let fixture = try PlayFixture(kind: "3d")
        let offset = try fixture.offset(of: tag)
        let length = Int(fixture.le32(at: offset + 4))
        let bytes = patched(fixture.bytes, at: offset + 8 + 12, [2, 0])
        let result = decodeWithDecoder(bytes, chunk: chunk)
        #expect(result.error == nil)
        #expect(result.events.contains(.unknownRecord(tag: tag, length: length)))
        #expect(result.decoder.skippedVersions == 1)
        let plain = try fixture.events()
        #expect(result.events.count == plain.count, "the other records are untouched")
        let kept = result.events.compactMap { event -> String? in if case .record(let r) = event { r.tag } else { nil } }
        #expect(kept.count == fixture.records.count - 1)
    }

    @Test func aVersionSkipNeedsNoValidPayload() throws {
        // A record of a version this decoder does not know may have any content: only its length is used.
        let fixture = try PlayFixture(kind: "2d")
        var bytes = Array(fixture.bytes[0..<fixture.headerEnd])
        let payload = [UInt8](repeating: 0xAB, count: 48)
        bytes += Array("FRMA".utf8) + le32(payload.count)
            + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0] + payload[16...]
        let result = decodeWithDecoder(bytes, chunk: 5)
        #expect(result.error == nil && result.decoder.skippedVersions == 1)
        #expect(result.events.last == .unknownRecord(tag: "FRMA", length: 48))
    }

    // MARK: The header

    @Test func theHeaderDecodesWithAndWithoutThePlayKey() throws {
        let plain = try StreamFixture(kind: "3d")
        for case .header(let header) in try plain.events() { #expect(header.play == nil) }
        for kind in PlayFixture.kinds {
            for case .header(let header) in try PlayFixture(kind: kind).events() {
                let play = try #require(header.play)
                #expect(play.versions == Dictionary(uniqueKeysWithValues: play.records.map { ($0, 1) }))
                #expect(play.baseRate == 14.4 && play.rates == [0.25, 0.5, 1, 2] && play.tick == 25)
                #expect(play.snapshotEvery == 250 && play.chapter == 6000 && play.frameHz == 30)
                #expect(play.cast == PlayHeader.Cast(kin: 16, house: 16, colony: 16) && play.energyCap == 16384)
                #expect(play.events["14"] == "focus_handoff" && play.events["21"] == "power_ended" && play.events.count == 21)
                #expect(play.labels.intentVersion == 3 && play.labels.intent[6] == "Tethering")
                #expect(play.labels.arch.count == 11 && play.labels.arch[8...] == ["Drifter", "Floater", "Sinker"])
                #expect(play.labels.arch[1] == (kind == "3d" ? "Grazer" : "Gatherer"), "the weather fixture has the storm names")
                #expect(play.labels.gate?.keyFound == (kind == "3d_n70"), "only the made-up fixture has a verdict row")
                #expect(play.weather.enabled == (kind == "3d"))
                #expect((play.alerts != nil) == (kind == "3d_g3b"), "only the G3b fixture has the G3b header keys")
                if kind == "3d_g3b" {
                    #expect(play.planet == PlayHeader.Planet(source: "runs/example:500000", houses: "exact"))
                } else if kind == "3d" {
                    #expect(play.weather.cells == 160 && play.weather.coarse == 40 && play.weather.nameZeta == 0.5)
                    #expect(play.weather.refs == PlayHeader.Weather.Refs(p: 0.004, dc: 0.5, u: 0.1, zeta: 0.025))
                    #expect(play.planet == PlayHeader.Planet(source: "runs/example:500000", houses: "exact"))
                } else {
                    #expect(play.planet == PlayHeader.Planet(source: nil, houses: "none"))
                }
            }
        }
    }

    /// The fixture with its header object edited by `edit`, the records untouched.
    private func stream(_ fixture: PlayFixture, editingHeader edit: (inout [String: Any]) -> Void) throws -> [UInt8] {
        var header = fixture.header
        edit(&header)
        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var text = [UInt8](json)
        while text.count % 4 != 0 { text.append(0x20) }
        return Array("ALSTRM".utf8) + [0, 1] + le32(text.count) + text + Array(fixture.bytes[fixture.headerEnd...])
    }

    @Test func aBadPlayKeyIsAMalformedHeader() throws {
        let fixture = try PlayFixture(kind: "3d")
        func play(_ edit: @escaping (inout [String: Any]) -> Void) -> (inout [String: Any]) -> Void {
            { header in
                var p = header["play"] as! [String: Any]
                edit(&p)
                header["play"] = p
            }
        }
        let edits: [(String, (inout [String: Any]) -> Void)] = [
            ("base_rate", play { $0["base_rate"] = 0 }),
            ("a rate", play { $0["rates"] = [1, -2] }),
            ("tick", play { $0["tick"] = 0 }),
            ("a snapshot cadence of zero", play { $0["snapshot_every"] = 0 }),
            ("energy_cap", play { $0["energy_cap"] = 0 }),
            ("weather cells", play { var w = $0["weather"] as! [String: Any]; w["cells"] = 161; $0["weather"] = w }),
            ("a duplicate tag", play { $0["records"] = ["FRMA", "FRMA"] }),
            ("a missing key", play { $0["labels"] = nil }),
            ("the wrong type", play { $0["tick"] = "25" }),
        ]
        for (name, edit) in edits {
            let result = decode(try stream(fixture, editingHeader: edit))
            guard case .malformedHeader? = result.error else { Issue.record("\(name): \(String(describing: result.error))"); continue }
            #expect(result.events.isEmpty, Comment(rawValue: name))
        }
        // A key a later compatible version adds is ignored, in the header and in the play key.
        let extra = try stream(fixture, editingHeader: { header in
            header["future"] = [1, 2]
            var p = header["play"] as! [String: Any]
            p["future"] = "x"
            header["play"] = p
        })
        #expect(decode(extra).error == nil)
        // The snapshot cadence need not be a multiple of the tick (the engine's `alife play` checks that against its own
        // constant; a stream written with another cadence, a test's, is still a stream).
        let odd = decode(try stream(fixture, editingHeader: play { $0["snapshot_every"] = 10 }))
        let allEvents = try fixture.events().count
        #expect(odd.error == nil && odd.events.count == allEvents)
        // No play key at all is a plain stream header with play records in it.
        let bare = decode(try stream(fixture, editingHeader: { $0["play"] = nil }))
        let all = try fixture.events().count
        #expect(bare.error == nil && bare.events.count == all)
    }

    // MARK: Refusals

    private func refused(_ bytes: [UInt8], chunk: Int? = 13) -> StreamError? { decode(bytes, chunk: chunk).error }

    private func isMalformed(_ error: StreamError?, _ tag: String, containing text: String) -> Bool {
        guard case .malformedRecord(let t, let why)? = error else { return false }
        return t == tag && why.contains(text)
    }

    @Test(arguments: [(0, "house"), (1, "kin"), (2, "colony")])
    func aSlotNoEntryOwnsIsRefused(column: Int, name: String) throws {
        let fixture = try PlayFixture(kind: "3d")
        let offset = try fixture.offset(of: "IDNT")
        let jsonLength = Int(fixture.le32(at: offset + 8 + 24))
        let rows = offset + 8 + 32 + (jsonLength + 3) / 4 * 4
        // Tables hold 2 houses, 3 kin groups and 1 colony: the row 0 slots 5, 5, 1 are outside; 252 is too.
        let outside: [UInt8] = [5, 5, 1]
        for value in [outside[column], 252] {
            let bytes = patched(fixture.bytes, at: rows + 12 + column, [UInt8(value)])
            #expect(isMalformed(refused(bytes), "IDNT", containing: "\(name) slot \(value)"), "\(name) \(value)")
        }
        // 253, 254 and 255 are the reserved values and always valid.
        for value in [253, 254, 255] {
            #expect(refused(patched(fixture.bytes, at: rows + 12 + column, [UInt8(value)])) == nil)
        }
    }

    /// The offset of the first occurrence of `text` inside the `IDNT` record (its JSON), for a same-length patch.
    private func idntOffset(of text: String, in fixture: PlayFixture) throws -> Int {
        let start = try fixture.offset(of: "IDNT")
        let end = start + 8 + Int(fixture.le32(at: start + 4))
        let needle = Array(text.utf8)
        let hay = fixture.bytes
        let at = try #require((start..<(end - needle.count)).first { hay[$0..<($0 + needle.count)].elementsEqual(needle) })
        return at
    }

    @Test func aSlotIsTheSlotOfAnEntryNotItsPosition() throws {
        let fixture = try PlayFixture(kind: "3d")
        let offset = try fixture.offset(of: "IDNT")
        let jsonLength = Int(fixture.le32(at: offset + 8 + 24))
        let rows = offset + 8 + 32 + (jsonLength + 3) / 4 * 4
        // House entry 1 moves to slot 3 (a gap at 1 and 2); the rows that named slot 1 are then refused ...
        let moved = patched(fixture.bytes, at: try idntOffset(of: "\"slot\":1,\"id\":5000000030", in: fixture) + 7, [0x33])
        #expect(isMalformed(refused(moved), "IDNT", containing: "house slot 1"))
        // ... until they name slot 3: rows 2 and 3 of the fixture hold the house byte 1.
        var fixed = moved
        for row in [2, 3] { fixed = patched(fixed, at: rows + 16 * row + 12, [3]) }
        let events = decode(fixed, chunk: 13)
        #expect(events.error == nil)
        let tables = events.events.compactMap { event -> IdentityTables? in
            if case .record(.idnt(let ids)) = event { ids.tables } else { nil }
        }.first
        #expect(tables?.houses.map(\.slot) == [0, 3])
        // Two entries on one slot are refused, as is a slot of 16 or more.
        let repeated = patched(fixture.bytes, at: try idntOffset(of: "\"slot\":1,\"id\":5000000030", in: fixture) + 7, [0x30])
        #expect(isMalformed(refused(repeated), "IDNT", containing: "repeated slot"))
    }

    /// The record's tables carry null where the observer has nothing yet (a group with no similarity sample, no
    /// births column for a group that held no slot in the window, a derived table before the first record).
    @Test func tableValuesTheObserverCannotYetGiveAreNull() throws {
        let json = """
            {"houses":[],"exact":false,"origin":250,"kin":[{"slot":3,"id":9,"name":"Nameless 9","named_after":null,\
            "size":12,"clade":4,"clade_share":null,"sim_self":null,"x":null,"y":null,"arch":null,"births":null,\
            "deaths":null}],"colonies":[{"slot":0,"id":2,"size":4,"kin":-1,"x":null,"y":null}],"unknown_immigrant":0}
            """
        let tables = try JSONDecoder().decode(IdentityTables.self, from: Data(json.utf8))
        let kin = try #require(tables.kin.first)
        #expect(kin.slot == 3 && kin.simSelf == nil && kin.births == nil && kin.x == nil && kin.cladeShare == nil)
        #expect(tables.colonies.first?.x == nil && tables.colonies.first?.kin == -1)
    }

    @Test func aTerritoryOwnerOutsideItsTableIsRefused() throws {
        let fixture = try PlayFixture(kind: "3d")
        let offset = try fixture.offset(of: "TERR")
        let jsonLength = Int(fixture.le32(at: offset + 8 + 24))
        let owner = offset + 8 + 32 + (jsonLength + 3) / 4 * 4
        #expect(isMalformed(refused(patched(fixture.bytes, at: owner, [3])), "TERR", containing: "owner slot 3"))
        #expect(refused(patched(fixture.bytes, at: owner, [254])) == nil)
    }

    @Test func aFocusSlotOutsideTheTableIsRefused() throws {
        let fixture = try PlayFixture(kind: "3d")
        let offset = try fixture.offset(of: "FOCS")
        #expect(isMalformed(refused(patched(fixture.bytes, at: offset + 8 + 77, [16])), "FOCS", containing: "house slot 16"))
        #expect(isMalformed(refused(patched(fixture.bytes, at: offset + 8 + 78, [200])), "FOCS", containing: "kin slot 200"))
        #expect(refused(patched(fixture.bytes, at: offset + 8 + 77, [15])) == nil)
    }

    @Test func aFrameThatContradictsTheHeaderOrItselfIsRefused() throws {
        let fixture = try PlayFixture(kind: "3d")
        let offset = try fixture.offset(of: "FRMA")
        let payload = offset + 8
        // N differs from the header's.
        #expect(isMalformed(refused(patched(fixture.bytes, at: payload + 32, le32(65))), "FRMA", containing: "N 65"))
        // n above N, and n that is not the popcount of the bitmap (the length no longer matches, or the popcount does not).
        #expect(isMalformed(refused(patched(fixture.bytes, at: payload + 16, le32(65))), "FRMA", containing: "exceeds N"))
        let wrongCount = refused(patched(fixture.bytes, at: payload + 16, le32(6)))
        guard case .lengthMismatch(let tag, _, _)? = wrongCount else { Issue.record("\(String(describing: wrongCount))"); return }
        #expect(tag == "FRMA")
        // A flipped bitmap bit keeps the length and breaks the popcount.
        var bitmap = Array(fixture.bytes[(payload + 36)..<(payload + 40)])
        bitmap[0] ^= 0x80  // row 7 alive
        #expect(isMalformed(refused(patched(fixture.bytes, at: payload + 36, bitmap)), "FRMA", containing: "bitmap lists 8"))
        // More bonds than the slots allow, a colour mode that does not exist.
        #expect(isMalformed(refused(patched(fixture.bytes, at: payload + 20, le32(1000))), "FRMA", containing: "n_bonds"))
        #expect(isMalformed(refused(patched(fixture.bytes, at: payload + 24, le32(7))), "FRMA", containing: "color_index 7"))
        // A bond (i, j) with i >= j, or j >= n: the bonds start after the bitmap and 28·n bytes.
        let bonds = payload + 36 + 8 + 28 * 7
        #expect(isMalformed(refused(patched(fixture.bytes, at: bonds, le32(1) + le32(1))), "FRMA", containing: "bond 0"))
        #expect(isMalformed(refused(patched(fixture.bytes, at: bonds + 4, le32(7))), "FRMA", containing: "bond 0"))
        // A bitmap bit at or past N is set (rows beyond a 64-row world).
        let n70 = try PlayFixture(kind: "3d_n70")
        let frame70 = try n70.offset(of: "FRMA") + 8
        var last = Array(n70.bytes[(frame70 + 36 + 8)..<(frame70 + 36 + 12)])
        last[0] |= 0x40  // row 70
        #expect(refused(patched(n70.bytes, at: frame70 + 36 + 8, last)) != nil)
    }

    @Test func recordsWhoseCountsAndJSONDisagreeAreRefused() throws {
        let fixture = try PlayFixture(kind: "3d")
        let evnt = try fixture.offset(of: "EVNT") + 8
        #expect(isMalformed(refused(patched(fixture.bytes, at: evnt + 16, le32(3))), "EVNT", containing: "count 3"))
        #expect(isMalformed(refused(patched(fixture.bytes, at: evnt + 16, le32(65))), "EVNT", containing: "cap of 64"))
        let strm = try fixture.offset(of: "STRM") + 8
        #expect(isMalformed(refused(patched(fixture.bytes, at: strm + 24, le32(3))), "STRM", containing: "n_storms 3"))
        #expect(isMalformed(refused(patched(fixture.bytes, at: strm + 24, le32(33))), "STRM", containing: "cap of 32"))
        let idnt = try fixture.offset(of: "IDNT") + 8
        #expect(isMalformed(refused(patched(fixture.bytes, at: idnt + 28, le32(1))), "IDNT", containing: "reserved"))
        #expect(isMalformed(refused(patched(fixture.bytes, at: idnt + 16, le32(65))), "IDNT", containing: "exceed N"))
        #expect(isMalformed(refused(patched(fixture.bytes, at: idnt + 24, le32(1 << 21))), "IDNT", containing: "json_len"))
        // JSON that is not JSON, and JSON of the wrong shape: the braces of the first object, then its first key.
        let jsonStart = idnt + 32
        #expect(isMalformed(refused(patched(fixture.bytes, at: jsonStart, [0x5B])), "IDNT", containing: "json"))
        let key = fixture.bytes[jsonStart..<(jsonStart + 12)]
        #expect(String(decoding: key, as: UTF8.self) == "{\"houses\":[{")
        #expect(isMalformed(refused(patched(fixture.bytes, at: jsonStart + 2, Array("xouses".utf8))), "IDNT", containing: "houses"))
        // A record shorter than the common head, and one whose length is not a multiple of 4.
        let clck = try fixture.offset(of: "CLCK")
        #expect(refused(patched(fixture.bytes, at: clck + 4, le32(8))) == .lengthMismatch(tag: "CLCK", declared: 8, expected: 16))
        #expect(isMalformed(refused(patched(fixture.bytes, at: clck + 4, le32(62))), "CLCK", containing: "multiple of 4"))
    }

    @Test func nonZeroPaddingAfterJSONIsRefused() throws {
        let fixture = try PlayFixture(kind: "3d")
        var checked = 0
        for (tag, lengthOffset, jsonOffset) in [("IDNT", 24, 32), ("TERR", 24, 32), ("STRM", 16, 32), ("EVNT", 20, 24), ("ENVS", 16, 20)] {
            for (index, offset) in fixture.offsets(of: tag).enumerated() {
                let jsonLength = Int(fixture.le32(at: offset + 8 + lengthOffset))
                guard jsonLength % 4 != 0 else { continue }
                let pad = offset + 8 + jsonOffset + jsonLength
                #expect(refused(fixture.bytes) == nil)
                #expect(isMalformed(refused(patched(fixture.bytes, at: pad, [1])), tag, containing: "padding"), "\(tag) \(index)")
                checked += 1
            }
        }
        #expect(checked >= 3, "the fixture has records whose JSON needs padding")
    }

    @Test func aMapWithoutItsPlanesFlagsIsRefused() throws {
        let fixture = try PlayFixture(kind: "3d")
        let map = try fixture.offset(of: "MAP1") + 8
        #expect(isMalformed(refused(patched(fixture.bytes, at: map + 14, [1, 0])), "MAP1", containing: "T_s and the winds"))
        #expect(isMalformed(refused(patched(fixture.bytes, at: map + 16, [160, 0, 3, 0])), "MAP1", containing: "wind blocks"))
        #expect(isMalformed(refused(patched(fixture.bytes, at: map + 40, [1])), "MAP1", containing: "reserved"))
    }

    @Test func aClockWithAnUnknownReasonOrSavingByteIsRefused() throws {
        let fixture = try PlayFixture(kind: "3d")
        let clck = try fixture.offset(of: "CLCK") + 8
        #expect(isMalformed(refused(patched(fixture.bytes, at: clck + 40, [6])), "CLCK", containing: "reason 6"))
        #expect(isMalformed(refused(patched(fixture.bytes, at: clck + 41, [2])), "CLCK", containing: "saving"))
    }

    @Test func aLengthThatDisagreesWithTheCountsIsRefusedBeforeThePayloadArrives() throws {
        let fixture = try PlayFixture(kind: "3d")
        let focs = try fixture.offset(of: "FOCS")
        let bytes = patched(fixture.bytes, at: focs + 4, le32(80))
        // Only the prefix has arrived: the 8-byte prefix and the 16-byte head.
        // The records before it complete, so the error is reported by `failure` (and thrown by the next call).
        var decoder = FrameDecoder()
        _ = try decoder.feed(Data(bytes[0..<(focs + 24)]))
        #expect(decoder.failure == .lengthMismatch(tag: "FOCS", declared: 80, expected: 84))
        // An FRMA that declares one row too many, with its counts in the fixed prefix.
        let frma = try fixture.offset(of: "FRMA")
        let declared = Int(fixture.le32(at: frma + 4)) + 4
        var second = FrameDecoder()
        let cut = frma + 8 + 16 + 20
        _ = try second.feed(Data(patched(fixture.bytes, at: frma + 4, le32(declared))[0..<cut]))
        #expect(second.failure == .lengthMismatch(tag: "FRMA", declared: declared, expected: declared - 4))
        #expect(throws: StreamError.lengthMismatch(tag: "FRMA", declared: declared, expected: declared - 4)) {
            _ = try second.feed(Data([0]))
        }
    }

    @Test func malformedRecordsHaveReadableDescriptions() {
        let error = StreamError.malformedRecord(tag: "FOCS", "house slot 16 is not a slot or 253 to 255")
        #expect(error.description == "malformed stream record FOCS: house slot 16 is not a slot or 253 to 255")
    }

    // MARK: Types

    @Test func theSlotConstantsAreTheReservedBytes() {
        #expect(Slot.unknown == 253 && Slot.other == 254 && Slot.none == 255 && Slot.maxTable == 16)
        #expect(Slot.isValid(0, tableSize: 1) && !Slot.isValid(1, tableSize: 1) && !Slot.isValid(252, tableSize: 16))
        #expect(Slot.isValid(253, tableSize: 0) && Slot.isValid(255, tableSize: 0))
    }

    @Test func aWeatherMapDecodesRainFromTheCompandedByte() throws {
        for case .record(.map1(let map)) in try PlayFixture(kind: "3d").events() {
            #expect(map.rain(0) == 0 && map.rain(255) == map.pRef)
            #expect(abs(map.rain(128) - map.pRef * (128.0 / 255) * (128.0 / 255)) < 1e-9)
            #expect(map.p.count == map.n * map.n && map.winds.count == 4 * map.nw * map.nw)
        }
    }

    @Test func clockReasonsAndFocusFlagsAreNamed() throws {
        let clocks = try PlayFixture(kind: "3d").events().compactMap { event -> PlayClock? in
            if case .record(.clck(let c)) = event { c } else { nil }
        }
        #expect(clocks.map(\.reason) == [.running, .userPause] && clocks.map(\.saving) == [false, true])
        #expect(clocks[1].rate == -1 && clocks[1].lateSteps == 3 && clocks[1].saveStallSteps == 21)
    }
}
