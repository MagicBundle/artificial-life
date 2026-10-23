import Foundation
import Testing

@testable import AlifeCore

/// The records G3b adds (`ALRT`, `RPRT`, `PLAN`) and the keys it adds to the header, `ENVS`, `CLCK` and `IDNT`, against the
/// independent Python decoder's sidecar (`play_tiny3d_g3b`) and against records built by hand for every refusal
/// (DESIGN §34.6). The fixtures check the happy path in `PlayRecordsTests`; this suite is the edges.
@Suite struct PlayRecordsG3bTests {
    static let kind = "3d_g3b"

    // MARK: Building records by hand

    /// The bytes of one record: tag, length, the common head, `words` (the fixed prefix), the JSON and its zero padding.
    private func record(_ tag: String, step: UInt64 = 12000, seq: Int = 0, version: Int = 1, flags: Int = 0,
                        words: [Int], json: String) -> [UInt8] {
        var js = Array(json.utf8)
        let length = js.count
        js += [UInt8](repeating: 0, count: (4 - length % 4) % 4)
        var payload: [UInt8] = []
        for i in 0..<8 { payload.append(UInt8((step >> (8 * UInt64(i))) & 0xFF)) }
        payload += le32(seq) + [UInt8(version & 0xFF), UInt8(version >> 8), UInt8(flags & 0xFF), UInt8(flags >> 8)]
        for w in words { payload += le32(w == -1 ? length : w) }
        payload += js
        return Array(tag.utf8) + le32(payload.count) + payload
    }

    /// `-1` in a prefix stands for the JSON's length.
    private func alrt(_ alerts: [[String: Any]], count: Int? = nil, flags: Int = 0, version: Int = 1) throws -> [UInt8] {
        record("ALRT", version: version, flags: flags, words: [count ?? alerts.count, -1], json: try text(alerts))
    }

    private func rprt(_ report: [String: Any], chapter: Int? = nil, mode: Int? = nil, flags: Int? = nil,
                      reserved: Int = 0) throws -> [UInt8] {
        let final = (report["final"] as? Bool) == true
        return record("RPRT", flags: flags ?? (final ? 2 : 0),
                      words: [chapter ?? (report["chapter"] as! Int), mode ?? (report["mode"] as! Int), -1, reserved],
                      json: try text(report))
    }

    private func plan(_ plan: [String: Any], flags: Int? = nil) throws -> [UInt8] {
        let reply = (plan["status"] as? String) == "preview"
        return record("PLAN", flags: flags ?? (reply ? 1 : 0), words: [-1], json: try text(plan))
    }

    private func text(_ object: Any) throws -> String {
        String(decoding: try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]), as: UTF8.self)
    }

    private func fields(_ fixture: PlayFixture, _ tag: String, _ index: Int = 0) throws -> [String: Any] {
        let matches = fixture.records.filter { $0["tag"] as? String == tag }
        return try #require(matches.indices.contains(index) ? matches[index]["fields"] as? [String: Any] : nil)
    }

    /// The first live alert, the template report of chapter 2 and the first `PLAN` as dictionaries, to edit.
    private func samples(_ fixture: PlayFixture) throws -> (alert: [String: Any], report: [String: Any], plan: [String: Any]) {
        let alerts = try #require(try fields(fixture, "ALRT", 1)["json"] as? [[String: Any]])
        return (alerts[0], try #require(try fields(fixture, "RPRT", 1)["json"] as? [String: Any]),
                try #require(try fields(fixture, "PLAN", 0)["json"] as? [String: Any]))
    }

    /// The fixture's header and nothing else, then `records`.
    private func stream(_ fixture: PlayFixture, _ records: [UInt8]...) -> [UInt8] {
        Array(fixture.bytes[0..<fixture.headerEnd]) + records.flatMap { $0 }
    }

    private func isMalformed(_ error: StreamError?, _ tag: String, containing text: String) -> Bool {
        guard case .malformedRecord(let t, let why)? = error else { return false }
        return t == tag && why.contains(text)
    }

    private func refused(_ bytes: [UInt8]) -> StreamError? { decode(bytes, chunk: 29).error }

    // MARK: The fixture, decoded

    @Test func theG3bFixtureHoldsOneOfEveryNewRecord() throws {
        let fixture = try PlayFixture(kind: Self.kind)
        let records = try fixture.events().compactMap { event -> StreamRecord? in if case .record(let r) = event { r } else { nil } }
        #expect(records.map(\.tag).filter { ["ALRT", "RPRT", "PLAN"].contains($0) }
            == ["ALRT", "RPRT", "ALRT", "RPRT", "RPRT", "RPRT", "PLAN", "PLAN", "PLAN", "PLAN", "PLAN", "PLAN", "PLAN"])
        let alerts = records.flatMap { r -> [PlayAlert] in if case .alrt(let b) = r { b.alerts } else { [] } }
        #expect(Set(alerts.map(\.state)) == [.raised, .updated, .cleared])
        #expect(Set(alerts.map(\.surface)) == [.alert, .log])
        #expect(Set(alerts.map(\.imp)) == [0, 1, 2, 3])
        #expect(Set(alerts.compactMap { $0.ref?.kind }) == [.world, .storm, .kin, .power, .organism, .house])
        let plans = records.compactMap { r -> PlanBody? in if case .plan(let p) = r { p.plan } else { nil } }
        #expect(Set(plans.map(\.status)) == [.preview, .queued, .applied, .since, .refused, .cancelled])
        #expect(plans.contains { $0.kind == .intent && $0.status == .refused })
        let applied = try #require(plans.first { $0.status == .applied })
        #expect(applied.stepApplied == 12000 && applied.ends == 18000 && applied.id == 3)
        let since = try #require(plans.first { $0.status == .since }?.since)
        #expect(since.from == 12000 && since.rows.first?.alive == 44100 && since.rows.first?.storm == 3)
    }

    @Test func aReportDecodesWithItsSectionsAndRefs() throws {
        let fixture = try PlayFixture(kind: Self.kind)
        let reports = try fixture.events().compactMap { event -> PlayReport? in if case .record(.rprt(let r)) = event { r } else { nil } }
        #expect(reports.map(\.mode) == [0, 0, 1, 2] && reports.map(\.chapter) == [1, 2, 2, 2])
        #expect(reports.map(\.isFinal) == [true, true, false, true] && reports.map(\.isBacklog) == [true, false, false, false])
        let report = reports[1].report
        #expect(report.nAbs == 2 && report.a == 6000 && report.b == 12000 && !report.isWithheld)
        #expect(report.headline?.refs == ["S7"] && report.entities["S7"]?.kind == "storm" && report.entities["K17"]?.name == "Aldric")
        #expect(report.body.map(\.id) == ["TO", "K1"] && report.interventions.map(\.id) == ["I1"])
        #expect(report.since.influence == ReportBody.Since.Influence(have: 4, cap: 6))
        #expect(report.since.ending == [ReportBody.Since.Ending(action: "regen_pulse", end: 18000)])
        #expect(report.versions == ReportBody.Versions(report: 2, template: 2, checker: 1, lexicon: 3))
        #expect(reports[2].report.fallback == "ordering" && !reports[2].report.final)
        let away = reports[3].report
        #expect(away.isWithheld && away.headline == nil && away.lead == nil && away.body.isEmpty && away.entities.isEmpty)
    }

    @Test func theHeaderCarriesTheGameKeys() throws {
        let fixture = try PlayFixture(kind: Self.kind)
        guard case .header(let header)? = try fixture.events().first else { Issue.record("no header"); return }
        let play = try #require(header.play)
        let alerts = try #require(play.alerts)
        #expect(alerts.version == 1 && alerts.cap == PlayHeader.Alerts.Cap(n: 4, window: 1500, exempt: ["imp3", "pause", "chapter", "engine"]))
        #expect(alerts.classes["chapter"]?.pause == true && alerts.classes["chapter"]?.surface == .alert)
        #expect(alerts.classes["lineage_formed"]?.surface == .log && alerts.classes["lineage_formed"]?.imp == 2)
        #expect(alerts.classes["pop_fall"]?.calibrated == false && alerts.classes["storm_named"]?.calibrated == true)
        #expect(play.autopause?.chapter == true && play.autopause?.alert["pop_fall"] == true && play.autopause?.alert["storm_named"] == false)
        let food = try #require(play.powers?["food"])
        #expect(food.action == "regen_pulse" && food.cost == 1 && food.cooldown == 3000 && food.scope == "global")
        #expect(food.duration == [3000, 12000])
        #expect(food.params?["mult"] == .array([.array([.number(0.6), .number(0.9)]), .array([.number(1.15), .number(1.5)])]))
        #expect(play.powers?["strike"]?.duration == nil && play.gameSpecs == "game-v0")
        #expect(play.records.suffix(3) == ["ALRT", "RPRT", "PLAN"] && play.versions["PLAN"] == 1)
    }

    @Test func theEnvironmentCarriesInfluencePatchesAndPowers() throws {
        let store = try WorldFixtures.store(Self.kind)
        let influence = try #require(store.influence)
        #expect(influence == PlayEnvironment.Influence(have: 4, cap: 6, window: 18000, freeAt: [13000, 15500]))
        #expect(store.patches.map(\.i) == [0, 1] && store.patches[1].sigma == 4.5 && store.patches[0].gain == 1.5)
        #expect(store.powerStates["food"] == PlayEnvironment.PowerState(ready: nil, busy: nil))
        #expect(store.powerStates["strike"]?.busy == "cooldown until step 15000" && store.powerStates["strike"]?.ready == 15000)
        let effects = try #require(store.environment?.environment.effects)
        #expect(effects.map(\.source) == ["player", "rule"])
        #expect(store.powerSpecs.keys.sorted() == ["food", "strike"])
        // A G3a world has none of it.
        let old = try WorldFixtures.store("3d")
        #expect(old.influence == nil && old.patches.isEmpty && old.powerStates.isEmpty && old.powerSpecs.isEmpty)
    }

    @Test func theClockFlagsAndReasonsOfAChapterAreNamed() throws {
        let clocks = try PlayFixture(kind: Self.kind).events().compactMap { event -> PlayClock? in
            if case .record(.clck(let c)) = event { c } else { nil }
        }
        #expect(clocks.map(\.reason) == [.chapter, .alert] && clocks.map(\.saving) == [true, false])
        #expect(clocks.map(\.chapterPauseArmed) == [true, true] && clocks.map(\.alertPauseArmed) == [true, false])
        #expect(clocks.map(\.chapter) == [2, 3] && clocks.map(\.inChapter) == [6000, 1] && clocks.allSatisfy { $0.chapterLen == 6000 })
    }

    @Test func aKinGroupNamesItsThread() throws {
        let store = try WorldFixtures.store(Self.kind)
        let kin = try #require(store.identities?.tables.kin)
        #expect(kin.map(\.thread) == [4, 5, 6])
        #expect(try WorldFixtures.store("3d").identities?.tables.kin.allSatisfy { $0.thread == nil } == true, "a G3a stream has none")
    }

    // MARK: Chunking, truncation, unknown versions

    @Test func theUnknownVersionOfTheFixtureIsSkippedAndCounted() throws {
        let fixture = try PlayFixture(kind: Self.kind)
        let skipped = try #require(fixture.skipped.first)
        #expect(skipped["tag"] as? String == "ALRT" && skipped["why"] as? String == "version")
        let result = decodeWithDecoder(fixture.bytes, chunk: 7)
        #expect(result.error == nil && result.decoder.skippedVersions == 1)
    }

    @Test(arguments: ["ALRT", "RPRT", "PLAN"], PlayRecordsTests.chunkings)
    func aNewTagOfAnotherVersionIsSkippedAndCounted(tag: String, chunk: Int?) throws {
        let fixture = try PlayFixture(kind: Self.kind)
        let offset = try fixture.offset(of: tag)
        let length = Int(fixture.le32(at: offset + 4))
        let result = decodeWithDecoder(patched(fixture.bytes, at: offset + 8 + 12, [2, 0]), chunk: chunk)
        #expect(result.error == nil && result.events.contains(.unknownRecord(tag: tag, length: length)))
        #expect(result.decoder.skippedVersions == 2, "the fixture's own skipped ALRT, and this one")
    }

    @Test func aTruncationIsRefusedExceptBetweenRecords() throws {
        let fixture = try PlayFixture(kind: Self.kind)
        let boundaries = Set(fixture.boundaries)
        var cuts = Set(stride(from: 1, to: fixture.bytes.count, by: 53))
        for b in boundaries { cuts.formUnion((b - 12)...(b + 36)) }
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
                #expect(failure == nil, "a stream cut after \(cut) bytes is complete")
            } else {
                guard case .truncated? = failure else { Issue.record("cut \(cut): \(String(describing: failure))"); return }
            }
        }
    }

    // MARK: Refusals

    @Test func anAlertBatchThatContradictsItselfIsRefused() throws {
        let fixture = try PlayFixture(kind: Self.kind)
        let alert = try samples(fixture).alert
        #expect(refused(stream(fixture, try alrt([alert]))) == nil, "the sample is valid")
        func with(_ edit: (inout [String: Any]) -> Void) throws -> [UInt8] {
            var a = alert
            edit(&a)
            return stream(fixture, try alrt([a]))
        }
        #expect(isMalformed(refused(stream(fixture, try alrt([alert], count: 2))), "ALRT", containing: "count 2"))
        #expect(isMalformed(refused(stream(fixture, try alrt([alert, alert]))), "ALRT", containing: "twice"))
        #expect(isMalformed(refused(try with { $0["imp"] = 4 }), "ALRT", containing: "importance 4"))
        #expect(isMalformed(refused(try with { $0["id"] = 12_000_599 }), "ALRT", containing: "alert range"))
        #expect(isMalformed(refused(try with { $0["version"] = 2 }), "ALRT", containing: "raised at version 2"))
        #expect(isMalformed(refused(try with { $0["state"] = "updated"; $0["version"] = 1 }), "ALRT", containing: "updated at version 1"))
        #expect(isMalformed(refused(try with { $0["title"] = String(repeating: "t", count: 49) }), "ALRT", containing: "title"))
        #expect(isMalformed(refused(try with { $0["text"] = String(repeating: "t", count: 161) }), "ALRT", containing: "text"))
        #expect(refused(try with { $0["title"] = String(repeating: "t", count: 48); $0["text"] = String(repeating: "t", count: 160) }) == nil)
        #expect(isMalformed(refused(try with { $0["state"] = "dead" }), "ALRT", containing: "json"))
        #expect(isMalformed(refused(try with { $0["surface"] = "sound" }), "ALRT", containing: "json"))
        #expect(isMalformed(refused(try with { $0["ref"] = ["kind": "moon", "id": 1] }), "ALRT", containing: "json"))
        #expect(isMalformed(refused(try with { $0["class"] = nil }), "ALRT", containing: "class"))
        // More than sixteen alerts, and a JSON over the cap, are refused before the payload is waited for.
        let seventeen = (0..<17).map { i -> [String: Any] in var a = alert; a["id"] = 12_000_600 + i; return a }
        #expect(isMalformed(refused(stream(fixture, try alrt(seventeen))), "ALRT", containing: "cap of 16"))
        let big = record("ALRT", words: [1, 24 * 1024 + 1], json: "[]")
        #expect(isMalformed(refused(stream(fixture, big)), "ALRT", containing: "24 KiB"))
        // Non-zero padding after the JSON.
        let alerts = try fixture.offset(of: "ALRT")
        let jsonLength = Int(fixture.le32(at: alerts + 8 + 20))
        #expect(jsonLength % 4 != 0, "the fixture's JSON needs padding")
        #expect(isMalformed(refused(patched(fixture.bytes, at: alerts + 8 + 24 + jsonLength, [1])), "ALRT", containing: "padding"))
    }

    @Test func sixteenAlertsAreOneRecordAndTheJSONCapsAreExact() throws {
        let fixture = try PlayFixture(kind: Self.kind)
        let alert = try samples(fixture).alert
        let sixteen = (0..<16).map { i -> [String: Any] in var a = alert; a["id"] = 12_000_600 + i; return a }
        let result = decode(stream(fixture, try alrt(sixteen)))
        #expect(result.error == nil)
        guard case .record(.alrt(let batch))? = result.events.last else { Issue.record("no ALRT"); return }
        #expect(batch.alerts.count == 16)
        // The length that a prefix declares is checked against the cap before the payload is waited for: the cap itself is
        // allowed, one byte more is not.
        guard case .header(let header)? = try fixture.events().first else { Issue.record("no header"); return }
        func expected(_ tag: String, jsonLength: Int) throws -> Int {
            var prefix = [UInt8](repeating: 0, count: 16)  // the common head
            switch tag {
            case "ALRT": prefix += le32(1) + le32(jsonLength)
            case "RPRT": prefix += le32(1) + le32(0) + le32(jsonLength) + le32(0)
            default: prefix += le32(jsonLength)
            }
            return try prefix.withUnsafeBytes { try RecordDecoder.expectedLength(tag, ByteReader(raw: $0, start: 0), header: header) }
        }
        for (tag, cap, fixed) in [("ALRT", 24 << 10, 24), ("RPRT", 64 << 10, 32), ("PLAN", 8 << 10, 20)] {
            #expect(try expected(tag, jsonLength: cap) == fixed + cap, "\(tag) at the cap")
            #expect(throws: StreamError.self) { try expected(tag, jsonLength: cap + 1) }
            #expect(try expected(tag, jsonLength: 0) == fixed, "\(tag) with no JSON")
            #expect(try expected(tag, jsonLength: 1) == fixed + 4, "\(tag) pads to a multiple of 4")
        }
    }

    @Test func aReportThatContradictsItsHeadOrItselfIsRefused() throws {
        let fixture = try PlayFixture(kind: Self.kind)
        let report = try samples(fixture).report
        #expect(refused(stream(fixture, try rprt(report))) == nil, "the sample is valid")
        #expect(isMalformed(refused(stream(fixture, try rprt(report, chapter: 3))), "RPRT", containing: "chapter 3"))
        #expect(isMalformed(refused(stream(fixture, try rprt(report, mode: 1))), "RPRT", containing: "mode 1"))
        #expect(isMalformed(refused(stream(fixture, try rprt(report, flags: 0))), "RPRT", containing: "final flag"))
        #expect(isMalformed(refused(stream(fixture, try rprt(report, reserved: 1))), "RPRT", containing: "reserved"))
        var third = report
        third["mode"] = 3
        #expect(isMalformed(refused(stream(fixture, try rprt(third))), "RPRT", containing: "mode 3"))
        var backwards = report
        backwards["a"] = 12001
        #expect(isMalformed(refused(stream(fixture, try rprt(backwards))), "RPRT", containing: "after"))
        var guessing = report
        guessing["withheld"] = ["the cut 11000 is not the step of a metrics row"]
        #expect(isMalformed(refused(stream(fixture, try rprt(guessing))), "RPRT", containing: "withheld"))
        // Any one part left in a withheld report is a guess.
        for (part, value) in [("headline", report["headline"]!), ("lead", report["lead"]!), ("body", report["body"]!),
                              ("interventions", report["interventions"]!)] {
            var one = guessing
            for p in ["headline", "lead"] { one[p] = NSNull() }
            for p in ["body", "interventions"] { one[p] = [Any]() }
            one[part] = value
            #expect(isMalformed(refused(stream(fixture, try rprt(one))), "RPRT", containing: "withheld"), "\(part)")
        }
        var empty = guessing
        empty["headline"] = NSNull()
        empty["lead"] = NSNull()
        empty["body"] = [Any]()
        empty["interventions"] = [Any]()
        #expect(refused(stream(fixture, try rprt(empty))) == nil, "an empty withheld report is the honest one")
        var broken = report
        broken["since"] = nil
        #expect(isMalformed(refused(stream(fixture, try rprt(broken))), "RPRT", containing: "since"))
        let big = record("RPRT", words: [1, 0, 64 * 1024 + 1, 0], json: "{}")
        #expect(isMalformed(refused(stream(fixture, big)), "RPRT", containing: "64 KiB"))
        // A report of 64 KiB exactly is accepted (the cap is on the JSON), one byte more is not.
        var wide = report
        let filler = String(repeating: "x", count: 64 * 1024)
        wide["facts_digest"] = String(filler.prefix(64 * 1024 - (try text(report).utf8.count - 64)))
        #expect(try text(wide).utf8.count == 64 * 1024)
        #expect(refused(stream(fixture, try rprt(wide))) == nil)
        wide["facts_digest"] = (wide["facts_digest"] as! String) + "x"
        #expect(isMalformed(refused(stream(fixture, try rprt(wide))), "RPRT", containing: "64 KiB"))
    }

    @Test func aPlanThatContradictsItselfIsRefused() throws {
        let fixture = try PlayFixture(kind: Self.kind)
        let preview = try samples(fixture).plan
        #expect(refused(stream(fixture, try plan(preview))) == nil, "the sample is valid")
        func with(_ edit: (inout [String: Any]) -> Void, flags: Int? = nil) throws -> [UInt8] {
            var p = preview
            edit(&p)
            return stream(fixture, try plan(p, flags: flags))
        }
        #expect(isMalformed(refused(try with({ $0["status"] = "refused" })), "PLAN", containing: "refused is a reason"))
        #expect(isMalformed(refused(try with({ $0["refused"] = "no" })), "PLAN", containing: "refused is a reason"))
        #expect(isMalformed(refused(try with({ _ in }, flags: 0)), "PLAN", containing: "reply"))
        #expect(isMalformed(refused(try with({ $0["status"] = "applied"; $0["step_applied"] = NSNull() })), "PLAN", containing: "names its step"))
        #expect(isMalformed(refused(try with({ $0["text"] = String(repeating: "t", count: 161) })), "PLAN", containing: "160"))
        #expect(isMalformed(refused(try with({ $0["caution"] = [String(repeating: "c", count: 101)] })), "PLAN", containing: "caution"))
        #expect(refused(try with({ $0["caution"] = [String(repeating: "c", count: 100)] })) == nil)
        #expect(isMalformed(refused(try with({ $0["status"] = "thinking" })), "PLAN", containing: "json"))
        #expect(isMalformed(refused(try with({ $0["kind"] = "wish" })), "PLAN", containing: "json"))
        #expect(isMalformed(refused(try with({ $0["id"] = -1 })), "PLAN", containing: "not negative"))
        #expect(isMalformed(refused(try with({ $0["cost"] = -1 })), "PLAN", containing: "not negative"))
        #expect(isMalformed(refused(try with({ $0["influence"] = ["have": 1] })), "PLAN", containing: "json"))
        let big = record("PLAN", words: [8 * 1024 + 1], json: "{}")
        #expect(isMalformed(refused(stream(fixture, big)), "PLAN", containing: "8 KiB"))
    }

    @Test func aMissingReplyFlagOnlyMattersForAPreview() throws {
        let fixture = try PlayFixture(kind: Self.kind)
        var applied = try samples(fixture).plan
        applied["status"] = "queued"
        #expect(refused(stream(fixture, try plan(applied, flags: 0))) == nil)
        #expect(refused(stream(fixture, try plan(applied, flags: 1))) == nil, "a reply may be any status (a refused intent is)")
    }

    // MARK: A reader that predates the records

    @Test(arguments: PlayRecordsTests.chunkings)
    func aReaderThatDoesNotKnowAnyTagPassesThemAllOn(chunk: Int?) throws {
        let fixture = try PlayFixture(kind: Self.kind)
        var bytes = fixture.bytes
        for record in fixture.records {
            bytes = patched(bytes, at: record["offset"] as! Int, Array("Q\((record["tag"] as! String).dropFirst())".utf8))
        }
        let result = decode(bytes, chunk: chunk)
        #expect(result.error == nil)
        guard case .header(let header)? = result.events.first else { Issue.record("no header"); return }
        #expect(header.play?.alerts != nil)
        #expect(result.events.dropFirst().allSatisfy { if case .unknownRecord = $0 { true } else { false } })
    }

    // MARK: The header

    @Test func aBadGameKeyInTheHeaderIsAMalformedHeader() throws {
        let fixture = try PlayFixture(kind: Self.kind)
        func stream(_ edit: (inout [String: Any]) -> Void) throws -> [UInt8] {
            var header = fixture.header
            var play = header["play"] as! [String: Any]
            edit(&play)
            header["play"] = play
            let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
            var text = [UInt8](json)
            while text.count % 4 != 0 { text.append(0x20) }
            return Array("ALSTRM".utf8) + [0, 1] + le32(text.count) + text + Array(fixture.bytes[fixture.headerEnd...])
        }
        func classes(_ edit: @escaping (inout [String: Any]) -> Void) -> (inout [String: Any]) -> Void {
            { play in
                var alerts = play["alerts"] as! [String: Any]
                edit(&alerts)
                play["alerts"] = alerts
            }
        }
        let edits: [(String, (inout [String: Any]) -> Void)] = [
            ("no classes", classes { $0["classes"] = [String: Any]() }),
            ("a class importance of 4", classes { var c = $0["classes"] as! [String: Any]; var k = c["chapter"] as! [String: Any]; k["imp"] = 4; c["chapter"] = k; $0["classes"] = c }),
            ("a surface that is none", classes { var c = $0["classes"] as! [String: Any]; var k = c["chapter"] as! [String: Any]; k["surface"] = "sound"; c["chapter"] = k; $0["classes"] = c }),
            ("a cap of zero", classes { var c = $0["cap"] as! [String: Any]; c["n"] = 0; $0["cap"] = c }),
            ("a negative cooldown", { play in var p = play["powers"] as! [String: Any]; var f = p["food"] as! [String: Any]; f["cooldown"] = -1; p["food"] = f; play["powers"] = p }),
            ("a negative cost", { play in var p = play["powers"] as! [String: Any]; var f = p["food"] as! [String: Any]; f["cost"] = -1; p["food"] = f; play["powers"] = p }),
            ("a power without an action", { play in var p = play["powers"] as! [String: Any]; var f = p["food"] as! [String: Any]; f["action"] = nil; p["food"] = f; play["powers"] = p }),
            ("autopause of the wrong type", { $0["autopause"] = ["chapter": "yes", "alert": [String: Bool]()] }),
        ]
        for (name, edit) in edits {
            let result = decode(try stream(edit))
            guard case .malformedHeader? = result.error else { Issue.record("\(name): \(String(describing: result.error))"); continue }
            #expect(result.events.isEmpty, Comment(rawValue: name))
        }
        // Dropping the keys is the header a G3a world has: the records are still decoded.
        let bare = decode(try stream { $0["alerts"] = nil; $0["autopause"] = nil; $0["powers"] = nil; $0["game_specs"] = nil })
        #expect(bare.error == nil)
        guard case .header(let header)? = bare.events.first else { Issue.record("no header"); return }
        #expect(header.play?.alerts == nil && header.play?.autopause == nil && header.play?.powers == nil && header.play?.gameSpecs == nil)
    }
}

/// The strike card's count of organisms in its column (DESIGN §34.5.1) is a key the fixtures do not carry: decoded from JSON.
@Suite struct PlanSinceDiscTests {
    private func since(_ disc: String) throws -> PlanBody.Since {
        let json = """
        {"from": 12000, "rows": [{"step": 12000, "alive": 100, "starved": 1, "killed": 2, "born": 3, "storm": 4, "catastrophe": 5}]\(disc)}
        """
        return try JSONDecoder().decode(PlanBody.Since.self, from: Data(json.utf8))
    }

    @Test func theDiscIsOptionalAndItsSecondCountComesLater() throws {
        #expect(try since("").disc == nil)
        #expect(try since(#", "disc": {"before": 160}"#).disc == PlanBody.Since.Disc(before: 160, after: nil))
        #expect(try since(#", "disc": {"before": 160, "after": 61}"#).disc == PlanBody.Since.Disc(before: 160, after: 61))
        #expect(try since("").rows.count == 1)
    }
}
