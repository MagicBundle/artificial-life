import Foundation
import Testing

@testable import AlifeCore

/// The committed stream `tests/data/play_features.bin`, written by the observer series' real encoders (`tests/make_play_features.py`): a 3D
/// world with all five feature layers on, decoded whole and in the awkward chunkings, and reduced into a store.
@Suite struct FeatureStreamFixtureTests {
    private func bytes() throws -> [UInt8] {
        [UInt8](try Data(contentsOf: Repo.streamData.appendingPathComponent("play_features.bin")))
    }

    @Test(arguments: [nil, 1, 7, 64]) func theWritersStreamDecodesWhole(chunk: Int?) throws {
        let result = decode(try bytes(), chunk: chunk)
        #expect(result.error == nil)
        guard case .header(let header)? = result.events.first, let play = header.play else { Issue.record("no header"); return }
        #expect(header.colorModes.last == "role" && header.colorModes.count == 8)
        #expect(play.records == ["FRMA", "FOCS", "STRC", "FSIG", "FSOL"])
        #expect(play.structures == .init(enabled: true, cap: 4096, cells: [6, 12, 12], cell: 4))
        #expect(play.niche == .init(enabled: true, farm: true, cap: 4095, cells: [6, 12, 12], cell: 4))
        #expect(play.comm == .init(enabled: true, w: 3, range: 8, control: "none"))
        #expect(play.bonds == .init(flow: true, scale: 1000))
        let records = result.events.compactMap { event -> StreamRecord? in if case .record(let r) = event { r } else { nil } }
        #expect(records.map(\.tag) == ["FRMA", "STRC", "FSOL", "FSIG"])
        #expect(!result.events.contains { if case .unknownRecord = $0 { true } else { false } })
        guard case .frma(let frame) = records[0], case .strc(let material) = records[1], case .fsol(let soil) = records[2],
              case .fsig(let voice) = records[3] else { Issue.record("records out of order"); return }
        #expect(frame.n == 3 && frame.rows == [1, 2, 5] && frame.hasBondFlow && frame.flow == [-100, 33] && frame.bonds == [0, 1, 1, 2])
        #expect(header.colorMode(index: frame.colorIndex) == "role" && frame.rgb == [0xF0E442, 0x0072B2, 0xBEBEBE])
        let seven = UInt8(1 + 254 * 1000 / 4096)  // the level of 1000 units under a cap of 4096
        let materialIndices: [UInt32] = [(0 * 12 + 1) * 12 + 1, (2 * 12 + 5) * 12 + 7]
        #expect(material.count == 2 && material.indices == materialIndices && material.levels == [255, seven])
        #expect(soil.indices == [3 * 12 + 3] && soil.levels == [255])
        #expect(voice.uid == 10 && voice.w == 3 && voice.heardValid && voice.audible == 2)
        #expect(voice.says == [255, 0, 128] && voice.hears == [10, 20, 30])
    }

    @Test func aStoreHoldsTheLayersAndTheInspectorShowsThem() throws {
        var decoder = FrameDecoder()
        let events = try decoder.feed(Data(try bytes()))
        guard case .header(let header)? = events.first else { Issue.record("no header"); return }
        var store = WorldStore()
        store.reset(header: header)
        for case .record(let record) in events { store.apply(record) }
        #expect(store.layers.structures && store.layers.soil && store.layers.voice && store.layers.bondFlow && store.layers.roles)
        #expect(store.material?.count == 2 && store.soil?.count == 1 && store.materialSerial == 1 && store.soilSerial == 1)
        #expect(store.focusVoice == nil, "the voice is shown for the organism a FOCS follows, and this stream has no FOCS")
    }
}
