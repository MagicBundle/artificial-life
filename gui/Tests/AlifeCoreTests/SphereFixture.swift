import Foundation
import Testing
import simd

@testable import AlifeCore

/// The sphere fixtures of `tests/data` (written by `tests/make_sphere_gui_fixture.py`): the protocol 2 play stream and the JSON of what
/// the engine's `alife.geom` and the independent decoder say the Swift code must reproduce.
enum SphereFixture {
    static let stream: [UInt8] = {
        [UInt8]((try? Data(contentsOf: Repo.streamData.appendingPathComponent("play_sphere.bin"))) ?? Data())
    }()

    nonisolated(unsafe) static let expected: [String: Any] = {
        guard let data = try? Data(contentsOf: Repo.streamData.appendingPathComponent("sphere_gui.json")),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [:] }
        return object
    }()

    /// The header and the records of the stream, decoded whole.
    static func decoded() throws -> (header: StreamHeader, records: [StreamRecord]) {
        let result = decode(stream)
        #expect(result.error == nil)
        guard case .header(let header)? = result.events.first else { throw StreamError.badMagic }
        return (header, result.events.compactMap { if case .record(let r) = $0 { r } else { nil } })
    }

    static func store() throws -> WorldStore {
        let (header, records) = try decoded()
        var store = WorldStore()
        store.reset(header: header)
        for record in records { store.apply(record) }
        return store
    }

    static func numbers(_ any: Any?) -> [Double] { (any as? [Any])?.compactMap { ($0 as? NSNumber)?.doubleValue } ?? [] }
    static func vector(_ any: Any?) -> SIMD3<Double> {
        let v = numbers(any)
        return SIMD3(v[0], v[1], v[2])
    }
}

extension StreamRecord {
    /// The first record of `tag` in a list.
    static func first(_ tag: String, in records: [StreamRecord]) -> StreamRecord? { records.first { $0.tag == tag } }
}
