import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeCore

/// Stores, frames and cameras for the interaction tests: the committed play fixtures, and frames built by hand.
enum WorldFixtures {
    /// The uid base of the fixture's organisms (`tests/play_fixtures.py`: `UID0`).
    static let uid0: Int64 = 5_000_000_000

    /// The store a client holds after the whole fixture (`3d`: L = 48, height 24, 7 alive rows at step 104).
    static func store(_ kind: String = "3d") throws -> WorldStore {
        let events = try PlayFixture(kind: kind).events()
        guard case .header(let header)? = events.first else { throw StreamError.badMagic }
        var store = WorldStore()
        store.reset(header: header)
        for case .record(let record) in events { store.apply(record) }
        return store
    }

    /// A frame of the listed rows at `positions` (uids `first + i`), nothing else filled in.
    static func frame(_ positions: [SIMD3<Float>], first: Int64 = 1000, step: UInt64 = 10, uids: [Int64]? = nil) -> PlayFrame {
        let n = positions.count
        let words = max(1, (n + 31) / 32)
        var bitmap = [UInt32](repeating: 0, count: words)
        for r in 0..<n { bitmap[r >> 5] |= 1 << UInt32(r & 31) }
        return PlayFrame(
            head: RecordHead(step: step, seq: 0, version: 1, flags: 0), n: n, capacity: max(n, 32), colorIndex: 0, span: 0,
            bitmap: bitmap, rows: (0..<n).map { UInt32($0) }, pos: positions.flatMap { [$0.x, $0.y, $0.z] },
            orient: [UInt16](repeating: 0, count: 2 * n), rgb: [UInt32](repeating: 0xFFFFFF, count: n),
            uid: uids ?? (0..<n).map { first + Int64($0) }, bonds: [])
    }

    /// The fixture's world with its radius and frame replaced: organisms where the test wants them. With `layers` it also
    /// holds the fixture's other records (identities, territory, storms, pools, focus, clock); without, only the frame, so a
    /// pick that finds nothing means nothing and not the territory under it.
    static func store(radius: Double? = nil, positions: [SIMD3<Float>], uids: [Int64]? = nil, layers: Bool = false,
                      kind: String = "3d") throws -> WorldStore {
        let source = try store(kind)
        var header = try #require(source.header)
        if let radius { header.radius = radius }
        var out = WorldStore()
        out.reset(header: header)
        if layers {
            for record in [source.identities.map(StreamRecord.idnt), source.territory.map(StreamRecord.terr),
                           source.storms.map(StreamRecord.strm), source.environment.map(StreamRecord.envs),
                           source.focus.map(StreamRecord.focs), source.clock.map(StreamRecord.clck)].compactMap({ $0 }) {
                out.apply(record)
            }
        }
        out.apply(.frma(frame(positions, uids: uids)))
        return out
    }

    /// The planet pose at a given distance and target, on a view of `view` points.
    static func projector(size: Float = 48, height: Float = 24, target: SIMD3<Float>? = nil, distance: Float, view: CGSize = CGSize(width: 900, height: 700)) -> WorldProjector {
        var camera = Camera.planet(size: size, height: height, aspect: Float(view.width / view.height))
        if let target { camera.target = target }
        camera.distance = distance
        return WorldProjector(camera: camera, size: size, view: view)
    }
}
