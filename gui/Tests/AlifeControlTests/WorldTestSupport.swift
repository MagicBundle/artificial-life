import AlifeCore
import Foundation
import simd

/// Frames built by hand for the tests of the World model (the fixtures' own are in `tests/data`).
enum TestFrames {
    /// A frame of the listed rows at `positions`, with uids `uids` (`first + i` by default).
    static func frame(_ positions: [SIMD3<Float>], step: UInt64 = 10, uids: [Int64]? = nil, first: Int64 = 1000) -> PlayFrame {
        let n = positions.count
        var bitmap = [UInt32](repeating: 0, count: max(1, (n + 31) / 32))
        for r in 0..<n { bitmap[r >> 5] |= 1 << UInt32(r & 31) }
        return PlayFrame(
            head: RecordHead(step: step, seq: 0, version: 1, flags: 0), n: n, capacity: max(n, 32), colorIndex: 0, span: 0,
            bitmap: bitmap, rows: (0..<n).map { UInt32($0) }, pos: positions.flatMap { [$0.x, $0.y, $0.z] },
            orient: [UInt16](repeating: 0, count: 2 * n), rgb: [UInt32](repeating: 0xFFFFFF, count: n),
            uid: uids ?? (0..<n).map { first + Int64($0) }, bonds: [])
    }
}
