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

extension Camera {
    /// The pose v1 started a world at, kept for the tests of the shaders that want the whole world centred in view: the 3D world seen
    /// from above (x right, y up) and the 2D world face on (y down), the whole world in the smaller dimension of a window of `aspect`.
    /// It is not the Home view (`WorldPose.home`, one world wide): the map's patch does not reach the corners of this one.
    static func world(for header: StreamHeader, aspect: Float = 1) -> Camera {
        let size = Float(header.size)
        var camera = header.dims == 3
            ? Camera.overview(size: size, height: Float(header.height), aspect: aspect)
            : Camera.topDown(size: size, aspect: aspect)
        if header.dims == 3 {
            let halfY = atan(min(1, aspect) * tan(camera.fovY / 2))
            camera = Camera(target: SIMD3(size / 2, size / 2, 0), distance: size / 2 * 1.04 / tan(halfY), yaw: .pi / 2, pitch: -Camera.maxPitch,
                            fovY: camera.fovY, near: 0.1, far: 8 * max(size, Float(header.height)))
        }
        camera.scene = Camera.patchScene(center: SIMD2(size / 2, size / 2), size: size, height: Float(header.height))
        return camera
    }
}
