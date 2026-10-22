import Foundation
import simd

// The camera that follows something (DESIGN §33.4.8): what it follows (`FollowTarget`), where that is (`FollowState`, which
// resolves a target against the newest records), and how the camera gets there (`CameraGlide`, a critically damped approach
// to the nearest image of the goal). Pure: the window gives it the time that passed.

/// What the camera follows. Any manual camera drag cancels it.
package enum FollowTarget: Equatable, Sendable {
    case none
    /// A storm, by id: the position of its newest `STRM` entry.
    case storm(id: Int)
    /// A kin group, by id: the centroid of its rows.
    case lineage(kinID: Int)
    /// An organism, by uid: its row of the newest frame. At its death the camera holds the point it died at until the hand-off
    /// (or the player) says where to go.
    case organism(uid: Int64)

    package var isNone: Bool { self == .none }
}

/// Resolves a `FollowTarget` to a point of the floor or the volume against the world's newest records.
package struct FollowState: Equatable, Sendable {
    /// The centroid of a lineage is recomputed every this many frames.
    package static let centroidEvery = 4

    package var target = FollowTarget.none
    /// Where the target last was: the death point of an organism, the table's position of a lineage.
    package private(set) var lastPoint: SIMD3<Float>?
    private var hint = 0
    private var centroid: SIMD3<Float>?
    private var centroidSerial = Int.min
    private var slotByUID: [Int64: UInt8] = [:]
    private var slotKey: (UInt64, UInt32)?

    package init(target: FollowTarget = .none) { self.target = target }

    package static func == (a: FollowState, b: FollowState) -> Bool { a.target == b.target && a.lastPoint == b.lastPoint }

    package mutating func set(_ target: FollowTarget) {
        self.target = target
        lastPoint = nil
        hint = 0
        centroid = nil
        centroidSerial = Int.min
    }

    /// The point to follow now, or nil with nothing to follow. `size` is `L`.
    package mutating func goal(store: WorldStore, size: Float) -> SIMD3<Float>? {
        switch target {
        case .none: return nil
        case .storm(let id):
            if let storm = store.storms?.storms.storms.first(where: { $0.id == id }) {
                lastPoint = SIMD3(Float(storm.x), Float(storm.y), 0)
            }
            return lastPoint
        case .organism(let uid):
            if let index = store.listedIndex(ofUID: uid, hint: hint), let frame = store.frame {
                hint = index
                lastPoint = frame.position(index)
            }
            return lastPoint
        case .lineage(let id):
            if centroidSerial == Int.min || store.frameSerial - centroidSerial >= Self.centroidEvery {
                centroidSerial = store.frameSerial
                if let point = lineageCentroid(id, store: store, size: size) { centroid = point }
            }
            if let table = tablePosition(of: id, store: store), centroid == nil { return table }
            return centroid ?? lastPoint
        }
    }

    /// The circular mean over the torus of the frame's rows whose `IDNT` slot is the group's (rows born since the last
    /// `IDNT` are not in it and are ignored), with the mean height; nil when none is found.
    private mutating func lineageCentroid(_ id: Int, store: WorldStore, size: Float) -> SIMD3<Float>? {
        guard let identities = store.identities, let frame = store.frame,
              let slot = identities.tables.kin.first(where: { $0.id == id })?.slot, slot >= 0, slot < 255
        else { return nil }
        let key = (identities.head.step, identities.head.seq)
        if slotKey == nil || slotKey! != key {
            slotKey = key
            slotByUID = Dictionary(identities.rows.map { ($0.uid, $0.kin) }, uniquingKeysWith: { a, _ in a })
        }
        let wanted = UInt8(slot)
        let k = 2 * Float.pi / size
        var sx: Float = 0, cx: Float = 0, sy: Float = 0, cy: Float = 0, z: Float = 0
        var n = 0
        for i in 0..<frame.n where slotByUID[frame.uid[i]] == wanted {
            let p = frame.position(i)
            sx += sin(p.x * k); cx += cos(p.x * k)
            sy += sin(p.y * k); cy += cos(p.y * k)
            z += p.z
            n += 1
        }
        guard n > 0 else { return nil }
        return SIMD3(Torus.wrap(atan2(sx, cx) / k, size: size), Torus.wrap(atan2(sy, cy) / k, size: size), z / Float(n))
    }

    /// The group's position in the `IDNT` table (the fallback while its rows are not found).
    private func tablePosition(of id: Int, store: WorldStore) -> SIMD3<Float>? {
        guard let kin = store.identities?.tables.kin.first(where: { $0.id == id }), let x = kin.x, let y = kin.y else { return nil }
        return SIMD3(Float(x), Float(y), Float(kin.z ?? 0))
    }
}

/// The approach of the camera's target (and distance) to a goal: critically damped, so it never overshoots, trails a goal that moves
/// steadily by twice its speed times the constant (two lags in series), and is exact for any step of time. The offset of the target from the goal is
/// taken at the **nearest image**, so a goal across the seam is approached the short way and the camera never whips.
package struct CameraGlide: Equatable, Sendable {
    /// Follow: the spec's one second. Fly: a quicker approach to a new place.
    package static let followConstant: Float = 1.0
    package static let flyConstant: Float = 0.35
    /// Under these the approach has arrived: units, units per second, and the relative error of the distance.
    package static let arrivedDistance: Float = 0.02
    package static let arrivedSpeed: Float = 0.05
    package static let arrivedZoom: Float = 0.002

    package var velocity = SIMD3<Float>.zero
    package var zoomVelocity: Float = 0
    /// The time constant in seconds: the rate of approach is `1 / timeConstant`.
    package var timeConstant: Float = CameraGlide.followConstant

    package init() {}

    /// The state `(offset e, velocity v)` after `dt` seconds of a critically damped approach with rate `w`:
    /// `e' = (e + (v + w e) t) exp(-w t)`, `v' = (v - w (v + w e) t) exp(-w t)`.
    package static func advance(offset e: Float, velocity v: Float, rate w: Float, dt: Float) -> (offset: Float, velocity: Float) {
        let decay = exp(-w * dt)
        let c = v + w * e
        return ((e + c * dt) * decay, (v - w * c * dt) * decay)
    }

    /// Moves `camera` towards `goal` (and its distance towards `distance`, if given) for `dt` seconds on a torus of side
    /// `size`. True while it has not arrived. The target is not wrapped here (the caller wraps it into `[0, size)`).
    @discardableResult
    package mutating func advance(_ camera: inout Camera, toward goal: SIMD3<Float>, distance: Float?, dt: Float, size: Float) -> Bool {
        let dt = min(max(dt, 0), 0.1)
        let w = 1 / max(timeConstant, 1e-3)
        let offset = SIMD3(
            Torus.wrapDelta(camera.target.x - goal.x, size: size), Torus.wrapDelta(camera.target.y - goal.y, size: size),
            camera.target.z - goal.z)
        var next = SIMD3<Float>.zero
        for axis in 0..<3 {
            let r = Self.advance(offset: offset[axis], velocity: velocity[axis], rate: w, dt: dt)
            next[axis] = r.offset
            velocity[axis] = r.velocity
        }
        camera.target = goal + next
        var zoomArrived = true
        if let distance {
            let r = Self.advance(offset: camera.distance - distance, velocity: zoomVelocity, rate: w, dt: dt)
            camera.distance = max(distance + r.offset, Camera.minDistance)
            zoomVelocity = r.velocity
            zoomArrived = abs(r.offset) <= Self.arrivedZoom * max(distance, 1) && abs(r.velocity) <= Self.arrivedSpeed
        } else {
            zoomVelocity = 0
        }
        return !(simd_length(next) <= Self.arrivedDistance && simd_length(velocity) <= Self.arrivedSpeed && zoomArrived)
    }

    package mutating func stop() {
        velocity = .zero
        zoomVelocity = 0
    }
}
