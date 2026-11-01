import Foundation
import simd

/// What the camera follows on the planet (`FollowTarget`, DESIGN §33.4.8), resolved against the newest records to a direction of the planet,
/// and the approach of the camera's point of view to it. Pure: the window gives it the time that passed.
package struct SphereFollow: Equatable, Sendable {
    package var target = FollowTarget.none
    /// Where the target last was: the death point of an organism, the table's position of a lineage.
    package private(set) var last: SIMD3<Float>?
    private var hint = 0
    private var slotByUID: [Int64: UInt8] = [:]
    private var slotKey: (UInt64, UInt32)?

    package init(target: FollowTarget = .none) { self.target = target }

    package static func == (a: SphereFollow, b: SphereFollow) -> Bool { a.target == b.target && a.last == b.last }

    package mutating func set(_ target: FollowTarget) {
        self.target = target
        last = nil
        hint = 0
    }

    /// The unit direction of the target now, or nil with nothing to follow. An organism that is gone leaves the camera where it died.
    package mutating func goal(store: WorldStore) -> SIMD3<Float>? {
        switch target {
        case .none: return nil
        case .storm(let id):
            if let storm = store.storms?.storms.storms.first(where: { $0.id == id }), let d = SphereOverlay.direction(of: storm) { last = d }
            return last
        case .organism(let uid):
            if let index = store.listedIndex(ofUID: uid, hint: hint), let frame = store.frame {
                hint = index
                let p = frame.position(index)
                if simd_length(p) > 0 { last = simd_normalize(p) }
            }
            return last
        case .lineage(let id):
            if let d = lineageDirection(id, store: store) { last = d }
            return last
        }
    }

    /// The mean direction of the frame's rows whose `IDNT` slot is the group's (rows born since the last `IDNT` are ignored); the table's
    /// centroid while none is found.
    private mutating func lineageDirection(_ id: Int, store: WorldStore) -> SIMD3<Float>? {
        guard let identities = store.identities, let kin = identities.tables.kin.first(where: { $0.id == id }) else { return nil }
        if let frame = store.frame, kin.slot >= 0, kin.slot < 255 {
            let key = (identities.head.step, identities.head.seq)
            if slotKey == nil || slotKey! != key {
                slotKey = key
                slotByUID = Dictionary(identities.rows.map { ($0.uid, $0.kin) }, uniquingKeysWith: { a, _ in a })
            }
            let wanted = UInt8(kin.slot)
            var sum = SIMD3<Float>.zero
            for i in 0..<frame.n where slotByUID[frame.uid[i]] == wanted {
                let p = frame.position(i)
                let l = simd_length(p)
                if l > 0 { sum += p / l }
            }
            if simd_length(sum) > 1e-4 { return simd_normalize(sum) }
        }
        if let x = kin.x, let y = kin.y, let z = kin.z {
            let v = SIMD3<Float>(Float(x), Float(y), Float(z))
            if simd_length(v) > 0 { return simd_normalize(v) }
        }
        return nil
    }

    /// The approach of the view to a goal direction: the point looked at turns about the centre a fraction `1 - exp(-dt / constant)` of
    /// the way round, the heading carried along (the parallel transport of the grab). True while it has not arrived.
    package static func approach(_ pose: inout SpherePose, toward goal: SIMD3<Float>, distance: Float? = nil, dt: Float,
                                 constant: Float = 0.6) -> Bool {
        let dt = min(max(dt, 0), 0.1)
        let k = 1 - exp(-dt / max(constant, 1e-3))
        let angle = SphereFrame.angle(pose.up, goal)
        if angle > 1e-5 {
            let next = SphereFrame.slerp(pose.up, goal, k)
            pose = pose.rotated(by: SphereFrame.rotation(from: pose.up, to: next))
        }
        var zooming = false
        if let distance {
            let d = exp(log(pose.distance) + (log(max(distance, SpherePose.minDistance)) - log(pose.distance)) * k)
            pose.distance = d
            zooming = abs(d - distance) > 0.004 * distance
        }
        return SphereFrame.angle(pose.up, goal) > 2e-4 || zooming
    }
}
