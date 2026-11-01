import CoreGraphics
import Foundation
import simd

/// What is under the pointer on the planet (DESIGN §46.8): an organism, a storm, a warm pool, a territory, or the bare floor cell.
package enum SpherePick: Equatable, Sendable {
    /// A listed row of the newest frame.
    case organism(uid: Int64, index: Int)
    case storm(id: Int)
    case pool(index: Int)
    /// A territory pixel with an owner: the pixel of the territory map and the owner's kin group id.
    case territory(pixel: Int, kin: Int)
    /// The floor cell under the pointer (a place), as a field pixel and the unit direction of the hit.
    case cell(pixel: Int, direction: SIMD3<Float>)
    case none

    package var isNone: Bool { self == .none }

    /// The place of a pick on the floor, if it has one: where a power aims, what the readout names.
    package var direction: SIMD3<Float>? {
        if case .cell(_, let d) = self { return d }
        return nil
    }
}

/// The client's hit test on the planet: pure, in the view's points (origin at the top left). A ray through the pointer goes against the
/// floor sphere (the planet), and the organisms are the nearest to the ray among those the planet does not hide. The tolerances are in
/// points, so they are the same at any backing scale.
///
/// * **Organisms**, from the region level in (at the planet level they are tiny symbols, tens of thousands under a pointer): where the drawn
///   radius is under the tolerance (6 pt) the winner is the row nearest the pointer on the screen, ties by depth; where it is above, the row
///   nearest the eye among those whose disc holds the pointer.
/// * Then **storms** (inside the ring, or 10 pt around a small one), **warm pools**, then a **territory** pixel with an owner, then the
///   floor **cell**.
package enum SpherePicker {
    package static let tolerance: CGFloat = 6
    package static let stormTolerance: CGFloat = 10

    /// The ndc (x right, y up, in [-1, 1]) of a point of the view in points, origin at the top left.
    package static func ndc(of point: CGPoint, in view: CGSize) -> SIMD2<Float> {
        SIMD2(Float(point.x / max(view.width, 1)) * 2 - 1, 1 - Float(point.y / max(view.height, 1)) * 2)
    }

    /// The point of the view in points for ndc.
    package static func point(ofNDC ndc: SIMD2<Float>, in view: CGSize) -> CGPoint {
        CGPoint(x: CGFloat(ndc.x * 0.5 + 0.5) * view.width, y: CGFloat(0.5 - ndc.y * 0.5) * view.height)
    }

    /// The view point where a planet-frame point is drawn; nil behind the eye.
    package static func screen(_ p: SIMD3<Float>, camera: SphereCamera, view: CGSize) -> CGPoint? {
        guard view.width >= 1, view.height >= 1 else { return nil }
        let aspect = Float(view.width / view.height)
        return camera.ndc(of: p, aspect: aspect).map { point(ofNDC: $0, in: view) }
    }

    /// Points per world unit across the view at distance `depth` along the view axis.
    package static func pointsPerUnit(depth: Float, camera: SphereCamera, view: CGSize) -> Float {
        Float(view.height) / (2 * max(depth, 1e-3) * tan(camera.pose.fovY / 2))
    }

    package static func pick(at point: CGPoint, view: CGSize, camera: SphereCamera, store: WorldStore, display: WorldDisplay,
                             level: SphereLevel, maxRadius: Float = .infinity) -> SpherePick {
        guard view.width >= 1, view.height >= 1, let sphere = store.header?.sphere else { return .none }
        let aspect = Float(view.width / view.height)
        if display.agents, level != .planet, let frame = store.frame,
           let organism = organism(at: point, view: view, camera: camera, frame: frame, bodyRadius: Float(store.header?.radius ?? 1),
                                   maxRadius: maxRadius) {
            return organism
        }
        let ndc = ndc(of: point, in: view)
        guard let g = camera.floorDirection(ndc: ndc, aspect: aspect) else { return .none }
        let rp = Float(sphere.planetRadius)
        let pointsPerArc = pointsPerUnit(depth: simd_length(camera.eye - g * rp), camera: camera, view: view)
        if let storms = store.storms?.storms.storms {
            var best: (id: Int, score: Float)?
            for storm in storms {
                guard let z = storm.z else { continue }
                let centre = simd_normalize(SIMD3<Float>(Float(storm.x), Float(storm.y), Float(z)))
                let arc = SphereFrame.angle(g, centre) * rp
                let ring = Float(storm.r)
                let reach = max(ring, Float(stormTolerance) / max(pointsPerArc, 1e-6))
                guard arc <= reach else { continue }
                let score = arc / max(ring, 1e-3)
                if best == nil || score < best!.score { best = (storm.id, score) }
            }
            if let best { return .storm(id: best.id) }
        }
        if let pools = store.environment?.environment.pools {
            var best: (index: Int, arc: Float)?
            for (i, pool) in pools.enumerated() {
                guard let z = pool.z else { continue }
                let centre = simd_normalize(SIMD3<Float>(Float(pool.x), Float(pool.y), Float(z)))
                let arc = SphereFrame.angle(g, centre) * rp
                guard arc <= 0.8 * Float(max(pool.sx, pool.sy)) else { continue }
                if best == nil || arc < best!.arc { best = (i, arc) }
            }
            if let best { return .pool(index: best.index) }
        }
        if display.territory != .off, let terr = store.territory, let nside = terr.nside {
            let pixel = Healpix.pixelFloat(of: g, nside: nside)
            let slot = Int(terr.owner[pixel])
            if slot < terr.ids.count, terr.ids[slot] >= 0 { return .territory(pixel: pixel, kin: terr.ids[slot]) }
        }
        return .cell(pixel: Healpix.pixelFloat(of: g, nside: sphere.nsideField), direction: g)
    }

    /// The organism under the pointer (not the planet's hidden side), nil if none.
    package static func organism(at point: CGPoint, view: CGSize, camera: SphereCamera, frame: PlayFrame, bodyRadius: Float,
                                maxRadius: Float = .infinity) -> SpherePick? {
        let aspect = Float(view.width / view.height)
        let vp = camera.viewProjectionMatrix(aspect: aspect)
        let forward = camera.forward, eye = camera.eye
        var best: (index: Int, score: CGFloat, depth: Float)?
        for i in 0..<frame.n {
            let p = frame.position(i)
            let clip = vp * SIMD4(p, 1)
            guard clip.w > 0 else { continue }
            let depth = simd_dot(p - eye, forward)
            guard depth > 0 else { continue }
            let screen = CGPoint(x: CGFloat(clip.x / clip.w * 0.5 + 0.5) * view.width, y: CGFloat(0.5 - clip.y / clip.w * 0.5) * view.height)
            let radius = min(CGFloat(bodyRadius * pointsPerUnit(depth: depth, camera: camera, view: view)), CGFloat(maxRadius))
            let reach = max(radius, tolerance)
            let d = hypot(screen.x - point.x, screen.y - point.y)
            guard d <= reach else { continue }
            // the planet hides what is behind its limb
            if camera.isOccluded(p) { continue }
            // a disc that holds the pointer beats a near miss on a small one; then the nearer to the eye
            let score = radius >= tolerance ? d - radius : d
            if let b = best {
                let holds = radius >= tolerance, bestHolds = b.score < 0
                if holds != bestHolds ? holds : (holds ? depth < b.depth : score < b.score) { best = (i, score, depth) }
            } else {
                best = (i, score, depth)
            }
        }
        return best.map { SpherePick.organism(uid: frame.uid[$0.index], index: $0.index) }
    }

    // MARK: What the tooltip says

    package static func describe(_ pick: SpherePick, store: WorldStore) -> String? {
        switch pick {
        case .none: return nil
        case .organism(let uid, _): return WorldPicker.describe(.organism(uid: uid, index: 0), store: store)
        case .storm(let id): return WorldPicker.describe(.storm(id: id), store: store)
        case .pool: return "warm pool"
        case .territory(_, let kin): return WorldPicker.describe(.territory(cell: 0, kin: kin), store: store)
        case .cell(_, let d): return SphereFrame.placeText(d)
        }
    }
}
