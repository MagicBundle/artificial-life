import CoreGraphics
import Foundation
import simd

/// What is under the pointer (DESIGN §33.4.8).
package enum WorldPick: Equatable, Sendable {
    /// A listed row of the newest frame.
    case organism(uid: Int64, index: Int)
    case storm(id: Int)
    /// A warm pool, by its index in `ENVS.pools`.
    case pool(index: Int)
    /// A `TERR` cell with an owner: the cell (`y * nx + x`) and the owner's kin group id.
    case territory(cell: Int, kin: Int)
    case none

    package var isNone: Bool { self == .none }
}

/// The client's hit test: pure, in the view's points (origin at the top left), over the newest frame, the storms, the pools and
/// the territory, with every point at the minimum image of the camera target. The pointer is in points and so is the view; the
/// drawable's pixels do not come into it (a Retina drawable has twice the pixels of the same rectangle), so the tolerances are
/// the same number of points at any backing scale.
///
/// - **Organisms** (region and organism levels, and only while they are drawn, `display.agents`): where the projected radius (the
///   drawn one: capped by the size policy) is under the tolerance (6 pt), the winner is the row nearest the pointer on the screen,
///   ties by depth; where it is above, the row nearest the eye among those whose disc holds the pointer. A hit on a disc beats a
///   near miss on a small one. At the planet level organisms are not picked: they are small symbols, and 40,000 of them are under
///   every pointer.
/// - Then **storms** (inside the ring, or 10 pt around a small one: the one the pointer is nearest to in ring radii), then
///   **warm pools**, then a **territory** cell with an owner, found by casting the pointer's ray onto the floor.
package enum WorldPicker {
    /// The pick radius of a small organism, in points.
    package static let tolerance: CGFloat = 6
    /// A storm ring smaller than this (in points) is picked within this radius.
    package static let stormTolerance: CGFloat = 10

    package static func pick(at point: CGPoint, projector: WorldProjector, store: WorldStore, display: WorldDisplay = WorldDisplay(),
                             maxRadius: Float = .infinity) -> WorldPick {
        guard projector.view.width >= 1, projector.view.height >= 1 else { return .none }
        if projector.level != .planet, display.agents,
           let hit = organism(at: point, projector: projector, store: store, maxRadius: maxRadius) {
            return hit
        }
        if let storms = store.storms?.storms, let id = storm(at: point, projector: projector, storms: storms) {
            return .storm(id: id)
        }
        if let environment = store.environment?.environment, let index = pool(at: point, projector: projector, environment: environment) {
            return .pool(index: index)
        }
        if display.territory != .off, let territory = store.territory, let hit = cell(at: point, projector: projector, territory: territory) {
            return hit
        }
        return .none
    }

    // MARK: Organisms

    /// The organism under `point` by the rule above; nil if there is none. `maxRadius` is the most, in points, an organism is
    /// drawn with (the size policy's cap, §36.7): the projected radius is the drawn one, so a capped symbol is not picked by its
    /// physical disc.
    package static func organism(at point: CGPoint, projector: WorldProjector, store: WorldStore,
                                 maxRadius: Float = .infinity) -> WorldPick? {
        guard let frame = store.frame, frame.n > 0 else { return nil }
        let camera = projector.camera
        let aspect = projector.aspect
        let matrix = camera.viewProjectionMatrix(aspect: aspect)
        let width = Float(projector.view.width), height = Float(projector.view.height)
        let px = Float(point.x), py = Float(point.y)
        // A length r at depth w covers r / (w tan(fov/2)) of the half height of the view.
        let perUnit = Float(0.5) * height / tan(camera.fovY / 2)
        let radius = Float(store.header?.radius ?? 1)
        let tol = Float(tolerance)
        let size = projector.size
        let target = camera.target

        var disc: (index: Int, depth: Float)?
        var near: (index: Int, distance: Float, depth: Float)?
        for i in 0..<frame.n {
            let p = frame.position(i)
            let xy = Torus.nearestImage(of: SIMD2(p.x, p.y), to: SIMD2(target.x, target.y), size: size)
            let clip = matrix * SIMD4(xy.x, xy.y, p.z, 1)
            guard clip.w > 0 else { continue }
            let sx = (clip.x / clip.w * 0.5 + 0.5) * width
            let sy = (0.5 - clip.y / clip.w * 0.5) * height
            let dx = sx - px, dy = sy - py
            let projected = min(radius * perUnit / clip.w, maxRadius)
            let reach = max(tol, projected)
            let d2 = dx * dx + dy * dy
            if d2 > reach * reach { continue }
            if projected >= tol {
                // The pointer is on a disc big enough to see: the nearest to the eye wins.
                if disc == nil || clip.w < disc!.depth { disc = (i, clip.w) }
            } else {
                let d = d2.squareRoot()
                if near == nil || d < near!.distance || (d == near!.distance && clip.w < near!.depth) { near = (i, d, clip.w) }
            }
        }
        guard let index = disc?.index ?? near?.index else { return nil }
        return .organism(uid: frame.uid[index], index: index)
    }

    // MARK: Storms and pools

    package static func storm(at point: CGPoint, projector: WorldProjector, storms: PlayStorms) -> Int? {
        var best: (id: Int, score: CGFloat)?
        for glyph in projector.glyphs(of: storms) {
            // The ring is a circle on the floor, drawn as an ellipse flattened by the view: so is the target (at least a tolerance tall).
            let reach = max(glyph.radius, stormTolerance)
            let reachY = max(reach * glyph.flatten, min(reach, stormTolerance))
            let dx = (glyph.center.x - point.x) / reach, dy = (glyph.center.y - point.y) / reachY
            let score = hypot(dx, dy)
            guard score <= 1 else { continue }
            if best == nil || score < best!.score { best = (glyph.id, score) }
        }
        return best?.id
    }

    package static func pool(at point: CGPoint, projector: WorldProjector, environment: PlayEnvironment) -> Int? {
        var best: (index: Int, distance: CGFloat)?
        for (index, pool) in environment.pools.enumerated() {
            let at = SIMD3<Float>(Float(pool.x), Float(pool.y), 0)
            guard let center = projector.screen(at) else { continue }
            let radius = max(projector.points(ofUnits: Float(max(pool.sx, pool.sy)) * 2, at: at) ?? 0, 6)
            let d = hypot(center.x - point.x, center.y - point.y)
            if d <= radius, best == nil || d < best!.distance { best = (index, d) }
        }
        return best?.index
    }

    // MARK: Territory

    /// The point of the floor (z = 0) under `point`, unwrapped as the ray meets it; nil when the ray does not meet the floor
    /// in front of the eye, and where the floor is not read (farther than `WorldFog.readReach` of the point looked at: in the haze, a
    /// cell of the territory is not under the pointer).
    package static func floorPoint(at point: CGPoint, projector: WorldProjector) -> SIMD2<Float>? {
        let camera = projector.camera
        let nx = Float(point.x / projector.view.width) * 2 - 1
        let ny = 1 - Float(point.y / projector.view.height) * 2
        let half = tan(camera.fovY / 2)
        let direction = camera.forward + camera.right * (nx * half * projector.aspect) + camera.up * (ny * half)
        let eye = camera.position
        guard abs(direction.z) > 1e-6 else { return nil }
        let t = -eye.z / direction.z
        guard t > 0, t.isFinite else { return nil }
        let hit = eye + direction * t
        if projector.size > 0,
           !WorldFog.isReadable(offset: SIMD2(hit.x - camera.target.x, hit.y - camera.target.y), size: projector.size) { return nil }
        return SIMD2(hit.x, hit.y)
    }

    /// The `TERR` cell under `point` if it has an owner.
    package static func cell(at point: CGPoint, projector: WorldProjector, territory: PlayTerritory) -> WorldPick? {
        guard let floor = floorPoint(at: point, projector: projector) else { return nil }
        let size = projector.size
        let wrapped = Torus.wrap(floor, size: size)
        let cx = min(max(Int(wrapped.x / size * Float(territory.nx)), 0), territory.nx - 1)
        let cy = min(max(Int(wrapped.y / size * Float(territory.ny)), 0), territory.ny - 1)
        let index = cy * territory.nx + cx
        guard territory.owner.indices.contains(index) else { return nil }
        let slot = Int(territory.owner[index])
        guard slot < territory.ids.count else { return nil }
        return .territory(cell: index, kin: territory.ids[slot])
    }

    // MARK: What the tooltip says

    /// One line for a hover: nothing for `.none`.
    package static func describe(_ pick: WorldPick, store: WorldStore) -> String? {
        switch pick {
        case .none: return nil
        case .organism(let uid, _):
            guard let row = store.identity(ofUID: uid), let tables = store.identities?.tables else {
                return "organism \(uid) (born since the last snapshot)"
            }
            let kin = Names.kin(slot: row.kin, tables: tables)
            let house = Names.house(slot: row.house, tables: tables)
            return "organism \(uid) · \(kin) · House \(house)"
        case .storm(let id):
            guard let storm = store.storms?.storms.storms.first(where: { $0.id == id }) else { return "storm \(id)" }
            return Names.storm(storm)
        case .pool: return "warm pool"
        case .territory(_, let kin):
            let name = store.identities?.tables.kin.first { $0.id == kin }?.name ?? "kin group \(kin)"
            return "\(name) territory (dominance)"
        }
    }
}

/// The names of the things the World lists.
package enum Names {
    /// A storm's title: its name, or "disturbance N" below named strength.
    package static func storm(_ storm: PlayStorms.Storm) -> String {
        storm.name ?? "disturbance \(storm.id)"
    }

    /// The name of the kin group in `slot` of the table; the reserved slots in words.
    package static func kin(slot: UInt8, tables: IdentityTables) -> String {
        if let entry = tables.kin.first(where: { $0.slot == Int(slot) }) { return entry.name }
        return reserved(slot, noun: "kin group")
    }

    package static func house(slot: UInt8, tables: IdentityTables) -> String {
        if let entry = tables.houses.first(where: { $0.slot == Int(slot) }) { return entry.name }
        return reserved(slot, noun: "house")
    }

    private static func reserved(_ slot: UInt8, noun: String) -> String {
        switch slot {
        case Slot.unknown: return "unknown lineage"
        case Slot.other: return "other \(noun)s"
        case Slot.none: return "no \(noun)"
        default: return "\(noun) \(slot)"
        }
    }

    /// The display name of a label byte (`PlayHeader.labels`); "none" for 0 or a byte past the list.
    package static func label(_ byte: UInt8, in names: [String]) -> String {
        let i = Int(byte)
        guard i > 0, i < names.count, !names[i].isEmpty else { return "none" }
        return names[i]
    }
}
