import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeCore

/// The hit test, on a known camera (the planet pose over the fixture world, L = 48): what is under the pointer at each level.
@Suite struct WorldPickerTests {
    private let uid0 = WorldFixtures.uid0

    /// A region-level view (visible half extent 18 of L/2 = 24; 25 points per unit) centred on the middle of the world.
    private func region(_ target: SIMD3<Float> = SIMD3(24, 24, 0)) -> WorldProjector {
        WorldFixtures.projector(target: target, distance: 30)
    }

    /// The point of the view where `world` is drawn.
    private func screen(_ p: WorldProjector, _ world: SIMD3<Float>) throws -> CGPoint {
        try #require(p.screen(world))
    }

    private func pick(_ at: CGPoint, _ p: WorldProjector, _ store: WorldStore, display: WorldDisplay = WorldDisplay()) -> WorldPick {
        WorldPicker.pick(at: at, projector: p, store: store, display: display)
    }

    // MARK: Organisms

    @Test func theViewsLevelsAreWhatTheseTestsAssume() {
        #expect(region().level == .region)
        #expect(WorldFixtures.projector(distance: 70).level == .planet)
        #expect(WorldFixtures.projector(target: SIMD3(24, 24, 0), distance: 4).level == .organism)
    }

    @Test func aClickOnAnOrganismPicksItByUid() throws {
        let store = try WorldFixtures.store()
        let p = region()
        let frame = try #require(store.frame)
        // Row 4 of the newest frame is listed index 4 (rows 0 to 5 are alive, then 21).
        // Rows 0 to 5 and 21 are alive: listed index 6 is row 21, alone in its corner of the world.
        let uid = frame.uid[6]
        #expect(uid == uid0 + 147)
        let at = try screen(p, frame.position(6))
        #expect(pick(at, p, store) == .organism(uid: uid, index: 6))
    }

    @Test func atTheRegionLevelTheRowNearestThePointerWins() throws {
        // Radius 0.1 is 2.5 points: a small organism, picked within 6 points, the nearest one.
        let store = try WorldFixtures.store(radius: 0.1, positions: [SIMD3(24.2, 24, 0), SIMD3(23.9, 24, 0), SIMD3(24, 24.32, 0)])
        let p = region()
        let centre = try screen(p, SIMD3(24, 24, 0))
        #expect(pick(centre, p, store) == .organism(uid: 1001, index: 1), "2.5 points away beats 5 and 8")
        let right = CGPoint(x: centre.x + 4.9, y: centre.y)
        #expect(pick(right, p, store) == .organism(uid: 1000, index: 0))
        // 8 points from the third and from nothing else near: outside the tolerance of 6.
        let far = CGPoint(x: centre.x, y: centre.y - 22)
        #expect(pick(far, p, store) == .none, "nothing within 6 points")
    }

    @Test func theToleranceIsInPointsWhateverTheBackingScale() throws {
        // The pointer and the view are in points; a Retina drawable has twice the pixels of the same rectangle and the
        // same hit test. 5.5 points away hits, 6.5 does not, at any scale.
        let store = try WorldFixtures.store(radius: 0.1, positions: [SIMD3(24, 24, 0)])
        let p = region()
        let centre = try screen(p, SIMD3(24, 24, 0))
        #expect(pick(CGPoint(x: centre.x + 5.5, y: centre.y), p, store) != .none)
        #expect(pick(CGPoint(x: centre.x + 6.5, y: centre.y), p, store) == .none)
        for scale in [1.0, 2.0, 3.0] {
            let pixel = CGPoint(x: (centre.x + 5.5) * scale, y: centre.y * scale)  // what a drawable would report
            let points = CGPoint(x: pixel.x / scale, y: pixel.y / scale)
            #expect(pick(points, p, store) != .none, "scale \(scale)")
        }
    }

    @Test func aBigDiscIsHitInsideItsOutlineAndTheNearestToTheEyeWins() throws {
        // Radius 1 is 25 points at this zoom: the disc rule. Two discs hold the pointer; the one nearer the eye (higher z in
        // the top-down view) wins even though the other is centred under the pointer.
        let store = try WorldFixtures.store(positions: [SIMD3(24, 24, 1), SIMD3(24.5, 24, 5)])
        let p = region()
        let at = try screen(p, SIMD3(24, 24, 1))
        #expect(pick(at, p, store) == .organism(uid: 1001, index: 1))
        // Outside both discs: nothing, though a small organism would still have been hit there.
        let out = CGPoint(x: at.x - 40, y: at.y)
        #expect(pick(out, p, store) == .none)
    }

    @Test func aCappedOrganismIsPickedByItsDrawnRadiusNotItsPhysicalOne() throws {
        // Radius 1 is 25 points here; the size policy draws it with 4.5 pt. 20 points from its centre is inside the physical disc and
        // outside what is drawn (and outside the 6 pt tolerance): a click there is on nothing.
        let store = try WorldFixtures.store(positions: [SIMD3(24, 24, 1)])
        let p = region()
        let at = try screen(p, SIMD3(24, 24, 1))
        let beside = CGPoint(x: at.x + 20, y: at.y)
        #expect(pick(beside, p, store) == .organism(uid: 1000, index: 0), "physical sizes: inside the disc")
        let capped = WorldPicker.pick(at: beside, projector: p, store: store, display: WorldDisplay(), maxRadius: 4.5)
        #expect(capped == .none)
        #expect(WorldPicker.pick(at: at, projector: p, store: store, display: WorldDisplay(), maxRadius: 4.5) == .organism(uid: 1000, index: 0))
        // Within the 6 pt tolerance of a capped symbol: hit (the larger of 6 pt and the drawn radius).
        let near = CGPoint(x: at.x + 5.5, y: at.y)
        #expect(WorldPicker.pick(at: near, projector: p, store: store, display: WorldDisplay(), maxRadius: 4.5) == .organism(uid: 1000, index: 0))
    }

    @Test func equalDistancesAreBrokenByDepth() throws {
        let store = try WorldFixtures.store(radius: 0.1, positions: [SIMD3(24, 24, 1), SIMD3(24, 24, 6), SIMD3(24, 24, 3)])
        let p = region()
        let centre = try screen(p, SIMD3(24, 24, 0))
        #expect(pick(centre, p, store) == .organism(uid: 1001, index: 1), "the highest is nearest the eye")
    }

    @Test func aRowAcrossTheSeamIsPickedAtItsMinimumImage() throws {
        // The target is just inside the near edge; the organism just inside the far edge is drawn two units to its left.
        let store = try WorldFixtures.store(radius: 0.1, positions: [SIMD3(47.5, 24, 0)])
        let p = region(SIMD3(0.5, 24, 0))
        let at = try screen(p, SIMD3(47.5, 24, 0))
        let centre = try screen(p, SIMD3(0.5, 24, 0))
        #expect(at.x < centre.x && centre.x - at.x < 60, "drawn next to the target, not a world away: \(at) against \(centre)")
        #expect(pick(at, p, store) == .organism(uid: 1000, index: 0))
        #expect(pick(CGPoint(x: at.x + 30, y: at.y), p, store) == .none)
    }

    @Test func aReusedRowIsTheNewOrganismNotTheOldOne() throws {
        var store = try WorldFixtures.store(radius: 0.1, positions: [SIMD3(24, 24, 0)])
        let p = region()
        let at = try screen(p, SIMD3(24, 24, 0))
        #expect(pick(at, p, store) == .organism(uid: 1000, index: 0))
        // The organism dies and a newborn takes its row at the same place: the pick is the newborn's uid, so a selection
        // (which is a uid) of the old one does not follow the new.
        store.apply(.frma(WorldFixtures.frame([SIMD3(24, 24, 0)], step: 11, uids: [2000])))
        #expect(pick(at, p, store) == .organism(uid: 2000, index: 0))
        #expect(store.listedIndex(ofUID: 1000) == nil)
    }

    @Test func organismsAreNotPickedAtThePlanetLevel() throws {
        let store = try WorldFixtures.store(radius: 0.1, positions: [SIMD3(24, 24, 0)])
        let p = WorldFixtures.projector(target: SIMD3(24, 24, 0), distance: 70)
        #expect(p.level == .planet)
        let at = try screen(p, SIMD3(24, 24, 0))
        guard case .organism = pick(at, p, store) else { return }
        Issue.record("an organism was picked at the planet level")
    }

    @Test func hiddenAgentsAreNotPicked() throws {
        let store = try WorldFixtures.store(radius: 0.1, positions: [SIMD3(24, 24, 0)])
        let p = region()
        let at = try screen(p, SIMD3(24, 24, 0))
        var display = WorldDisplay()
        display.agents = false
        guard case .organism = pick(at, p, store, display: display) else { return }
        Issue.record("an organism that is not drawn was picked")
    }

    @Test func aRowBehindTheEyeIsNotPicked() throws {
        // A camera low over the floor looking along it: a row behind the eye has no projection and is never a candidate.
        var camera = Camera.planet(size: 48, height: 24, aspect: 1.3)
        camera.target = SIMD3(24, 24, 0)
        camera.distance = 5
        camera.pitch = 0
        camera.yaw = 0
        let p = WorldProjector(camera: camera, size: 48, view: CGSize(width: 900, height: 700))
        let store = try WorldFixtures.store(radius: 0.5, positions: [SIMD3(24, 24, 0)])
        _ = WorldPicker.pick(at: CGPoint(x: 450, y: 350), projector: p, store: store)  // must not crash on w <= 0 rows
    }

    // MARK: Storms, pools and territory

    @Test func aStormIsPickedInsideItsRing() throws {
        let store = try WorldFixtures.store()
        let p = WorldFixtures.projector(target: SIMD3(24, 24, 0), distance: 70)
        let storms = try #require(store.storms?.storms)
        let big = try #require(p.glyphs(of: storms).first { $0.id == 7 })
        #expect(pick(big.center, p, store) == .storm(id: 7))
        #expect(pick(CGPoint(x: big.center.x + big.radius * 0.8, y: big.center.y), p, store) == .storm(id: 7))
        #expect(pick(CGPoint(x: big.center.x + big.radius * 1.5, y: big.center.y), p, store) != .storm(id: 7))
    }

    @Test func aSmallStormCanStillBeHit() throws {
        // A storm of radius 0.1 is 2.5 points across at this zoom: its ring is drawn 5 points wide, and it is picked within 10.
        var store = try WorldFixtures.store(radius: 0.1, positions: [])
        let tiny = PlayStorms.Storm(id: 3, name: nil, named: false, hemi: 0, x: 24, y: 24, r: 0.1, ext: [0.1, 0.1], zeta: 0.3, zetaNow: 0.3,
                                    wind: 0, rain: 0, age: 10, parent: 0, state: "active", track: [], pred: [])
        store.apply(.strm(PlayStormRecord(head: RecordHead(step: 10, seq: 0, version: 1, flags: 0), tick: 25,
                                          storms: PlayStorms(detectStep: 10, storms: [tiny], ended: []))))
        let p = region()
        let record = try #require(store.storms?.storms)
        let glyph = try #require(p.glyphs(of: record).first)
        #expect(glyph.radius == 5, "the ring is never under 5 points")
        let centre = glyph.center
        #expect(pick(CGPoint(x: centre.x + 8, y: centre.y), p, store) == .storm(id: 3), "8 points away: inside the 10")
        #expect(pick(CGPoint(x: centre.x + 11, y: centre.y), p, store) == .none, "11 points away: outside")
    }

    @Test func aWarmPoolIsPickedInsideItsRing() throws {
        let store = try WorldFixtures.store()
        let p = WorldFixtures.projector(target: SIMD3(24, 24, 0), distance: 70)
        let pool = try screen(p, SIMD3(24, 12, 0))
        #expect(pick(pool, p, store) == .pool(index: 0))
    }

    @Test func aTerritoryCellWithAnOwnerIsPickedByItsKinGroup() throws {
        let store = try WorldFixtures.store()
        let p = WorldFixtures.projector(target: SIMD3(24, 24, 0), distance: 70)
        // Cells are 1.5 units: (3.1, 3.1) is cell (2, 2), owned by slot (0 + 0) % 3 = 0, kin group 17.
        let at = try screen(p, SIMD3(3.1, 3.1, 0))
        #expect(pick(at, p, store) == .territory(cell: 2 * 32 + 2, kin: 17))
        // (17.2, 0.8) is cell (11, 0): slot (1 + 0) % 3 = 1, kin group 21.
        #expect(pick(try screen(p, SIMD3(17.2, 0.8, 0)), p, store) == .territory(cell: 11, kin: 21))
        // A cell with no owner is nothing.
        #expect(pick(try screen(p, SIMD3(36, 36, 0)), p, store) == .none)
    }

    @Test func theTerritoryCellIsFoundAcrossTheSeam() throws {
        let store = try WorldFixtures.store()
        // The target is at the near edge, so cell (31, 31) at the far corner is drawn across the seam, owner none; cell
        // (0, 0) is drawn at the target. Picking at the screen position of world (0.5, 0.5) gives the cell (0, 0).
        let p = WorldFixtures.projector(target: SIMD3(0.5, 0.5, 0), distance: 30)
        let hit = pick(try screen(p, SIMD3(0.5, 0.5, 0)), p, store)
        #expect(hit == .territory(cell: 0, kin: 17))
        // Left of the target by one cell is world x = -1 = 47: cell 31, no owner.
        #expect(pick(try screen(p, SIMD3(-1.0, 0.5, 0)), p, store) == .none)
    }

    @Test func theTerritoryLayerCanBeSwitchedOff() throws {
        let store = try WorldFixtures.store()
        let p = WorldFixtures.projector(target: SIMD3(24, 24, 0), distance: 70)
        var display = WorldDisplay()
        display.territory = .off
        #expect(pick(try screen(p, SIMD3(3.1, 3.1, 0)), p, store, display: display) == .none)
    }

    @Test func theFloorPointIsTheInverseOfTheProjection() throws {
        let p = WorldFixtures.projector(target: SIMD3(20, 30, 0), distance: 40)
        for world in [SIMD3<Float>(20, 30, 0), SIMD3(14, 33, 0), SIMD3(27.5, 22, 0)] {
            let back = try #require(WorldPicker.floorPoint(at: try screen(p, world), projector: p))
            #expect(abs(back.x - world.x) < 1e-2 && abs(back.y - world.y) < 1e-2, "\(world) -> \(back)")
        }
        // Tilted, too: the ray still meets the floor where the projection puts the point.
        var tilted = p
        tilted.camera.pitch = -0.7
        tilted.camera.yaw = 1.0
        for world in [SIMD3<Float>(20, 30, 0), SIMD3(22, 28, 0)] {
            let back = try #require(WorldPicker.floorPoint(at: try screen(tilted, world), projector: tilted))
            #expect(abs(back.x - world.x) < 1e-2 && abs(back.y - world.y) < 1e-2, "\(world) -> \(back)")
        }
    }

    @Test func aRayThatMissesTheFloorPicksNothing() throws {
        // Looking up at the sky from near the floor.
        var camera = Camera.planet(size: 48, height: 24, aspect: 1.3)
        camera.target = SIMD3(24, 24, 10)
        camera.distance = 3
        camera.pitch = 0.4
        let p = WorldProjector(camera: camera, size: 48, view: CGSize(width: 900, height: 700))
        #expect(WorldPicker.floorPoint(at: CGPoint(x: 450, y: 350), projector: p) == nil)
        #expect(pick(CGPoint(x: 450, y: 350), p, try WorldFixtures.store()) == .none)
    }

    @Test func nothingIsUnderAnEmptyView() throws {
        let store = try WorldFixtures.store()
        let empty = WorldProjector(camera: Camera.planet(size: 48), size: 48, view: .zero)
        #expect(pick(.zero, empty, store) == .none)
        #expect(pick(CGPoint(x: 5, y: 5), region(), WorldStore()) == .none)
    }

    // MARK: Words

    @Test func theTooltipNamesWhatIsUnderThePointer() throws {
        let store = try WorldFixtures.store()
        #expect(WorldPicker.describe(.none, store: store) == nil)
        #expect(WorldPicker.describe(.organism(uid: uid0, index: 0), store: store) == "organism \(uid0) · Aldric kin · House Aldric")
        #expect(WorldPicker.describe(.organism(uid: uid0 + 100, index: 4), store: store)?.contains("born since the last snapshot") == true)
        #expect(WorldPicker.describe(.storm(id: 7), store: store) == "Merrow")
        #expect(WorldPicker.describe(.storm(id: 9), store: store) == "disturbance 9")
        #expect(WorldPicker.describe(.storm(id: 99), store: store) == "storm 99")
        #expect(WorldPicker.describe(.pool(index: 0), store: store) == "warm pool")
        #expect(WorldPicker.describe(.territory(cell: 0, kin: 21), store: store) == "Brennar kin territory (dominance)")
    }

    @Test func reservedSlotsHaveWords() throws {
        let store = try WorldFixtures.store()
        let tables = try #require(store.identities?.tables)
        #expect(Names.kin(slot: 254, tables: tables) == "other kin groups")
        #expect(Names.house(slot: 253, tables: tables) == "unknown lineage")
        #expect(Names.kin(slot: 255, tables: tables) == "no kin group")
        #expect(Names.kin(slot: 9, tables: tables) == "kin group 9")
        #expect(Names.label(0, in: ["", "Gatherer"]) == "none" && Names.label(1, in: ["", "Gatherer"]) == "Gatherer")
        #expect(Names.label(9, in: ["", "Gatherer"]) == "none")
    }
}
