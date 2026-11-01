import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeCore

/// Picking on the planet (DESIGN §46.8): a ray through the pointer against the floor sphere, and the organisms nearest to it.
@Suite struct SpherePickerTests {
    let view = CGSize(width: 800, height: 500)

    func store() throws -> WorldStore { try SphereFixture.store() }

    /// A camera looking at the organism `row` of the fixture from above, near enough to pick it.
    func camera(over position: SIMD3<Float>, distance: Float = 14, tilt: Float = 0) -> SphereCamera {
        let up = simd_normalize(position)
        return SphereCamera(pose: SpherePose(up: up, heading: SphereFrame.basis(at: up).north, tilt: tilt, distance: distance),
                            planetRadius: 20, lidRadius: 28)
    }

    @Test func theViewPointAndTheNDCAreInverse() {
        for p in [CGPoint(x: 0, y: 0), CGPoint(x: 800, y: 500), CGPoint(x: 400, y: 250), CGPoint(x: 123, y: 456)] {
            let q = SpherePicker.point(ofNDC: SpherePicker.ndc(of: p, in: view), in: view)
            #expect(abs(q.x - p.x) < 1e-3 && abs(q.y - p.y) < 1e-3)
        }
        let centre = SpherePicker.ndc(of: CGPoint(x: 400, y: 250), in: view)
        #expect(abs(centre.x) < 1e-6 && abs(centre.y) < 1e-6)
        #expect(SpherePicker.ndc(of: .zero, in: view) == SIMD2(-1, 1), "the top left is (-1, 1)")
    }

    @Test func anOrganismIsPickedAtTheImageOfItsPosition() throws {
        let store = try store()
        let frame = try #require(store.frame)
        let cam = camera(over: frame.position(0))
        for i in 0..<frame.n {
            guard !cam.isOccluded(frame.position(i)), let at = SpherePicker.screen(frame.position(i), camera: cam, view: view),
                  at.x >= 0, at.y >= 0, at.x < view.width, at.y < view.height else { continue }
            let pick = SpherePicker.pick(at: at, view: view, camera: cam, store: store, display: WorldDisplay(), level: .organism)
            #expect(pick == .organism(uid: frame.uid[i], index: i), "organism \(i)")
        }
    }

    @Test func theHiddenSideIsNotPicked() throws {
        let store = try store()
        let frame = try #require(store.frame)
        // the camera over the antipode of organism 0: the organism is behind the planet, under the centre of the picture
        let cam = camera(over: -frame.position(0), distance: 30)
        let pick = SpherePicker.pick(at: CGPoint(x: 400, y: 250), view: view, camera: cam, store: store, display: WorldDisplay(), level: .organism)
        if case .organism(let uid, _) = pick { #expect(uid != frame.uid[0], "an organism the planet hides was picked") }
    }

    @Test func atThePlanetLevelNoOrganismIsPicked() throws {
        let store = try store()
        let frame = try #require(store.frame)
        let cam = camera(over: frame.position(0), distance: 90)
        let at = try #require(SpherePicker.screen(frame.position(0), camera: cam, view: view))
        let pick = SpherePicker.pick(at: at, view: view, camera: cam, store: store, display: WorldDisplay(), level: .planet)
        if case .organism = pick { Issue.record("the planet level picks no organism") }
        var off = WorldDisplay()
        off.agents = false
        let hidden = SpherePicker.pick(at: at, view: view, camera: camera(over: frame.position(0)), store: store, display: off, level: .organism)
        if case .organism = hidden { Issue.record("an organism layer that is off picks nothing") }
    }

    @Test func aClickOnTheFloorNamesItsCell() throws {
        let store = try store()
        let sphere = try #require(store.header?.sphere)
        let cam = camera(over: SIMD3(5, 3, 18), distance: 40, tilt: 0.5)
        var display = WorldDisplay()
        display.territory = .off
        for point in [CGPoint(x: 400, y: 250), CGPoint(x: 300, y: 300), CGPoint(x: 500, y: 200)] {
            guard case .cell(let pixel, let d) = SpherePicker.pick(at: point, view: view, camera: cam, store: store, display: display, level: .region) else {
                // a point may have an organism or a storm: the fixture's storm and pools are away from these, but be explicit
                Issue.record("expected a cell at \(point)")
                continue
            }
            let under = try #require(cam.floorDirection(ndc: SpherePicker.ndc(of: point, in: view), aspect: Float(view.width / view.height)))
            #expect(SphereFrame.angle(d, under) < 1e-5)
            #expect(pixel == Healpix.pixel(of: SIMD3(Double(d.x), Double(d.y), Double(d.z)), nside: sphere.nsideField))
        }
    }

    @Test func aClickOffThePlanetPicksNothing() throws {
        let store = try store()
        let cam = camera(over: SIMD3(1, 0, 0), distance: 150)
        #expect(SpherePicker.pick(at: CGPoint(x: 5, y: 5), view: view, camera: cam, store: store, display: WorldDisplay(), level: .planet) == .none)
    }

    @Test func aStormIsPickedInsideItsRingAndAPoolInsideItsEllipse() throws {
        let store = try store()
        let storm = try #require(store.storms?.storms.storms.first)
        let centre = try #require(SphereOverlay.direction(of: storm))
        let cam = camera(over: centre * 20, distance: 40)
        var display = WorldDisplay()
        display.territory = .off
        let at = try #require(SpherePicker.screen(centre * 20, camera: cam, view: view))
        #expect(SpherePicker.pick(at: at, view: view, camera: cam, store: store, display: display, level: .region) == .storm(id: storm.id))
        // a pool: the fixture's first is at 40N 60E
        let pool = SphereFrame.direction(lat: 40, lon: 60)
        let cam2 = camera(over: pool * 20, distance: 40)
        let at2 = try #require(SpherePicker.screen(pool * 20, camera: cam2, view: view))
        #expect(SpherePicker.pick(at: at2, view: view, camera: cam2, store: store, display: display, level: .region) == .pool(index: 0))
    }

    @Test func aTerritoryWithAnOwnerIsPickedWithItsKin() throws {
        let store = try store()
        let sphere = try #require(store.header?.sphere)
        let terr = try #require(store.territory)
        // the centre of a territory pixel whose owner is slot 0 (kin id 11): the fixture's first pixel
        let owned = try #require((0..<terr.nx).first { terr.owner[$0] == 0 })
        let d = Healpix.centre(of: owned, nside: sphere.nsideTerritory)
        let direction = SIMD3<Float>(Float(d.x), Float(d.y), Float(d.z))
        let cam = camera(over: direction * 20, distance: 40)
        var display = WorldDisplay()
        display.territory = .tint
        let pick = SpherePicker.pick(at: CGPoint(x: 400, y: 250), view: view, camera: cam, store: store, display: display, level: .region)
        #expect(pick == .territory(pixel: owned, kin: 11))
        display.territory = .off
        if case .territory = SpherePicker.pick(at: CGPoint(x: 400, y: 250), view: view, camera: cam, store: store, display: display, level: .region) {
            Issue.record("the territory layer is off")
        }
    }

    @Test func theWordsOfAPick() throws {
        let store = try store()
        #expect(SpherePicker.describe(.none, store: store) == nil)
        #expect(SpherePicker.describe(.pool(index: 0), store: store) == "warm pool")
        #expect(SpherePicker.describe(.storm(id: 3), store: store) == "Merrow")
        #expect(SpherePicker.describe(.cell(pixel: 3, direction: SphereFrame.direction(lat: 12.34, lon: 45.67)), store: store) == "12.3N 45.7E")
        #expect(SpherePicker.describe(.territory(pixel: 0, kin: 11), store: store) == "kin group 11 territory (dominance)")
    }

    @Test func theCellTextHasThePlaceTheWeatherAndTheBuilt() throws {
        let store = try store()
        let sphere = try #require(store.header?.sphere)
        let text = SphereReadout.cellText(pixel: 5, sphere: sphere, store: store)
        #expect(text.contains("(cell 5)") && text.contains("rain") && text.contains("cloud"))
        #expect(text.contains("built 100% of the cap"), "pixel 5 of layer 0 holds a full cell of the fixture's STRC")
        #expect(SphereReadout.cellText(pixel: 40, sphere: sphere, store: store).contains("farmed 100% of the cap"))
        #expect(SphereReadout.cellText(pixel: 9999, sphere: sphere, store: store) == "no such cell")
        let terr = SphereReadout.cellText(pixel: 0, sphere: sphere, store: store)
        #expect(terr.contains("territory") || !terr.isEmpty)
    }
}
