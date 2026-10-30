import Foundation
import Testing
import simd

@testable import AlifeCore

/// What the planet's renderer is given: the cells' colours (`SphereColours`), the overlay lines (`SphereOverlay`), the follow
/// (`SphereFollow`) and the scene composed from a store (`SphereScene`).
@Suite struct SphereSceneTests {
    func store() throws -> WorldStore { try SphereFixture.store() }

    // MARK: Colours

    @Test func theFloorOfAWeatherWorldIsTheBoxMapsPaletteCell() throws {
        let store = try store()
        let sphere = try #require(store.header?.sphere)
        let map = try #require(store.sphereMap)
        let centres = SphereCentres.shared.centres(nside: sphere.nsideField)
        var display = WorldDisplay()
        display.structures = false
        display.soil = false
        let floor = SphereColours.floor(sphere: sphere, map: map, material: nil, soil: nil, display: display, layer: nil, centres: centres)
        #expect(floor.count == 192)
        for i in [0, 17, 100, 191] {
            let want = MapPalette.color(sea: Float(map.ts[i]) / 255, zeta: Float(map.zeta[i]) / 127, rain: Float(map.p[i]) / 255, cloud: 0,
                                        northness: SphereColours.northness(of: centres[i]), hemisphereWeight: 1, exposure: 1)
            #expect(simd_length(SIMD3(floor[i].x, floor[i].y, floor[i].z) - want) < 1e-6 && floor[i].w == 1, "cell \(i)")
        }
    }

    @Test func aDryPlanetHasItsOwnFloorAndTheMapSwitchHidesTheWeather() throws {
        let store = try store()
        let sphere = try #require(store.header?.sphere)
        let centres = SphereCentres.shared.centres(nside: sphere.nsideField)
        var display = WorldDisplay()
        display.map = false
        display.structures = false
        display.soil = false
        let dry = SphereColours.floor(sphere: sphere, map: store.sphereMap, material: nil, soil: nil, display: display, layer: nil, centres: centres)
        let none = SphereColours.floor(sphere: sphere, map: nil, material: nil, soil: nil, display: display, layer: nil, centres: centres)
        #expect(dry == none, "the map layer off is a dry planet")
        // the tint tells north from south: the two poles' cells differ
        let north = dry[0], south = dry[191]
        #expect(simd_length(SIMD3(north.x, north.y, north.z) - SIMD3(south.x, south.y, south.z)) > 1e-4)
        #expect(simd_length(SIMD3(north.x, north.y, north.z)) < 0.4, "a dark floor")
        let bright = { () -> [SIMD4<Float>] in
            var d = display
            d.exposure = 1.5
            return SphereColours.floor(sphere: sphere, map: nil, material: nil, soil: nil, display: d, layer: nil, centres: centres)
        }()
        #expect(bright[0].x > dry[0].x, "the brightness scales it")
    }

    @Test func theBuiltAndTheFarmedCellsAreBlendedOverTheFloor() throws {
        let store = try store()
        let sphere = try #require(store.header?.sphere)
        let centres = SphereCentres.shared.centres(nside: sphere.nsideField)
        var display = WorldDisplay()
        let without = { () -> [SIMD4<Float>] in
            var d = display
            d.structures = false
            d.soil = false
            return SphereColours.floor(sphere: sphere, map: store.sphereMap, material: nil, soil: nil, display: d, layer: nil, centres: centres)
        }()
        display.structures = true
        let with = SphereColours.floor(sphere: sphere, map: store.sphereMap, material: store.material, soil: store.soil, display: display,
                                       layer: nil, centres: centres)
        // the fixture's material: pixel 5 of layer 0 (full), pixel 100 of layer 1, pixel 191 of layer 2; the soil: pixel 40 of layer 0
        let full = CellLook.tileAlpha(level: 255)
        let c5 = SIMD3(without[5].x, without[5].y, without[5].z)
        let want5 = c5 + (CellLook.structures - c5) * full
        #expect(simd_length(SIMD3(with[5].x, with[5].y, with[5].z) - want5) < 1e-5)
        #expect(with[40] != without[40], "the soil is blended")
        #expect(with[6] == without[6], "an empty cell is the floor")
        // only layer 1: pixel 100 shows, pixel 5 (layer 0) does not
        let layer1 = SphereColours.floor(sphere: sphere, map: store.sphereMap, material: store.material, soil: nil, display: display, layer: 1, centres: centres)
        #expect(layer1[5] == without[5] && layer1[100] != without[100])
    }

    @Test func theCloudShellIsWhereTheCloudIsDeep() throws {
        let store = try store()
        let sphere = try #require(store.header?.sphere)
        let map = try #require(store.sphereMap)
        let display = WorldDisplay()
        let cloud = SphereColours.cloud(sphere: sphere, map: map, display: display)
        for i in [0, 3, 90, 150] {
            let want = MapPalette.cloudMix * MapPalette.smoothstep(MapPalette.cloudEdge.x, MapPalette.cloudEdge.y, Float(map.dc[i]) / 255)
            #expect(abs(cloud[i].w - want) < 1e-6)
        }
        var off = display
        off.map = false
        #expect(SphereColours.cloud(sphere: sphere, map: map, display: off).allSatisfy { $0 == .zero }, "no map, no cloud")
        #expect(SphereColours.cloud(sphere: sphere, map: nil, display: display).allSatisfy { $0 == .zero })
    }

    @Test func theTerritoryTakesTheOwnersColourAndTheHighlightFadesTheOthers() throws {
        let store = try store()
        let terr = SphereColours.territory(store.territory, nside: 3)
        #expect(terr.count == 108)
        // pixel 0 is slot 0 (kin 11), pixel 1 is slot 1 (kin 12), pixel 2 has no owner
        let (r0, g0, b0) = WorldPalette.components(WorldPalette.slots[0])
        #expect(abs(terr[0].x - Float(r0) / 255) < 1e-6 && abs(terr[0].y - Float(g0) / 255) < 1e-6 && abs(terr[0].z - Float(b0) / 255) < 1e-6)
        #expect(terr[2] == .zero && terr[0].w >= 0.5 && terr[0].w <= 1)
        let highlighted = SphereColours.territory(store.territory, nside: 3, highlight: 11)
        #expect(highlighted[0].w == 1 && highlighted[1].w < terr[1].w, "the selected group at full, the others fainter")
        #expect(SphereColours.territory(nil, nside: 3).allSatisfy { $0 == .zero })
    }

    @Test func aColourPacksAsARGB() {
        #expect(SphereColours.packed(SIMD4(1, 0, 0, 1)) == 0xFFFF0000)
        #expect(SphereColours.packed(SIMD4(0, 1, 0, 0.5)) == 0x8000FF00)
        #expect(SphereColours.packed(SIMD4(2, -1, 0.5, 1)) == 0xFFFF0080, "clamped")
    }

    // MARK: The overlay

    @Test func aCircleIsASmallCircleOfTheAngleAtTheRadius() {
        let centre = simd_normalize(SIMD3<Float>(1, 2, 3))
        let ring = SphereOverlay.circle(centre: centre, angle: 0.3, radius: 40)
        #expect(ring.count == 49 && ring.first == ring.last || simd_length(ring.first! - ring.last!) < 1e-4, "closed")
        for p in ring {
            #expect(abs(simd_length(p) - 40) < 1e-3)
            #expect(abs(SphereFrame.angle(simd_normalize(p), centre) - 0.3) < 1e-4)
        }
        let atPole = SphereOverlay.circle(centre: SIMD3(0, 0, 1), angle: 0.2, radius: 1)
        #expect(atPole.allSatisfy { abs(SphereFrame.angle($0, SIMD3(0, 0, 1)) - 0.2) < 1e-4 }, "a ring round the pole")
    }

    @Test func theStormsHaveARingATrackAndAPrediction() throws {
        let store = try store()
        let sphere = try #require(store.header?.sphere)
        let storms = try #require(store.storms?.storms.storms)
        let lines = SphereOverlay.storms(storms, sphere: sphere, selected: nil)
        // a ring of 48 segments, a track of 1 segment (two points), a prediction of 1 segment
        #expect(lines.count == 48 + 1 + 1)
        let ringLine = lines[0]
        let angle = SphereFrame.angle(simd_normalize(ringLine.a), SphereOverlay.direction(of: storms[0])!)
        #expect(abs(angle - Float(storms[0].r) / 20) < 1e-4, "the ring is the storm's arc radius on the floor")
        #expect(abs(simd_length(ringLine.a) - 20.35) < 1e-3, "lifted a little above the floor")
        let selected = SphereOverlay.storms(storms, sphere: sphere, selected: storms[0].id)
        #expect(selected[0].colorA == SphereOverlay.stormSelected && selected[0].widthA > lines[0].widthA)
        #expect(SphereOverlay.storms([], sphere: sphere, selected: nil).isEmpty)
        var noZ = storms[0]
        noZ.z = nil
        #expect(SphereOverlay.storms([noZ], sphere: sphere, selected: nil).isEmpty, "a box's storm has no place on a planet")
    }

    @Test func theWindStreamsAlongTheWind() throws {
        // a map with a uniform eastward wind of the reference speed: every streamline goes east, along a parallel
        let nside = 4
        let npix = Healpix.npix(nside)
        let winds = [Int8](repeating: 127, count: npix) + [Int8](repeating: 0, count: 3 * npix)
        let map = PlaySphereWeather(head: RecordHead(step: 0, seq: 0, version: 1, flags: 3), nside: nside, k: 1, f0: 0.025, pRef: 0.004,
                                    dcRef: 1, uRef: 0.1, zetaRef: 0.025, p: [UInt8](repeating: 0, count: npix), dc: [UInt8](repeating: 0, count: npix),
                                    zeta: [Int8](repeating: 0, count: npix), ts: [UInt8](repeating: 0, count: npix), winds: winds)
        let sphere = SphereHeader(planetRadius: 20, shellHeight: 8, shellEdges: [20, 24, 26, 28], nsideField: nside, nsideWeather: 8,
                                  nsideTerritory: 3, f0: 0.025, weatherDepth: 40)
        let lines = SphereOverlay.streamlines(map, sphere: sphere, layer: 0, seeds: 80, steps: 5, stepAngle: 0.03, lift: 3)
        #expect(!lines.isEmpty && lines.count % 5 == 0)
        for line in lines.prefix(60) {
            let a = line.a, b = line.b
            #expect(abs(simd_length(a) - 23) < 1e-3, "at the lift above the floor")
            let basis = SphereFrame.basis(at: a)
            let step = simd_normalize(b - a)
            #expect(simd_dot(step, basis.east) > 0.95, "the step is eastward")
            #expect(abs(SphereFrame.angle(simd_normalize(a), simd_normalize(b)) - 0.03) < 2e-3, "a step of the angle at full speed")
        }
        // a calm map draws nothing
        var calm = map
        calm.winds = [Int8](repeating: 0, count: 4 * npix)
        #expect(SphereOverlay.streamlines(calm, sphere: sphere, layer: 0, lift: 3).isEmpty)
    }

    @Test func theMarkerFacesTheEye() {
        let ring = SphereOverlay.marker(at: SIMD3(20, 0, 0), eye: SIMD3(60, 10, 5), radius: 2)
        #expect(ring.count == 28)
        let view = simd_normalize(SIMD3<Float>(60, 10, 5) - SIMD3(20, 0, 0))
        for line in ring {
            #expect(abs(simd_length(line.a - SIMD3(20, 0, 0)) - 2) < 1e-4)
            #expect(abs(simd_dot(line.a - SIMD3(20, 0, 0), view)) < 1e-4, "in the plane across the view")
        }
    }

    @Test func theBondsAreChordsBetweenTheirAgents() throws {
        let frame = try #require(try store().frame)
        let lines = SphereOverlay.bonds(frame, flow: false)
        #expect(lines.count == frame.nBonds && frame.nBonds == 2)
        for (k, line) in lines.enumerated() {
            let (i, j) = frame.bond(k)
            #expect(line.a == frame.position(i) && line.b == frame.position(j))
        }
    }

    // MARK: Following

    @Test func theFollowResolvesAnOrganismAStormAndALineage() throws {
        let store = try store()
        let frame = try #require(store.frame)
        var follow = SphereFollow(target: .organism(uid: frame.uid[2]))
        #expect(follow.goal(store: store).map { SphereFrame.angle($0, simd_normalize(frame.position(2))) < 1e-5 } == true)
        let storm = try #require(store.storms?.storms.storms.first)
        follow.set(.storm(id: storm.id))
        #expect(follow.goal(store: store).map { SphereFrame.angle($0, SphereOverlay.direction(of: storm)!) < 1e-5 } == true)
        follow.set(.organism(uid: 424242))
        #expect(follow.goal(store: store) == nil, "an organism that was never there has no place")
        follow.set(.none)
        #expect(follow.goal(store: store) == nil)
        // a dead organism leaves the camera where it died
        follow.set(.organism(uid: frame.uid[0]))
        let died = follow.goal(store: store)
        #expect(died != nil)
    }

    @Test func theApproachArrivesWithoutOvershootAndKeepsTheFrames() {
        var pose = SpherePose(lat: 0, lon: 0, tilt: 0.3, distance: 60)
        let goal = SphereFrame.direction(lat: 50, lon: 100)
        var previous = SphereFrame.angle(pose.up, goal)
        var moving = true
        var steps = 0
        while moving, steps < 600 {
            moving = SphereFollow.approach(&pose, toward: goal, distance: 15, dt: 1.0 / 60)
            let angle = SphereFrame.angle(pose.up, goal)
            #expect(angle <= previous + 1e-6, "no overshoot")
            #expect(abs(simd_length(pose.up) - 1) < 1e-4 && abs(simd_dot(pose.up, pose.heading)) < 1e-4)
            previous = angle
            steps += 1
        }
        #expect(!moving && steps < 600, "it arrives")
        #expect(SphereFrame.angle(pose.up, goal) < 3e-4 && abs(pose.distance - 15) < 0.1)
    }

    // MARK: The scene

    @Test func theSceneIsComposedFromTheStore() throws {
        let store = try store()
        let scene = try #require(SphereScene.compose(store: store, options: SphereScene.Options()))
        #expect(scene.floor.count == 192 && scene.cloud.count == 192 && scene.territory.count == 108)
        #expect(scene.overlay.count > 50, "the storm, the pools and the wind")
        var options = SphereScene.Options()
        options.selectedKin = 11
        let selected = try #require(SphereScene.compose(store: store, options: options))
        #expect(selected.territory[0].w == 1 && selected.territory != scene.territory)
        options.wind = false
        let calm = try #require(SphereScene.compose(store: store, options: options))
        #expect(calm.overlay.count < scene.overlay.count)
        #expect(SphereScene.compose(store: WorldStore(), options: SphereScene.Options()) == nil, "no planet, no scene")
    }

    @Test func theHomeViewFacesThePopulatedSide() throws {
        let store = try store()
        let frame = try #require(store.frame)
        let home = try #require(SphereScene.home(store: store, aspect: 1.5))
        var sum = SIMD3<Float>.zero
        for i in 0..<frame.n { sum += simd_normalize(frame.position(i)) }
        #expect(SphereFrame.angle(home.up, simd_normalize(sum)) < 1e-5 && home.isNorthUp && home.tilt == 0)
        #expect(SphereScene.home(store: WorldStore(), aspect: 1) == nil)
    }

    @Test func theFramesCentresAreCachedAndRightForEveryNside() {
        let a = SphereCentres.shared.centres(nside: 4), b = SphereCentres.shared.centres(nside: 4)
        #expect(a == b && a.count == 192)
        #expect(simd_length(a[0] - SIMD3(Float(Healpix.centre(of: 0, nside: 4).x), Float(Healpix.centre(of: 0, nside: 4).y), Float(Healpix.centre(of: 0, nside: 4).z))) < 1e-7)
    }
}
