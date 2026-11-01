import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeCore

/// What the independent review of S6 found and what its mutants showed the first tests did not pin: the signs and tolerances of the camera and the
/// picker, the pole, the validation of the stream, the north-up ease across 180 degrees, the drop lines, the real `tiny3s` stream.
@Suite struct SphereReviewTests {
    // MARK: Camera and picker

    @Test func aDragToTheRightTurnsTheViewClockwise() {
        var c = SphereCamera(pose: SpherePose(lat: 0, lon: 0, tilt: 0, distance: 50), planetRadius: 36, lidRadius: 52)
        c.turn(dx: 100, dy: 0)
        #expect(abs(c.pose.bearing - 0.6) < 1e-4, "dx > 0 turns the bearing of the screen's up direction towards east (the view turns to the right)")
        c.turn(dx: -200, dy: 0)
        #expect(abs(c.pose.bearing + 0.6) < 1e-4)
    }

    @Test func aSmallOrganismIsPickedWithinSixPointsAndNotBeyondNine() throws {
        let store = try SphereFixture.store()
        let frame = try #require(store.frame)
        let up = simd_normalize(frame.position(0))
        let cam = SphereCamera(pose: SpherePose(up: up, heading: SphereFrame.basis(at: up).north, tilt: 0, distance: 150), planetRadius: 20, lidRadius: 28)
        let view = CGSize(width: 800, height: 500)
        let at = try #require(SpherePicker.screen(frame.position(0), camera: cam, view: view))
        func pick(_ dx: CGFloat) -> SpherePick {
            SpherePicker.pick(at: CGPoint(x: at.x + dx, y: at.y), view: view, camera: cam, store: store, display: WorldDisplay(), level: .region)
        }
        #expect(pick(0) == .organism(uid: frame.uid[0], index: 0))
        #expect(pick(5) == .organism(uid: frame.uid[0], index: 0), "a small organism is forgiven 6 points")
        #expect(pick(9) != .organism(uid: frame.uid[0], index: 0), "and not 9")
    }

    @Test func aSmallStormIsPickedWithinTenPointsAndNotBeyondFourteen() throws {
        var store = try SphereFixture.store()
        var record = try #require(store.storms)
        record.storms.storms[0].r = 0.3
        store.apply(.strm(record))
        let centre = try #require(SphereOverlay.direction(of: record.storms.storms[0]))
        let cam = SphereCamera(pose: SpherePose(up: centre, heading: SphereFrame.basis(at: centre).north, tilt: 0, distance: 100), planetRadius: 20, lidRadius: 28)
        let view = CGSize(width: 800, height: 500)
        var display = WorldDisplay()
        display.territory = .off
        display.agents = false
        let at = try #require(SpherePicker.screen(centre * 20, camera: cam, view: view))
        func pick(_ dx: CGFloat) -> SpherePick {
            SpherePicker.pick(at: CGPoint(x: at.x + dx, y: at.y), view: view, camera: cam, store: store, display: display, level: .region)
        }
        #expect(pick(0) == .storm(id: 3) && pick(8) == .storm(id: 3), "a storm smaller than 10 points is picked within 10")
        if case .storm = pick(15) { Issue.record("a storm 15 points away was picked") }
    }

    // MARK: The pole and the bearing

    @Test func justOffAPoleTheEastIsStillTheEngines() {
        // |z x up| = 5e-4: above the engine's 1e-6 epsilon, so the east is the normalised z x up and not the pole's (1, 0, 0)
        let b = SphereFrame.basis(at: SIMD3(0.0005, 0, 1) * 21)
        #expect(simd_length(b.east - SIMD3(0, 1, 0)) < 1e-4, "east at \(b.east)")
        let exact = SphereFrame.basis(at: SIMD3(0, 0, 21))
        #expect(exact.east == SIMD3(1, 0, 0) && exact.up == SIMD3(0, 0, 1))
    }

    @Test func aBearingOfExactlyPiIsMinusPi() {
        let (yaw, pitch) = SphereFrame.bearing(at: SIMD3(21, 0, 0), heading: SIMD3(0, 0, -1))
        #expect(abs(yaw + .pi) < 1e-6 && pitch == 0, "the yaw is in [-pi, pi): south is -pi, not +pi (got \(yaw))")
    }

    // MARK: The stream's validation

    private func boxBytes() throws -> [UInt8] { try PlayFixture(kind: "3d").bytes }

    @Test func aProtocol3HeaderIsRefusedWhetherOrNotItHasPlanetKeys() throws {
        var bytes = try boxBytes()
        let jsonLength = Int(bytes[8]) | Int(bytes[9]) << 8 | Int(bytes[10]) << 16
        var object = try #require(try JSONSerialization.jsonObject(with: Data(bytes[12..<(12 + jsonLength)])) as? [String: Any])
        object["protocol"] = 3
        var json = [UInt8](try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]))
        json += [UInt8](repeating: 0x20, count: (4 - json.count % 4) % 4)
        let header = Array(bytes[0..<8]) + [UInt8(json.count & 255), UInt8((json.count >> 8) & 255), UInt8((json.count >> 16) & 255), 0] + json
        bytes = header + Array(bytes[(12 + jsonLength)...])
        guard case .malformedHeader? = decode(bytes).error else { Issue.record("a protocol 3 box header was accepted"); return }
    }

    /// The fixture's bytes with `change` applied to the payload of its (last) record of `tag`; `at` is the offset of the payload.
    private func record(_ tag: String, _ change: (inout [UInt8], Int) -> Void) throws -> [UInt8] {
        var bytes = SphereFixture.stream
        let jsonLength = Int(bytes[8]) | Int(bytes[9]) << 8 | Int(bytes[10]) << 16
        let body = Array(bytes[(12 + jsonLength)...])
        let found = try #require(body.lastRange(of: Array(tag.utf8)))
        change(&bytes, 12 + jsonLength + found.lowerBound + 8)
        return bytes
    }

    @Test func aWeatherMapWithAnyOddByteIsRefused() throws {
        for (name, change) in [
            ("reserved byte 44", { (b: inout [UInt8], p: Int) in b[p + 44] = 1 }),
            ("reserved byte 47", { b, p in b[p + 47] = 9 }),
            ("flags 1", { b, p in b[p + 14] = 1 }),
            ("k that does not make nside_weather", { b, p in b[p + 18] = 2 }),
            ("nside", { b, p in b[p + 16] = 5 }),
        ] as [(String, (inout [UInt8], Int) -> Void)] {
            guard case .malformedRecord(let tag, _)? = decode(try record("MAPH", change)).error else {
                Issue.record("MAPH with \(name) was accepted")
                continue
            }
            #expect(tag == "MAPH")
        }
    }

    @Test(arguments: [2, 4])
    func aCellRecordOfAnotherDepthIsSkippedAndCounted(nz: UInt8) throws {
        // nz 4 keeps every index inside the grid: only the depth against FZ = 3 can refuse it
        var decoder = FrameDecoder()
        let bytes = try record("STRC") { b, p in b[p + 20] = nz }
        let events = try decoder.feed(Data(bytes))
        #expect(decoder.skippedBodies == 1 && decoder.lastSkippedBody?.hasPrefix("STRC") == true)
        #expect(events.contains { if case .unknownRecord(let tag, _) = $0 { tag == "STRC" } else { false } })
    }

    @Test func theWeatherBlockOfAPlanetObeysItsOwnRules() throws {
        let header = try SphereFixture.decoded().header
        var play = try #require(header.play)
        try play.validate()
        for (name, nside, enabled) in [("not a multiple", 10, true), ("not a power of two", 12, true), ("enabled without weather", 0, true),
                                       ("weather without enabled", 8, false)] {
            var bad = play
            bad.weather.nside = nside
            bad.weather.enabled = enabled
            #expect(throws: StreamError.self, "\(name)") { try bad.validate() }
        }
        play.weather.nside = 16  // 4 x 4: a power of two
        try play.validate()
    }

    @Test func aHeaderThatAsksForTooManyCellsIsRefused() throws {
        // FZ * 12 * Nf^2 <= 2^20 (the engine's exact-sum limit): Nside 2048 with one layer is 50 M cells
        let bytes = SphereFixture.stream
        let jsonLength = Int(bytes[8]) | Int(bytes[9]) << 8 | Int(bytes[10]) << 16
        var object = try #require(try JSONSerialization.jsonObject(with: Data(bytes[12..<(12 + jsonLength)])) as? [String: Any])
        object["nside_field"] = 400
        var json = [UInt8](try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]))
        json += [UInt8](repeating: 0x20, count: (4 - json.count % 4) % 4)
        let out = Array(bytes[0..<8]) + [UInt8(json.count & 255), UInt8((json.count >> 8) & 255), UInt8((json.count >> 16) & 255), 0] + json + Array(bytes[(12 + jsonLength)...])
        guard case .malformedHeader(let why)? = decode(out).error else { Issue.record("the header was accepted"); return }
        #expect(why.contains("2^20"))
    }

    @Test func aPositionThatIsNotFiniteIsRefused() throws {
        let bytes = try record("FRMA") { b, p in
            // the first position's x: after the head (16), the fixed part (20), the bitmap
            let at = p + 36 + 4 * ((16 + 31) / 32)
            b[at] = 0; b[at + 1] = 0; b[at + 2] = 0xC0; b[at + 3] = 0x7F  // a quiet NaN
        }
        guard case .malformedRecord(let tag, let why)? = decode(bytes).error else { Issue.record("a NaN position was accepted"); return }
        #expect(tag == "FRMA" && why.contains("finite"))
    }

    // MARK: A real tiny3s stream

    @Test func aRealTiny3SStreamDecodesAndHasItsPlanet() throws {
        let bytes = [UInt8](try Data(contentsOf: Repo.streamData.appendingPathComponent("play_tiny3s.bin")))
        #expect(bytes.count < 1_000_000)
        let result = decode(bytes)
        #expect(result.error == nil)
        guard case .header(let header)? = result.events.first, let sphere = header.sphere else { Issue.record("no planet"); return }
        #expect(sphere.planetRadius == 36 && sphere.shellHeight == 16 && sphere.layers == 8 && sphere.nsideField == 8)
        #expect(sphere.nsideWeather == 16 && sphere.nsideTerritory == 9 && header.capacity == 1024)
        var store = WorldStore()
        store.reset(header: header)
        for case .record(let r) in result.events { store.apply(r) }
        #expect(store.frame?.n == 130 && store.sphereMap?.npix == 768 && store.sphereMap?.k == 1)
        #expect(store.territory?.nside == 9 && store.territory?.nx == 972)
        #expect(store.material?.nx == 768 && store.material?.nz == 8)
        let frame = try #require(store.frame)
        for i in 0..<frame.n {
            let h = simd_length(frame.position(i)) - 36
            #expect(h >= -1e-3 && h <= 16 + 1e-3, "agent \(i) at height \(h)")
        }
        let scene = try #require(SphereScene.compose(store: store, options: SphereScene.Options()))
        #expect(scene.floor.count == 768 && scene.territory.count == 972)
    }

    // MARK: The north-up ease

    @Test(arguments: [180.0, 162.0, 90.0, -170.0, 30.0])
    func theNorthUpEaseTurnsTheShorterWayWithoutStalling(degrees: Double) {
        let up = SphereFrame.direction(lat: 25, lon: 70)
        let b = SphereFrame.basis(at: up)
        let a = Float(degrees * .pi / 180)
        let from = simd_normalize(cos(a) * b.north + sin(a) * b.east)
        func angleToNorth(_ h: SIMD3<Float>) -> Float { abs(atan2(simd_dot(simd_cross(h, b.north), up), simd_dot(h, b.north))) }
        var previous = angleToNorth(from)
        let start = previous
        for k in 0...20 {
            let e = Float(k) / 20
            let h = SpherePose.easedHeading(from: from, up: up, fraction: e)
            let angle = angleToNorth(h)
            #expect(abs(simd_length(h) - 1) < 1e-5 && abs(simd_dot(h, up)) < 1e-5)
            #expect(angle <= previous + 1e-4, "the angle to north only shrinks (at \(e): \(angle) after \(previous))")
            #expect(abs(angle - start * (1 - e)) < 0.02, "and at the rate of a turn by the signed angle (\(angle) against \(start * (1 - e)))")
            previous = angle
        }
        #expect(previous < 1e-4)
        #expect(SpherePose.easedHeading(from: from, up: up, fraction: 0) == simd_normalize(from - simd_dot(from, up) * up))
        #expect(SpherePose.easedHeading(from: up, up: up, fraction: 0.3) == b.north, "a heading with no tangent part goes straight to north")
    }

    // MARK: The depth cue

    @Test func anOrganismAloftHasADropLineToTheFloorAndOneOnTheFloorHasNone() throws {
        let frame = TestFramesLike.frame([SIMD3(0, 0, 20.5), SIMD3(0, 0, 24), SIMD3(26, 0, 0)])  // heights 0.5, 4, 6 over Rp 20
        let lines = SphereOverlay.drops(frame, planetRadius: 20)
        #expect(lines.count == 2)
        for line in lines {
            #expect(abs(simd_length(line.b) - 20) < 1e-4 && SphereFrame.angle(line.a, line.b) < 1e-6, "straight down to the floor")
            #expect(line.colorA.w > line.colorB.w, "fainter at the floor")
        }
        #expect(SphereOverlay.drops(frame, planetRadius: 20, minHeight: 10).isEmpty)
        let capped = SphereOverlay.drops(frame, planetRadius: 20, minHeight: 2, cap: 1)
        #expect(capped.count == 1 && abs(simd_length(capped[0].a) - 26) < 1e-4, "the highest first")
    }

    // MARK: The launcher

    @Test func aPlanetTakesTheSpinupOfItsShapeWhateverItsWorldSize() {
        // a shape without a row (it has no overlay either) takes the long value, whatever the size
        #expect(PlanetSpec.defaultSpinup(forGeometry: "sphere", worldSize: 160) == PlanetSpec.sphereSpinup)
        #expect(PlanetSpec.defaultSpinup(forGeometry: "sphere", worldSize: 1024) == PlanetSpec.sphereSpinup)
        for row in PlanetSpec.sphereSpinups {
            #expect(PlanetSpec.defaultSpinup(forGeometry: "sphere", shape: row.shape, worldSize: 160) == row.steps)
            #expect(PlanetSpec.defaultSpinup(forGeometry: "sphere", shape: row.shape, worldSize: 1024) == row.steps)
            #expect(PlanetSpec.defaultSpinup(forGeometry: "box", shape: row.shape, worldSize: 320) == PlanetSpec.defaultSpinup, "a box ignores the shape")
            #expect(PlanetSpec.overlay(forShape: row.shape) != nil, "every shape with a spin-up has an overlay")
        }
        #expect(PlanetSpec.sphereSpinups.map(\.shape) == PlanetSpec.weatherOverlays.map(\.shape), "one row per overlay, in order")
        #expect(PlanetSpec.defaultSpinup(forGeometry: "box", worldSize: 160) == PlanetSpec.miniSpinup)
        #expect(PlanetSpec.defaultSpinup(forGeometry: nil, worldSize: 320) == PlanetSpec.defaultSpinup)
    }
}

/// A frame of the listed rows at `positions` (as the app tests' `TestFrames`).
fileprivate enum TestFramesLike {
    static func frame(_ positions: [SIMD3<Float>]) -> PlayFrame {
        let n = positions.count
        var bitmap = [UInt32](repeating: 0, count: max(1, (n + 31) / 32))
        for r in 0..<n { bitmap[r >> 5] |= 1 << UInt32(r & 31) }
        return PlayFrame(head: RecordHead(step: 1, seq: 0, version: 1, flags: 0), n: n, capacity: max(n, 32), colorIndex: 0, span: 0, bitmap: bitmap,
                         rows: (0..<n).map { UInt32($0) }, pos: positions.flatMap { [$0.x, $0.y, $0.z] }, orient: [UInt16](repeating: 0, count: 2 * n),
                         rgb: [UInt32](repeating: 0xFFFFFF, count: n), uid: (0..<n).map { Int64($0) }, bonds: [])
    }
}
