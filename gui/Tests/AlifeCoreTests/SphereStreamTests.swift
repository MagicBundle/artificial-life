import Foundation
import Testing
import simd

@testable import AlifeCore

/// The protocol 2 stream (the sphere world, DESIGN §46.7): the header, `FRMA` orientation, `MAPH`, the sphere `TERR`, `STRC`, `FSOL`,
/// `ENVS`, `STRM` and `EVNT`, decoded from the stream the engine's encoders write (`tests/make_sphere_gui_fixture.py`).
@Suite struct SphereStreamTests {
    private func header() throws -> StreamHeader { try SphereFixture.decoded().header }

    @Test func theHeaderIsAProtocol2PlanetWithNoImageLength() throws {
        let h = try header()
        #expect(h.protocolVersion == 2 && h.isSphere && h.size == 0 && h.fieldGrid == 0 && h.fieldLayers == 3 && h.dims == 3)
        let s = try #require(h.sphere)
        #expect(s.planetRadius == 20 && s.shellHeight == 8 && s.lidRadius == 28 && s.layers == 3)
        #expect(s.nsideField == 4 && s.nsideWeather == 8 && s.nsideTerritory == 3 && s.hasWeather && s.weatherDepth == 40)
        #expect(s.fieldCells == 192 && s.territoryCells == 108)
        #expect(h.radius == 1, "radius stays the body radius of an agent")
        let play = try #require(h.play)
        #expect(play.weather.nside == 8 && play.weather.coarse == 4 && play.weather.enabled)
        #expect(play.records.contains("MAPH") && !play.records.contains("MAP1"))
        #expect(play.structures?.cells == [3, 1, 192] && play.niche?.cells == [3, 1, 192])
        #expect(play.labels.intentVersion == 4)
    }

    @Test func theEdgesAreTheEnginesEqualVolumeLayers() throws {
        let s = try #require(try header().sphere)
        let want = SphereFixture.numbers((SphereFixture.expected["planet"] as? [String: Any])?["edges"])
        #expect(s.shellEdges.count == want.count)
        for (a, b) in zip(s.shellEdges, want) { #expect(abs(a - b) < 1e-9) }
        let radii = SphereFixture.numbers(SphereFixture.expected["layer_radii"])
        let layers = (SphereFixture.expected["layer_of"] as? [Int]) ?? []
        for (r, k) in zip(radii, layers) { #expect(s.layer(ofRadius: r) == k, "layer of radius \(r)") }
    }

    @Test(arguments: [nil, 1, 3, 64, 1000]) func theStreamDecodesWholeAndInAnyChunking(chunk: Int?) throws {
        let result = decode(SphereFixture.stream, chunk: chunk)
        #expect(result.error == nil)
        let tags = result.events.compactMap { event -> String? in if case .record(let r) = event { r.tag } else { nil } }
        #expect(tags == ["FRMA", "MAPH", "TERR", "STRC", "FSOL", "ENVS", "STRM", "EVNT", "CLCK"])
        #expect(!result.events.contains { if case .unknownRecord = $0 { true } else { false } })
    }

    @Test func theAgentsAreWhereTheEngineSaidAndTheirHeadingIsRebuiltFromTheBearing() throws {
        let (_, records) = try SphereFixture.decoded()
        guard case .frma(let frame)? = StreamRecord.first("FRMA", in: records) else { Issue.record("no FRMA"); return }
        let want = try #require(SphereFixture.expected["agents"] as? [String: Any])
        #expect(frame.rows.map(Int.init) == (want["rows"] as? [Int]))
        #expect(frame.n == 5 && frame.nBonds == 2 && frame.bonds == [0, 1, 2, 3])
        let positions = (want["pos"] as? [Any])?.map { SphereFixture.vector($0) } ?? []
        let headings = (want["heading"] as? [Any])?.map { SphereFixture.vector($0) } ?? []
        let yaws = SphereFixture.numbers(want["yaw"]), pitches = SphereFixture.numbers(want["pitch"])
        let lats = SphereFixture.numbers(want["lat"]), lons = SphereFixture.numbers(want["lon"])
        let texts = (want["text"] as? [String]) ?? []
        for i in 0..<frame.n {
            let p = frame.position(i)
            #expect(simd_length(SIMD3<Double>(Double(p.x), Double(p.y), Double(p.z)) - positions[i]) < 1e-5)
            #expect(Double(frame.yaw(i)) == yaws[i] && Double(frame.pitch(i)) == pitches[i], "half-float bearing of row \(i)")
            let h = SphereFrame.heading(at: p, yaw: frame.yaw(i), pitch: frame.pitch(i))
            #expect(simd_length(SIMD3<Double>(Double(h.x), Double(h.y), Double(h.z)) - headings[i]) < 2e-6, "heading of agent \(i)")
            #expect(abs(simd_length(h) - 1) < 1e-6)
            let (lat, lon) = SphereFrame.latLon(of: p)
            // the longitude of a pole is whatever atan2 of rounding noise says; only the latitude is its place
            #expect(abs(lat - lats[i]) < 1e-4)
            if abs(lat) < 89.99 { #expect(abs(lon - lons[i]) < 1e-4) }
            #expect(SphereFrame.placeText(p, planetRadius: 20) == texts[i] || abs(abs(lat) - 90) < 1e-3, "place text of agent \(i)")
        }
    }

    @Test func theBearingIsTheInverseOfTheHeading() {
        for position in [SIMD3<Float>(15, 5, 12), SIMD3(0, 0, 21), SIMD3(0, 0, -21), SIMD3(-8, 1, -17), SIMD3(20, -20, 1)] {
            for yaw in stride(from: Float(-3), through: 3, by: 1.5) {
                for pitch: Float in [-1, -0.3, 0, 0.7, 1.3] {
                    let h = SphereFrame.heading(at: position, yaw: yaw, pitch: pitch)
                    let (y, p) = SphereFrame.bearing(at: position, heading: h)
                    #expect(abs(p - pitch) < 1e-4)
                    var d = y - yaw
                    d -= 2 * .pi * (d / (2 * .pi)).rounded()
                    #expect(abs(d) < 1e-3, "yaw \(yaw) pitch \(pitch) at \(position)")
                }
            }
        }
        let vertical = SphereFrame.bearing(at: SIMD3(10, 0, 5), heading: simd_normalize(SIMD3(10, 0, 5)))
        #expect(vertical.yaw == 0 && abs(vertical.pitch - .pi / 2) < 1e-5)
    }

    @Test func theWeatherMapIsDequantisedAsTheEngineSays() throws {
        let (_, records) = try SphereFixture.decoded()
        guard case .maph(let map)? = StreamRecord.first("MAPH", in: records) else { Issue.record("no MAPH"); return }
        #expect(map.nside == 4 && map.k == 1 && map.npix == 192 && map.head.flags == 3 && map.head.version == 1)
        let want = try #require(SphereFixture.expected["maph"] as? [String: Any])
        let p = SphereFixture.numbers(want["p"]), dc = SphereFixture.numbers(want["dc"])
        let zeta = SphereFixture.numbers(want["zeta"]), ts = SphereFixture.numbers(want["ts"])
        let winds = (want["winds"] as? [Any])?.map { SphereFixture.numbers($0) } ?? []
        #expect(p.count == 192 && winds.count == 4)
        for i in 0..<192 {
            #expect(abs(Double(map.rain(map.p[i])) - p[i]) < 1e-9)
            #expect(abs(Double(map.dc[i]) / 255 * Double(map.dcRef) - dc[i]) < 1e-4)
            #expect(abs(Double(map.zeta[i]) / 127 * Double(map.zetaRef) - zeta[i]) < 1e-9)
            #expect(abs(Double(map.ts[i]) / 255 - ts[i]) < 1e-12)
            for layer in 0..<2 {
                let w = map.wind(layer: layer, pixel: i)
                #expect(abs(Double(w.x) - winds[2 * layer][i]) < 1e-6 && abs(Double(w.y) - winds[2 * layer + 1][i]) < 1e-6)
            }
        }
        #expect(abs(map.pRef - 0.004) < 1e-9 && abs(map.dcRef - 20) < 1e-6 && abs(map.uRef - 0.1) < 1e-9)
    }

    @Test func theTerritoryMapIsOneRowOfRingPixels() throws {
        let (_, records) = try SphereFixture.decoded()
        guard case .terr(let t)? = StreamRecord.first("TERR", in: records) else { Issue.record("no TERR"); return }
        #expect(t.nx == 108 && t.ny == 1 && t.ids == [11, 12] && t.snapshotStep == 750)
        #expect(t.owner.prefix(6) == [0, 1, 255, 0, 1, 255] && t.purity[7] == 7)
    }

    @Test func theCellRecordsAreLayerTimesNpixPlusPixel() throws {
        let store = try SphereFixture.store()
        let material = try #require(store.material), soil = try #require(store.soil)
        #expect(material.nx == 192 && material.ny == 1 && material.nz == 3)
        #expect(material.indices == [5, 192 + 100, 2 * 192 + 191] && material.levels == [255, UInt8(1 + 254 * 1000 / 4096), UInt8(1 + 254 * 2048 / 4096)])
        #expect(soil.indices == [40] && soil.levels == [255])
        let c = material.cell(1)
        #expect(c.x == 100 && c.y == 0 && c.z == 1, "pixel 100 of layer 1")
        #expect(material.level(x: 5, y: 0, z: 0) == 255 && material.level(x: 6, y: 0, z: 0) == 0)
    }

    @Test func theEnvironmentStormAndEventCarryTheirPlanetFramePlaces() throws {
        let store = try SphereFixture.store()
        let env = try #require(store.environment).environment
        #expect(env.pools.count == 2 && env.pools[0].lon == 60 && env.pools[0].lat == 40 && env.pools[1].lat == -40)
        let pool = env.pools[0]
        let radius = (pool.x * pool.x + pool.y * pool.y + (pool.z ?? 0) * (pool.z ?? 0)).squareRoot()
        #expect(abs(radius - 20) < 0.02, "a pool is a planet-frame point at the floor radius")
        let patches = try #require(env.patches)
        #expect(patches.count == 2 && patches[0].lon == -30 && patches[1].lat == -60)
        let unit = (patches[1].x * patches[1].x + patches[1].y * patches[1].y + (patches[1].z ?? 9) * (patches[1].z ?? 9)).squareRoot()
        #expect(abs(unit - 1) < 1e-3, "a patch is a unit vector")
        let storm = try #require(store.storms?.storms.storms.first)
        #expect(storm.name == "Merrow" && storm.r == 6.5 && storm.z != nil)
        #expect(storm.track.map(\.count) == [4, 4] && storm.pred.map(\.count) == [4], "track [t, x, y, z] and pred [lead, x, y, z]")
        let event = try #require(store.chronicle.events.first)
        #expect(event.z != nil && event.kind == 7)
        #expect(store.sphereMap != nil && store.map == nil && store.mapSerial == 1)
    }

    // MARK: Refusals

    private func mutated(_ change: (inout [String: Any]) -> Void) throws -> [UInt8] {
        var bytes = SphereFixture.stream
        let jsonLength = Int(bytes[8]) | Int(bytes[9]) << 8 | Int(bytes[10]) << 16 | Int(bytes[11]) << 24
        var object = try #require(try JSONSerialization.jsonObject(with: Data(bytes[12..<(12 + jsonLength)])) as? [String: Any])
        change(&object)
        var json = [UInt8](try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]))
        json += [UInt8](repeating: 0x20, count: (4 - json.count % 4) % 4)
        var out = Array(bytes[0..<8])
        out += [UInt8(json.count & 255), UInt8((json.count >> 8) & 255), UInt8((json.count >> 16) & 255), UInt8(json.count >> 24)]
        bytes = out + json + Array(bytes[(12 + jsonLength)...])
        return bytes
    }

    @Test func aHeaderThatBreaksTheProtocol2RulesIsRefused() throws {
        let cases: [(String, (inout [String: Any]) -> Void)] = [
            ("FG not 0", { $0["FG"] = 12 }),
            ("protocol 1 with planet keys", { $0["protocol"] = 1 }),
            ("protocol 3", { $0["protocol"] = 3 }),
            ("edges not FZ + 1", { $0["shell_edges"] = [20.0, 28.0] }),
            ("edges not ascending", { $0["shell_edges"] = [20.0, 25.0, 24.0, 28.0] }),
            ("edges do not start at the planet", { $0["planet_radius"] = 21.0 }),
            ("ring order", { $0["ring_order"] = "nest" }),
            ("nside 1", { $0["nside_field"] = 1 }),
            ("dims 2", { $0["dims"] = 2 }),
            ("a flat world with geometry", { $0["geometry"] = "torus" }),
        ]
        for (name, change) in cases {
            let result = decode(try mutated(change))
            guard case .malformedHeader? = result.error else {
                Issue.record("\(name): expected a malformed header, got \(String(describing: result.error))")
                continue
            }
        }
    }

    @Test func aBoxStreamStaysAProtocol1StreamWithItsOwnChecks() throws {
        let box = try PlayFixture(kind: "3d")
        let events = try box.events()
        guard case .header(let h)? = events.first else { Issue.record("no header"); return }
        #expect(h.protocolVersion == 1 && h.sphere == nil && !h.isSphere && h.size > 0 && h.fieldGrid > 0)
        // a box stream refuses a sphere's record and a sphere refuses the box's
        var tagged = SphereFixture.stream
        // (the header's JSON names MAPH too: the record's tag is the last occurrence)
        let jsonLength = Int(tagged[8]) | Int(tagged[9]) << 8 | Int(tagged[10]) << 16
        let body = Array(tagged[(12 + jsonLength)...])
        let inBody = try #require(body.firstRange(of: Array("MAPH".utf8)))
        let at = (12 + jsonLength + inBody.lowerBound)..<(12 + jsonLength + inBody.upperBound)
        tagged.replaceSubrange(at, with: Array("MAP1".utf8))
        let result = decode(tagged)
        guard case .malformedRecord(let tag, _)? = result.error else { Issue.record("MAP1 in a v2 stream was accepted"); return }
        #expect(tag == "MAP1")
    }

    @Test func theTerritoryMapIsSizedByItsOwnRecordNotTheHeader() throws {
        // a header whose nside_territory differs: the record still decodes, and its Nside is 3 (nx = 108)
        let result = decode(try mutated { $0["nside_territory"] = 4 })
        #expect(result.error == nil)
        var store = WorldStore()
        for case .record(let r) in result.events { store.apply(r) }
        #expect(store.territory?.nside == 3 && store.territory?.nx == 108)
        #expect(try SphereFixture.store().territory?.nside == 3)
        // a record that is not one row of 12 nside^2 cells is refused
        var bytes = SphereFixture.stream
        let jsonLength = Int(bytes[8]) | Int(bytes[9]) << 8 | Int(bytes[10]) << 16
        let body = Array(bytes[(12 + jsonLength)...])
        let at = 12 + jsonLength + (try #require(body.firstRange(of: Array("TERR".utf8)))).lowerBound
        bytes[at + 8 + 16 + 2] = 2  // ny = 2
        guard case .malformedRecord(let tag, _)? = decode(bytes).error else { Issue.record("a two-row sphere map was accepted"); return }
        #expect(tag == "TERR")
    }
}
