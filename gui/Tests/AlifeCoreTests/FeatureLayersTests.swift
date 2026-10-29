import Foundation
import Testing

@testable import AlifeCore

/// What the World window makes of the feature layers (DESIGN §45): which layers a header carries, the cell and bond-flow look, the
/// inspector's rows and voice, the legend, the metrics readout and the display toggles.
@Suite struct FeatureLayersTests {
    private let head = RecordHead(step: 104, seq: 0, version: 1, flags: 0)

    /// The 3D fixture's world with all the feature layers on in its header, a role-coloured frame with a flow, and `cells` as the material.
    private func featureStore(material: [(x: Int, y: Int, z: Int, level: UInt8)] = [], soil: [(x: Int, y: Int, z: Int, level: UInt8)] = [],
                              voice: PlayVoice? = nil, flow: [Int8]? = [0, 0, 64]) throws -> WorldStore {
        let source = try WorldFixtures.store("3d")
        var header = try #require(source.header)
        header.play?.structures = .init(enabled: true, cap: 4096, cells: [header.fieldLayers, header.fieldGrid, header.fieldGrid],
                                        cell: header.fieldCell)
        header.play?.niche = .init(enabled: true, farm: true, cap: 4095, cells: [header.fieldLayers, header.fieldGrid, header.fieldGrid],
                                   cell: header.fieldCell)
        header.play?.comm = .init(enabled: true, w: 3, range: 8, control: "none")
        header.play?.bonds = .init(flow: true, scale: 1000)
        header.play?.records += ["STRC", "FSIG", "FSOL"]
        header.colorModes.append("role")
        var out = WorldStore()
        out.reset(header: header)
        var frame = try #require(source.frame)
        frame.colorIndex = header.colorModes.count - 1
        frame.flow = flow
        for i in 0..<frame.n { frame.rgb[i] = 0xBEBEBE }
        let sourceFocus = try #require(source.focus)
        let focusIndex = try #require(source.listedIndex(ofUID: sourceFocus.uid))
        frame.rgb[focusIndex] = 0xF0E442
        for record in [source.identities.map(StreamRecord.idnt), source.focus.map(StreamRecord.focs),
                       source.labels.map(StreamRecord.intn), StreamRecord.frma(frame)].compactMap({ $0 }) {
            out.apply(record)
        }
        func cells(_ list: [(x: Int, y: Int, z: Int, level: UInt8)]) -> PlayCells {
            let n = header.fieldGrid
            let sorted = list.sorted { ($0.z * n + $0.y) * n + $0.x < ($1.z * n + $1.y) * n + $1.x }
            return PlayCells(head: head, nx: n, ny: n, nz: header.fieldLayers,
                             indices: sorted.map { UInt32(($0.z * n + $0.y) * n + $0.x) }, levels: sorted.map(\.level))
        }
        if !material.isEmpty { out.apply(.strc(cells(material))) }
        if !soil.isEmpty { out.apply(.fsol(cells(soil))) }
        if let voice { out.apply(.fsig(voice)) }
        return out
    }

    private func row(_ card: InspectorCard, _ label: String) -> String? { card.rows.first { $0.label == label }?.value }

    // MARK: Which layers

    @Test func aHeaderNamesTheLayersItCarries() throws {
        let store = try featureStore()
        let layers = store.layers
        #expect(layers == FeatureLayers(header: store.header) && layers.any)
        #expect(layers.structures && layers.soil && layers.voice && layers.bondFlow && layers.roles)
        let plain = try WorldFixtures.store("3d").layers
        #expect(plain == FeatureLayers() && !plain.any)
        #expect(FeatureLayers(header: nil) == FeatureLayers())
        let plainStore = try WorldFixtures.store("3d")
        #expect(store.carries("STRC") && !plainStore.carries("STRC"))
    }

    // MARK: Looks

    @Test func theCellOpacityIsAffineInTheLevelAndZeroForNothing() {
        #expect(CellLook.tileAlpha(level: 0) == 0 && CellLook.cubeAlpha(level: 0) == 0)
        #expect(CellLook.tileAlpha(level: 255) == CellLook.tile.base + CellLook.tile.slope)
        #expect(CellLook.cubeAlpha(level: 255) == CellLook.cube.base + CellLook.cube.slope)
        var last: Float = 0
        for level in 1...255 {
            let a = CellLook.tileAlpha(level: UInt8(level))
            #expect(a > last && a <= 1, "a fuller cell is more opaque")
            #expect(CellLook.cubeAlpha(level: UInt8(level)) < a, "a cube is fainter than a tile")
            last = a
        }
        #expect(CellLook.maxInstances == 65536)
    }

    @Test func theBondFlowIsScaledAndSigned() {
        #expect(BondFlowLook.strength(0) == 0 && BondFlowLook.strength(127) == 1 && BondFlowLook.strength(-127) == 1)
        #expect(BondFlowLook.strength(-64) == 64.0 / 127)
        #expect(BondFlowLook.quanta(127, scale: 1000) == 1000 && BondFlowLook.quanta(-127, scale: 1000) == -1000)
        #expect(BondFlowLook.quanta(0, scale: 1000) == 0)
    }

    // MARK: Looking up a cell

    @Test func aLevelIsFoundByItsCellAndByAPointOfTheWorld() throws {
        let store = try featureStore(material: [(2, 3, 1, 90), (5, 5, 0, 255), (11, 11, 5, 7)])
        let material = try #require(store.material)
        #expect(material.level(x: 2, y: 3, z: 1) == 90 && material.level(x: 5, y: 5, z: 0) == 255 && material.level(x: 11, y: 11, z: 5) == 7)
        #expect(material.level(x: 2, y: 3, z: 0) == 0 && material.level(x: 12, y: 0, z: 0) == 0 && material.level(x: -1, y: 0, z: 0) == 0)
        // the cell is 4 long: x 8 to 12, y 12 to 16, z 4 to 8; x and y wrap with the world (48), z does not
        #expect(material.level(atX: 9.9, y: 12.0, z: 5.0, cell: 4) == 90)
        #expect(material.level(atX: 9.9 + 48, y: 12.0 - 48, z: 5.0, cell: 4) == 90, "the world is periodic in x and y")
        #expect(material.level(atX: 9.9, y: 12.0, z: 25, cell: 4) == 0, "z is clamped to the top layer, which is empty here")
        #expect(material.level(atX: .nan, y: 0, z: 0, cell: 4) == 0 && material.level(atX: 1, y: 1, z: 1, cell: 0) == 0)
    }

    @Test func theColumnMaximumIsWhatTheFlatMapDraws() throws {
        let store = try featureStore(material: [(2, 3, 1, 90), (2, 3, 4, 200), (7, 0, 0, 5)])
        let image = try #require(store.material).columnMaximum()
        #expect(image[3 * 12 + 2] == 200 && image[0 * 12 + 7] == 5 && image.filter { $0 > 0 }.count == 2)
    }

    // MARK: The inspector

    @Test func theFocusedOrganismShowsItsRoleFlowMaterialAndSoil() throws {
        let probe = try featureStore()
        let focus = try #require(probe.focus)
        let index = try #require(probe.listedIndex(ofUID: focus.uid))
        let p = try #require(probe.frame).position(index)
        let cell = (x: Int(p.x / 4) % 12, y: Int(p.y / 4) % 12, z: min(Int(p.z / 4), 5))
        let store = try featureStore(material: [(cell.x, cell.y, cell.z, 129)], soil: [(cell.x, cell.y, cell.z, 255)])
        let card = Inspector.organism(uid: focus.uid, store: store)
        #expect(row(card, "Role") == "Forager")
        #expect(row(card, "Bond flow") == "-504 q/step net over 1 bond", "the organism is the lower end of its bond and gives 64/127 of 1000")
        #expect(row(card, "Built here") == "51% of the cap (about 2064 units)")
        #expect(row(card, "Soil here") == "100% of the cap (about 4095 units)")
        #expect(card.voice == nil, "no voice record yet")
    }

    @Test func aRoleCoveredByTheHurtRedSaysSo() throws {
        let probe = try featureStore()
        let focus = try #require(probe.focus)
        var store = WorldStore()
        store.reset(header: probe.header)
        var frame = try #require(probe.frame)
        frame.rgb[try #require(probe.listedIndex(ofUID: focus.uid))] = 0xFF0000
        for record in [probe.identities.map(StreamRecord.idnt), probe.focus.map(StreamRecord.focs), StreamRecord.frma(frame)].compactMap({ $0 }) {
            store.apply(record)
        }
        #expect(row(Inspector.organism(uid: focus.uid, store: store), "Role") == "hurt in the last step (red)")
    }

    @Test func aNoteSaysHowToSeeTheRoleInAnotherColourMode() throws {
        let probe = try featureStore()
        let focus = try #require(probe.focus)
        var store = WorldStore()
        store.reset(header: probe.header)
        var frame = try #require(probe.frame)
        frame.colorIndex = 0
        for record in [probe.identities.map(StreamRecord.idnt), probe.focus.map(StreamRecord.focs), StreamRecord.frma(frame)].compactMap({ $0 }) {
            store.apply(record)
        }
        let card = Inspector.organism(uid: focus.uid, store: store)
        #expect(row(card, "Role") == nil && card.notes.contains("colour by role (C) to see its role"))
    }

    @Test func aWorldWithoutTheLayersHasNoneOfTheRows() throws {
        let store = try WorldFixtures.store("3d")
        let card = Inspector.organism(uid: try #require(store.focus).uid, store: store)
        for label in ["Role", "Bond flow", "Built here", "Soil here"] { #expect(row(card, label) == nil, Comment(rawValue: label)) }
        #expect(card.voice == nil)
    }

    @Test func theVoiceIsShownOnlyForTheOrganismTheFocusFollows() throws {
        let probe = try featureStore()
        let focus = try #require(probe.focus)
        let voice = PlayVoice(head: head, uid: focus.uid, w: 3, heardValid: true, audible: 2, says: [0, 51, 255], hears: [255, 0, 102])
        let store = try featureStore(voice: voice)
        let panel = try #require(Inspector.organism(uid: focus.uid, store: store).voice)
        #expect(panel.channels == 3 && panel.says == [0, 0.2, 1] && panel.hears == [1, 0, 0.4] && panel.audible == 2 && !panel.isSilent)
        // the card of another organism has none
        #expect(Inspector.organism(uid: focus.uid + 1, store: store).voice == nil)
        // without the observer channel the heard half is absent
        let quiet = PlayVoice(head: head, uid: focus.uid, w: 2, heardValid: false, audible: 0, says: [0, 0], hears: [0, 0])
        let spoken = try #require(Inspector.organism(uid: focus.uid, store: try featureStore(voice: quiet)).voice)
        #expect(spoken.hears == nil && spoken.audible == nil && spoken.isSilent && spoken.channels == 2)
    }

    @Test func theCardsEqualityCoversTheVoice() throws {
        let probe = try featureStore()
        let focus = try #require(probe.focus)
        let a = Inspector.organism(uid: focus.uid, store: try featureStore(voice: PlayVoice(
            head: head, uid: focus.uid, w: 1, heardValid: false, audible: 0, says: [5], hears: [0])))
        let b = Inspector.organism(uid: focus.uid, store: try featureStore(voice: PlayVoice(
            head: head, uid: focus.uid, w: 1, heardValid: false, audible: 0, says: [6], hears: [0])))
        #expect(a != b && a == a)
    }

    @Test func theWordsOfACell() {
        #expect(FeatureWords.cellText(level: 255, cap: 4096) == "100% of the cap (about 4096 units)")
        #expect(FeatureWords.cellText(level: 1, cap: 4096) == "0% of the cap (about 0 units)")
        #expect(FeatureWords.role(forColour: 0x0072B2) == "Guard" && FeatureWords.role(forColour: 0xCC79A7) == "Germ")
        #expect(FeatureWords.role(forColour: 0xFF0000) == nil && FeatureWords.role(forColour: 0xFF0072B2) == "Guard")
    }

    // MARK: The legend and the toggles

    @Test func theLegendListsWhatIsDrawn() throws {
        let layers = try featureStore().layers
        var display = WorldDisplay()
        let all = FeatureLegend.entries(layers: layers, display: display, colourMode: "role").map(\.title)
        #expect(all == ["Generalist", "Forager", "Guard", "Germ", "Built material", "Farmed soil", "Bond: gives", "Bond: receives"])
        #expect(FeatureLegend.entries(layers: layers, display: display, colourMode: "tag").map(\.title)
            == ["Built material", "Farmed soil", "Bond: gives", "Bond: receives"], "the roles only in the role mode")
        display.structures = false
        display.soil = false
        display.bonds = false
        #expect(FeatureLegend.entries(layers: layers, display: display, colourMode: "tag").isEmpty)
        #expect(FeatureLegend.entries(layers: FeatureLayers(), display: WorldDisplay(), colourMode: "role").isEmpty)
        #expect(FeatureLegend.entries(layers: layers, display: WorldDisplay(), colourMode: "role")[1].colour
            == SIMD3(Float(0xF0) / 255, Float(0xE4) / 255, Float(0x42) / 255))
    }

    @Test func theNewLayersAreOnByDefaultAndTheKeysToggleThem() {
        let display = WorldDisplay()
        #expect(display.structures && display.soil && display.bondFlow)
        #expect(WorldKeys.action(for: "5") == .toggleStructures && WorldKeys.action(for: "6") == .toggleSoil)
    }

    // MARK: Metrics

    @Test func aRowOfAFeatureRunCarriesItsKeys() throws {
        let line = """
        {"step": 500, "n_alive": 800, "build_units": 4200, "build_mat_total": 31.5, "build_here_mean": 0.125, "bond_flux": 9000,
         "bond_flow_by_role": [[0, 1], [2, 3]], "arms_att_atk": null, "arms_key_R": 0.62, "comm_speaker_frac": 0.25,
         "farm_soil_total": 1234567, "bond_ok": true, "unrelated_key": 5, "buoy_heat": 3}
        """
        let snapshot = try #require(MetricsTailer.parse(Data(line.utf8)))
        #expect(snapshot.features == ["build_units": 4200, "build_mat_total": 31.5, "build_here_mean": 0.125, "bond_flux": 9000,
                                      "arms_key_R": 0.62, "comm_speaker_frac": 0.25, "farm_soil_total": 1234567])
        let groups = FeatureMetrics.groups(snapshot.features)
        #expect(groups.map(\.title) == ["Arms race", "Structures", "Bonds and roles", "Communication", "Farming"])
        #expect(groups[1].rows.map(\.label) == ["here mean", "mat total", "units"])
        #expect(groups[2].rows == [.init(key: "bond_flux", label: "flux", value: 9000)])
        #expect(FeatureMetrics.text(4200) == Numbers.grouped(4200))
        #expect(FeatureMetrics.text(0.125) == "0.125" && FeatureMetrics.text(31.5) == "31.50" && FeatureMetrics.text(1234.5) == "1234")
    }

    @Test func aRowWithoutFeaturesHasNoGroups() throws {
        let snapshot = try #require(MetricsTailer.parse(Data(#"{"step": 7, "n_alive": 3}"#.utf8)))
        #expect(snapshot.features.isEmpty && FeatureMetrics.groups(snapshot.features).isEmpty)
    }
}
