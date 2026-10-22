import Foundation
import Testing

@testable import AlifeCore

/// The words of the inspector card, from the fixture's records.
@Suite struct InspectorTests {
    private let uid0 = WorldFixtures.uid0

    private func row(_ card: InspectorCard, _ label: String) -> InspectorCard.Row? { card.rows.first { $0.label == label } }

    private func focus(_ store: WorldStore, _ edit: (inout PlayFocus) -> Void) throws -> WorldStore {
        var s = store
        var f = try #require(s.focus)
        edit(&f)
        s.apply(.focs(f))
        return s
    }

    // MARK: An organism

    @Test func aNewbornIsCompleteFromItsFocusRecord() throws {
        // uid0 + 100 is not in the last IDNT (born since); its slots and labels come with the FOCS.
        let store = try WorldFixtures.store()
        #expect(store.identity(ofUID: uid0 + 100) == nil)
        let card = Inspector.organism(uid: uid0 + 100, store: store)
        #expect(card.title == "Organism \(uid0 + 100)" && card.alive)
        #expect(card.subtitle == "Cestria · House Aldric")
        #expect(row(card, "Archetype") == InspectorCard.Row("Archetype", "Gatherer", provisional: true))
        #expect(row(card, "Intent") == InspectorCard.Row("Intent", "Roaming", provisional: true))
        #expect(row(card, "Generation")?.value == "12" && row(card, "Bonds")?.value == "1")
        #expect(row(card, "Age")?.value == "4 steps (0.0 game min)")
        #expect(row(card, "Parent")?.value == "organism \(uid0 + 28)")
        #expect(row(card, "Clade") == nil, "the clade is the IDNT row's, which a newborn has not got yet")
        #expect(card.notes.contains("labels are provisional (labels v2)"))
        #expect(card.energy?.fraction == 0.25 && card.energy?.text == "4096 of 16384")
        #expect(row(card, "Follow") != nil, "the fixture's FOCS is followed")
    }

    @Test func theCladeComesFromTheIdentityRow() throws {
        let store = try WorldFixtures.store()
        let row0 = try #require(store.identity(ofUID: uid0 + 14))
        let focused = try focus(store) { $0.uid = uid0 + 14 }
        let card = Inspector.organism(uid: uid0 + 14, store: focused)
        #expect(row(card, "Clade")?.value == String(row0.clade))
    }

    @Test func aParentIsSelectableWhenItIsInTheFrame() throws {
        // The fixture's newest frame holds uid0 + 7 (row 1) and not uid0 + 28 (row 4 was reused).
        let store = try WorldFixtures.store()
        #expect(Inspector.organism(uid: uid0 + 100, store: store).selectableParent == nil)
        let alive = try focus(store) { $0.parentUid = uid0 + 7 }
        #expect(Inspector.organism(uid: uid0 + 100, store: alive).selectableParent == uid0 + 7)
        let founder = try focus(store) { $0.parentUid = -1 }
        let card = Inspector.organism(uid: uid0 + 100, store: founder)
        #expect(row(card, "Parent")?.value == "none (a founder or an immigrant)" && card.selectableParent == nil)
    }

    @Test func aFocusOfAnotherOrganismIsNotThisOnes() throws {
        // uid0 is in the frame and in IDNT, but the FOCS is of uid0 + 100: the card has only what the frame knows.
        let store = try WorldFixtures.store()
        let card = Inspector.organism(uid: uid0, store: store)
        #expect(card.alive && card.subtitle == "waiting for its report")
        #expect(row(card, "Position")?.value == "1.2, 24.0")
        #expect(row(card, "Archetype") == nil && card.energy == nil)
    }

    @Test func aDeadOrganismSaysSoAndNamesItsSuccessor() throws {
        let store = try focus(try WorldFixtures.store()) { $0.flags = 0 }
        let card = Inspector.organism(uid: uid0 + 100, store: store, successor: uid0 + 7)
        #expect(!card.alive && card.subtitle == "no longer alive" && card.successor == uid0 + 7)
        #expect(row(card, "Follow passed to")?.value == "organism \(uid0 + 7)")
        #expect(card.energy == nil)
        #expect(Inspector.organism(uid: uid0 + 100, store: store).rows.isEmpty, "no successor known: nothing to say")
    }

    @Test func anOrganismThatIsNotInTheWorldSaysSo() throws {
        let store = try WorldFixtures.store()
        let card = Inspector.organism(uid: 424_242, store: store)
        #expect(!card.alive && card.subtitle == "not in the world")
        // Before the first frame nothing is known, so it is not declared dead.
        #expect(Inspector.organism(uid: 424_242, store: WorldStore()).alive)
    }

    @Test func aForkOfCladesIsMarkedApproximate() throws {
        var store = try WorldFixtures.store()
        var record = try #require(store.identities)
        record.tables.exact = false
        store.apply(.idnt(record))
        let card = Inspector.organism(uid: uid0 + 100, store: store)
        #expect(card.subtitle == "Cestria · House Aldric (approximate)")
        #expect(card.notes.contains { $0.contains("dynasties approximate") })
    }

    @Test func labelsAreWarmingUpBeforeTheyAreEvaluatedAndABuddingOverlayIsAForecast() throws {
        let store = try WorldFixtures.store()
        let warming = try focus(store) { $0.labelFlags = 0 }
        let a = Inspector.organism(uid: uid0 + 100, store: warming)
        #expect(row(a, "Archetype")?.value == "warming up" && row(a, "Intent")?.value == "warming up")
        let budding = try focus(store) { $0.labelFlags = 3; $0.intent = 1 }
        #expect(row(Inspector.organism(uid: uid0 + 100, store: budding), "Intent")?.value == "Budding (forecast)")
        let two = try focus(store) { $0.archSecondary = 2 }
        #expect(row(Inspector.organism(uid: uid0 + 100, store: two), "Archetype")?.value == "Gatherer / Hunter")
    }

    @Test func reservedSlotsAreWordsInTheSubtitle() throws {
        let store = try focus(try WorldFixtures.store()) { $0.kin = 254; $0.house = 253 }
        #expect(Inspector.organism(uid: uid0 + 100, store: store).subtitle == "other kin groups · House unknown lineage")
    }

    @Test func theEnergyBarIsClampedToItsCap() throws {
        let over = try focus(try WorldFixtures.store()) { $0.energy = 99_999 }
        #expect(Inspector.organism(uid: uid0 + 100, store: over).energy?.fraction == 1)
        let under = try focus(try WorldFixtures.store()) { $0.energy = -5 }
        #expect(Inspector.organism(uid: uid0 + 100, store: under).energy?.fraction == 0)
    }

    // MARK: The rest of the selections

    @Test func aStormsCard() throws {
        let store = try WorldFixtures.store()
        let card = Inspector.storm(id: 7, store: store)
        #expect(card.title == "Merrow" && card.subtitle == "northern hemisphere")
        #expect(row(card, "Peak vorticity")?.value == "0.75 f0" && row(card, "Age")?.value == "1250 steps (1.4 game min)")
        #expect(card.notes.isEmpty)
        let disturbance = Inspector.storm(id: 9, store: store)
        #expect(disturbance.title == "Disturbance 9" && disturbance.subtitle == "southern hemisphere, coasting")
        #expect(row(disturbance, "Split from")?.value == "storm 7")
        #expect(disturbance.notes.count == 1)
        let gone = Inspector.storm(id: 5, store: store)
        #expect(!gone.alive && gone.subtitle == "dissipated or merged")
    }

    @Test func aKinGroupsCard() throws {
        let store = try WorldFixtures.store()
        let card = try #require(Inspector.kin(id: 17, store: store))
        #expect(card.title == "Aldric kin")
        #expect(row(card, "Members")?.value == "3 (42.9% of the population)")
        #expect(row(card, "Births")?.value == "2" && row(card, "Deaths")?.value == "1 starved")
        #expect(row(card, "Territory")?.value.hasSuffix("cells (dominance)") == true)
        #expect(Inspector.kin(id: 99_999, store: store)?.alive == false)
    }

    @Test func aPoolsCard() throws {
        let store = try WorldFixtures.store()
        #expect(Inspector.pool(index: 0, store: store).rows.count == 2)
        #expect(Inspector.pool(index: 5, store: store).alive == false)
    }

    @Test func theCardOfASelection() throws {
        let store = try WorldFixtures.store()
        #expect(Inspector.card(for: .none, store: store) == nil)
        #expect(Inspector.card(for: .organism(uid: uid0 + 100), store: store)?.title == "Organism \(uid0 + 100)")
        #expect(Inspector.card(for: .storm(id: 7), store: store)?.title == "Merrow")
        #expect(Inspector.card(for: .kin(id: 17), store: store)?.title == "Aldric kin")
        #expect(Inspector.card(for: .pool(index: 0), store: store)?.title == "Warm pool")
        let withSuccessor = Inspector.card(for: .organism(uid: 55), store: store, successors: [55: 66])
        #expect(withSuccessor?.successor == 66)
    }

    @Test func aFirstLetterIsCapitalised() {
        #expect("disturbance 9".capitalizedFirst == "Disturbance 9" && "".capitalizedFirst == "" && "Merrow".capitalizedFirst == "Merrow")
    }
}
