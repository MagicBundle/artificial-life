import Foundation
import Testing

@testable import AlifeCore

@Suite struct WorldSelectionTests {
    private func event(kind: Int, fam: String, a: Int, b: Int = 0, x: Double? = nil, y: Double? = nil, imp: Int = 1, id: UInt64 = 1000) -> WorldEvent {
        WorldEvent(id: id, step: 1, kind: kind, fam: fam, a: a, b: b, x: x, y: y, z: nil, imp: imp, text: "t")
    }

    // MARK: The hand-off

    @Test func aHandOffMovesTheSelectionOnlyWhenItIsFromWhatIsSelected() {
        var selection = WorldSelection.organism(uid: 100)
        let first = selection.applyHandoff(event(kind: EventKind.focusHandoff, fam: "focus", a: 100, b: 200))
        #expect(first && selection == .organism(uid: 200))
        // A late hand-off of the old organism, after the player clicked another one, never overrides the click.
        selection = .organism(uid: 300)
        let late = selection.applyHandoff(event(kind: EventKind.focusHandoff, fam: "focus", a: 100, b: 200))
        #expect(!late && selection == .organism(uid: 300))
    }

    @Test func aHandOffIsNotAppliedToAnythingElse() {
        let handoff = event(kind: EventKind.focusHandoff, fam: "focus", a: 5, b: 6)
        for start in [WorldSelection.none, .storm(id: 5), .kin(id: 5), .pool(index: 5)] {
            var selection = start
            let applied = selection.applyHandoff(handoff)
            #expect(!applied && selection == start)
        }
        var selection = WorldSelection.organism(uid: 5)
        let death = selection.applyHandoff(event(kind: EventKind.focusDeath, fam: "focus", a: 5))
        let nobody = selection.applyHandoff(event(kind: EventKind.focusHandoff, fam: "focus", a: 5, b: -1))
        #expect(!death && !nobody, "a death moves nothing, nor does a hand-off to nobody")
        #expect(selection == .organism(uid: 5))
    }

    @Test func theOrganismWithUidZeroIsAnOrganismLikeAnyOther() {
        // The first founder of a world is uid 0: only the other families number their entities from 1.
        var selection = WorldSelection.organism(uid: 9)
        let toZero = selection.applyHandoff(event(kind: EventKind.focusHandoff, fam: "focus", a: 9, b: 0))
        #expect(toZero && selection == .organism(uid: 0))
        let fromZero = selection.applyHandoff(event(kind: EventKind.focusHandoff, fam: "focus", a: 0, b: 4))
        #expect(fromZero && selection == .organism(uid: 4))
        #expect(ChronicleAction.of(event(kind: EventKind.focusDeath, fam: "focus", a: 0, x: 1, y: 2)).select == .organism(uid: 0))
        #expect(ChronicleAction.of(event(kind: EventKind.focusHandoff, fam: "focus", a: 5, b: 0)).select == .organism(uid: 0))
        #expect(ChronicleAction.of(event(kind: EventKind.stormNamed, fam: "storm", a: 0)).select == .none, "storm ids start at 1")
    }

    @Test func aChainOfHandOffsIsFollowedOneByOne() {
        var selection = WorldSelection.organism(uid: 1)
        for (from, to) in [(1, 2), (2, 3), (3, 4)] {
            let applied = selection.applyHandoff(event(kind: EventKind.focusHandoff, fam: "focus", a: from, b: to))
            #expect(applied)
        }
        #expect(selection == .organism(uid: 4) && selection.organismUID == 4)
    }

    @Test func theFixturesHandOffIsAppliedOnceItsEventsAreReduced() throws {
        let store = try WorldFixtures.store()
        let handoff = try #require(store.chronicle.events.first { $0.kind == EventKind.focusHandoff })
        var selection = WorldSelection.organism(uid: Int64(handoff.a))
        let applied = selection.applyHandoff(handoff)
        #expect(applied && selection == .organism(uid: Int64(handoff.b)))
    }

    // MARK: The registry

    @Test func theKindNumbersAreTheHeadersRegistry() throws {
        let play = try #require(try WorldFixtures.store().play)
        #expect(Dictionary(uniqueKeysWithValues: play.events.map { (Int($0.key)!, $0.value) }) == EventKind.registry)
        #expect(EventKind.registry[EventKind.focusHandoff] == "focus_handoff")
        #expect(EventKind.registry[EventKind.focusDeath] == "focus_death")
        #expect(EventKind.registry[EventKind.stormNamed] == "storm_named")
    }

    // MARK: A click on the chronicle

    @Test func aStormEventFliesToItAndSelectsTheStorm() {
        let action = ChronicleAction.of(event(kind: EventKind.stormNamed, fam: "storm", a: 7, x: 4.5, y: 20.25))
        #expect(action == ChronicleAction(at: SIMD2(4.5, 20.25), select: .storm(id: 7)))
    }

    @Test func aKinEventSelectsTheKinGroup() {
        let action = ChronicleAction.of(event(kind: EventKind.kinBorn, fam: "kin", a: 17, x: 6, y: 24))
        #expect(action.select == .kin(id: 17) && action.at == SIMD2(6, 24))
        // A kin group that dissolved has no place.
        let gone = ChronicleAction.of(event(kind: EventKind.kinDissolved, fam: "kin", a: 17))
        #expect(gone.at == nil && gone.select == .kin(id: 17))
    }

    @Test func aHandOffSelectsTheNewOrganismAndADeathTheDeadOne() {
        let handoff = ChronicleAction.of(event(kind: EventKind.focusHandoff, fam: "focus", a: 100, b: 200, x: 1, y: 2))
        #expect(handoff.select == .organism(uid: 200))
        let death = ChronicleAction.of(event(kind: EventKind.focusDeath, fam: "focus", a: 100, x: 1, y: 2))
        #expect(death.select == .organism(uid: 100) && death.at == SIMD2(1, 2))
    }

    @Test func housesAndTheSystemSelectNothing() {
        let house = ChronicleAction.of(event(kind: EventKind.houseFounded, fam: "house", a: 5_000_000_000))
        #expect(house.select == .none && house.at == nil)
        let director = ChronicleAction.of(event(kind: EventKind.directorEntry, fam: "director", a: 0, x: 3, y: 4))
        #expect(director.select == .none && director.at == SIMD2(3, 4), "it still flies to where the director acted")
        #expect(ChronicleAction.of(event(kind: EventKind.checkpoint, fam: "system", a: 0)) == ChronicleAction(at: nil, select: .none))
    }
}
