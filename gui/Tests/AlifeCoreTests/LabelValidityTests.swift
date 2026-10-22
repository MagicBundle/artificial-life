import Foundation
import Testing

@testable import AlifeCore

/// The N1 gate on the labels of the window (DESIGN §33.7): valid only with a verdict row for the world and rule set, the label's
/// byte among those that pass, and the environment at its defaults.
@Suite struct LabelValidityTests {
    private let uid0 = WorldFixtures.uid0

    private func gate(found: Bool = true, archetypes: [Int] = [1, 2], intents: [Int] = [4]) -> PlayHeader.Labels.Gate {
        .init(rulesDigest: "d", worldKey: "k", keyFound: found, archetypes: archetypes, intents: intents)
    }

    private func labels(_ gate: PlayHeader.Labels.Gate?) -> PlayHeader.Labels {
        .init(intentVersion: 3, arch: ["", "Grazer", "Hunter", "Provider"], intent: ["", "Budding"], gate: gate)
    }

    private func env(_ levers: [String: JSONValue]) -> PlayEnvironment {
        .init(pools: [], season: nil, levers: levers, effects: [], cooldowns: [], influence: nil)
    }

    private func row(_ card: InspectorCard, _ label: String) -> InspectorCard.Row? { card.rows.first { $0.label == label } }

    @Test func aWorldWithoutAVerdictRowHasNoValidLabel() {
        for g in [nil, gate(found: false, archetypes: [], intents: [])] {
            let v = LabelValidity(labels: labels(g), environment: env([:]))
            #expect(!v.archetypeIsValid(1) && !v.intentIsValid(4) && !v.mixIsValid([1]))
            #expect(v.reason == "no held-out verdict for this world and rule set")
        }
        // even a list that the writer should not have sent is not read without a row
        let odd = LabelValidity(labels: labels(gate(found: false)), environment: env([:]))
        #expect(!odd.archetypeIsValid(1))
    }

    @Test func aPassingByteIsValidInADefaultEnvironmentOnly() {
        let v = LabelValidity(labels: labels(gate()), environment: env([:]))
        #expect(v.reason == nil)
        #expect(v.archetypeIsValid(1) && v.archetypeIsValid(2) && !v.archetypeIsValid(3) && !v.archetypeIsValid(0))
        #expect(v.intentIsValid(4) && !v.intentIsValid(1) && !v.intentIsValid(0))
        let moved = LabelValidity(labels: labels(gate()), environment: env(["regen_mult": .number(0.5)]))
        #expect(!moved.archetypeIsValid(1) && !moved.intentIsValid(4))
        #expect(moved.reason == "the environment is away from its defaults")
        let unknown = LabelValidity(labels: labels(gate()), environment: nil)
        #expect(!unknown.archetypeIsValid(1), "before the first ENVS nothing is known to be at its default")
    }

    @Test func aMixIsValidOnlyIfEveryArchetypeInItIs() {
        let v = LabelValidity(labels: labels(gate()), environment: env([:]))
        #expect(v.mixIsValid([0.6, 0.4, 0]))
        #expect(!v.mixIsValid([0.6, 0.3, 0.1]), "Provider has no passing verdict")
        #expect(v.mixIsValid([0.6, 0.395, 0.005]), "a share below 1% is not claimed")
    }

    @Test func theNoteSaysWhatIsShown() {
        let v = LabelValidity(labels: labels(gate()), environment: env([:]))
        #expect(v.note(allValid: true) == "labels are validated for this world (labels v3)")
        #expect(v.note(allValid: false) == "labels are provisional (labels v3: no passing verdict for this label)")
    }

    @Test func theInspectorTagsOnlyTheLabelsThatPass() throws {
        // The fixture's FOCS (uid0 + 100): archetype 1 (Grazer), intent Roaming (9), labels evaluated.
        let pass = PlayHeader.Labels.Gate(rulesDigest: "d", worldKey: "k", keyFound: true, archetypes: [1], intents: [9])
        let store = try WorldFixtures.store(gate: pass)
        let card = Inspector.organism(uid: uid0 + 100, store: store)
        #expect(row(card, "Archetype") == InspectorCard.Row("Archetype", "Grazer", provisional: false))
        #expect(row(card, "Intent") == InspectorCard.Row("Intent", "Roaming", provisional: false))
        #expect(card.notes.contains("labels are validated for this world (labels v3)"))
        // only the intent passes: the archetype row is tagged, the intent row is not
        let half = try WorldFixtures.store(gate: .init(rulesDigest: "d", worldKey: "k", keyFound: true, archetypes: [2], intents: [9]))
        let c2 = Inspector.organism(uid: uid0 + 100, store: half)
        #expect(row(c2, "Archetype")?.provisional == true && row(c2, "Intent")?.provisional == false)
        #expect(c2.notes.contains("labels are provisional (labels v3: no passing verdict for this label)"))
        // a lever away from its default withdraws both
        let moved = try WorldFixtures.store(gate: pass, levers: ["regen_mult": .number(0.5)])
        let c3 = Inspector.organism(uid: uid0 + 100, store: moved)
        #expect(row(c3, "Archetype")?.provisional == true && row(c3, "Intent")?.provisional == true)
        #expect(c3.notes.contains("labels are provisional (labels v3: the environment is away from its defaults)"))
        // a secondary archetype must pass as well
        var secondary = try WorldFixtures.store(gate: pass)
        var f = try #require(secondary.focus)
        f.archSecondary = 2
        secondary.apply(.focs(f))
        #expect(row(Inspector.organism(uid: uid0 + 100, store: secondary), "Archetype")?.provisional == true)
    }

    @Test func aWarmingUpLabelIsNeverValid() throws {
        let pass = PlayHeader.Labels.Gate(rulesDigest: "d", worldKey: "k", keyFound: true, archetypes: [1], intents: [9])
        var store = try WorldFixtures.store(gate: pass)
        var f = try #require(store.focus)
        f.labelFlags = 0
        store.apply(.focs(f))
        let card = Inspector.organism(uid: uid0 + 100, store: store)
        #expect(row(card, "Archetype") == InspectorCard.Row("Archetype", "warming up", provisional: true))
    }

    @Test func theStormFixtureHasTheNamesOfAStormWorld() throws {
        let play = try #require(try WorldFixtures.store().play)
        #expect(play.labels.arch[1] == "Grazer" && play.labels.gate?.keyFound == false)
        let plain = try #require(try WorldFixtures.store("2d").play)
        #expect(plain.labels.arch[1] == "Gatherer")
    }
}
