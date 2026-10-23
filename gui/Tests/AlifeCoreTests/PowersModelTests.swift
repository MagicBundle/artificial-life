import Foundation
import Testing

@testable import AlifeCore

/// The exchange of a power with the world (DESIGN §34.5, §34.7), as a state machine over lines and `PLAN` records: no window.
@Suite struct PowersModelTests {
    // MARK: A header with all four powers

    private static let table: [String: PlayHeader.Power] = [
        "food": .init(action: "regen_pulse", label: "Food", cost: 1, cooldown: 3000, scope: "global", duration: [3000, 12000],
                      params: ["mult": .array([.array([.number(0.6), .number(0.9)]), .array([.number(1.15), .number(1.5)])])]),
        "bloom": .init(action: "fertility_shift", label: "Bloom", cost: 1, cooldown: 3000, scope: "patch", duration: [3000, 12000],
                       params: ["gain": .array([.number(1.5), .number(2.0)])]),
        "blight": .init(action: "habitat_loss", label: "Blight", cost: 2, cooldown: 6000, scope: "patch", duration: [3000, 12000], params: [:]),
        "strike": .init(action: "catastrophe", label: "Strike", cost: 1, cooldown: 3000, scope: "column", duration: nil,
                        params: ["radius": .array([.number(6.25), .number(25)]), "kill_p": .array([.number(0.3), .number(0.6)])]),
    ]

    private func header() throws -> PlayHeader {
        var play = try #require(WorldFixtures.store("3d_g3b").header?.play)
        play.powers = Self.table
        return play
    }

    private func offers(states: [String: PlayEnvironment.PowerState] = [:]) throws -> [PowerOffer] {
        PowerOffer.offers(header: try header(), states: states)
    }

    private func offer(_ name: PowerCommand.Name, states: [String: PlayEnvironment.PowerState] = [:]) throws -> PowerOffer {
        try #require(offers(states: states).first { $0.name == name })
    }

    private let patches: [PlayEnvironment.Patch] = [.init(i: 0, x: 24, y: 12, gain: 1.5, sigma: 6), .init(i: 1, x: 8.5, y: 40, gain: 1.25, sigma: 4.5)]

    private func plan(_ id: Int, _ status: PlanBody.Status, cost: Int = 1, refused: String? = nil, kind: PlanBody.Kind = .power,
                      precedent: JSONValue? = nil, since: PlanBody.Since? = nil, ends: Int? = nil, stepApplied: Int? = nil) -> PlayPlan {
        let body = PlanBody(
            id: id, kind: kind, status: status, power: "food", action: "regen_pulse", raw: [:], params: [:], clamped: [],
            duration: 6000, stepApplied: stepApplied, ends: ends, cost: cost,
            influence: .init(have: 5, cap: 6, window: 18_000, freeAt: [30_000]), ready: nil, holds: "regen", text: "Food x1.3 for 6,000 steps.",
            caution: ["holds the regen lever until step 18,000"], precedent: precedent, since: since, refused: refused)
        return PlayPlan(head: RecordHead(step: 12_000, seq: 0, version: 1, flags: status == .preview ? 1 : 0), plan: body)
    }

    private func line(_ command: PlayCommand?) -> String? { try? command?.line() }

    // MARK: The offers

    @Test func theGridFollowsTheHeaderInTheOrderOfTheGrid() throws {
        let all = try offers()
        #expect(all.map(\.name) == [.food, .bloom, .blight, .strike] && all.map(\.cost) == [1, 1, 2, 1])
        let food = all[0]
        #expect(food.scope == .global && food.steps == 3000...12000 && food.cooldown == 3000)
        #expect(food.parameters.map(\.key) == [.mult] && food.parameters[0].ranges == [0.6...0.9, 1.15...1.5])
        #expect(food.parameters[0].initial == 1.3 && food.startingValues == [.mult: 1.3, .steps: 6000])
        let strike = all[3]
        #expect(strike.scope == .column && strike.steps == nil && strike.parameters.map(\.key) == [.radius, .kill])
        #expect(strike.startingValues == [.radius: 15.625, .kill: 0.45], "the middle of each range; the header's kill_p is the line's kill")
        #expect(all[1].scope == .patch && all[1].parameters.map(\.key) == [.gain] && all[2].parameters.isEmpty)
    }

    @Test func aWorldWithNoPowersOffersNone() throws {
        #expect(PowerOffer.offers(header: nil, states: [:]).isEmpty)
        let plain = try WorldFixtures.store("3d").header?.play
        #expect(PowerOffer.offers(header: plain, states: [:]).isEmpty)
    }

    @Test func busyAndReadyComeFromTheEnvironment() throws {
        let states: [String: PlayEnvironment.PowerState] = [
            "strike": .init(ready: 15_000, busy: "Strike is cooling down until step 15,000."), "food": .init(ready: nil, busy: nil),
        ]
        let all = try offers(states: states)
        #expect(all[0].usable && all[0].busy == nil && all[3].busy != nil && !all[3].usable && all[3].readyAt == 15_000)
        #expect(all[1].usable, "a power the environment says nothing about can be tried: the world refuses what it must")
        #expect(all[3].cooldownFraction(atStep: 12_000) == 1, "3,000 of 3,000 still to run")
        #expect(abs(all[3].cooldownFraction(atStep: 13_500) - 0.5) < 1e-9)
        #expect(all[3].cooldownFraction(atStep: 15_000) == 0 && all[0].cooldownFraction(atStep: 12_000) == 0)
    }

    @Test func aParameterTakesTheNearestAcceptedValue() throws {
        let mult = try offer(.food).parameters[0]
        #expect(mult.clamped(0.8) == 0.8 && mult.clamped(1.0) == 0.9 && mult.clamped(1.05) == 1.15 && mult.clamped(5) == 1.5)
        #expect(mult.clamped(0.1) == 0.6)
        #expect(mult.clamped(1.025) == 0.9, "half way across the hole: the lower edge, the gentler move")
        #expect(mult.lowest == 0.6 && mult.highest == 1.5)
    }

    // MARK: Food, the global power

    @Test func foodIsPreviewedAtOnceAndConfirmedWithTheSameIdAndValues() throws {
        var model = PowersModel()
        let preview = model.choose(try offer(.food))
        #expect(line(preview) == "power preview food id=1 mult=1.3 steps=6000")
        #expect(model.phase == .previewing(id: 1) && model.selected == .food && !model.isAiming)
        let early = model.confirm()
        #expect(early == nil, "nothing to confirm before the world has described it")
        let described = model.receive(plan(1, .preview))
        #expect(described == nil)
        #expect(model.phase == .previewed(id: 1) && model.preview?.text == "Food x1.3 for 6,000 steps.")
        #expect(model.canConfirm, "Return has something to send")
        let sent = model.confirm()
        #expect(line(sent) == "power food id=1 mult=1.3 steps=6000" && model.phase == .submitted(id: 1) && !model.canConfirm)
        let twice = model.confirm()
        #expect(twice == nil, "once")
        _ = model.receive(plan(1, .queued))
        #expect(model.phase == .submitted(id: 1))
        _ = model.receive(plan(1, .applied, ends: 18_000, stepApplied: 12_000))
        #expect(model.phase == .idle && model.selected == nil, "done: the card is in the log")
    }

    @Test func movingASliderMakesThePreviewStaleAndAsksAgain() throws {
        var model = PowersModel()
        _ = model.choose(try offer(.food))
        _ = model.receive(plan(1, .preview))
        let again = model.set(.mult, to: 0.8)
        #expect(line(again) == "power preview food id=2 mult=0.8 steps=6000")
        #expect(model.phase == .previewing(id: 2) && model.preview == nil, "the old description is gone")
        let stale = model.confirm()
        #expect(stale == nil)
        // The late answer to request 1 is none of its business.
        _ = model.receive(plan(1, .preview))
        #expect(model.phase == .previewing(id: 2))
        _ = model.receive(plan(2, .preview))
        let confirmed = model.confirm()
        #expect(line(confirmed) == "power food id=2 mult=0.8 steps=6000")
    }

    @Test func slidersStayInsideTheTableAndStepsGoToTheQuantum() throws {
        var model = PowersModel()
        _ = model.choose(try offer(.food))
        let dead = model.set(.mult, to: 1.0)
        #expect(line(dead)?.contains("mult=0.9") == true, "the dead zone is not a value")
        let odd = model.set(.steps, to: 7100)
        #expect(line(odd)?.hasSuffix("steps=7000") == true, "a multiple of 250")
        let high = model.set(.steps, to: 99_999)
        #expect(line(high)?.hasSuffix("steps=12000") == true)
        let low = model.set(.steps, to: 1)
        #expect(line(low)?.hasSuffix("steps=3000") == true)
        let foreign = model.set(.radius, to: 5)
        #expect(foreign == nil, "Food has no radius")
        #expect(model.values[.mult] == 0.9 && model.values[.steps] == 3000)
    }

    // MARK: Patches and columns

    @Test func aPatchPowerWaitsForAClickAndTakesTheNearestPatchOnTheTorus() throws {
        var model = PowersModel()
        let first = model.choose(try offer(.bloom))
        #expect(first == nil && model.phase == .aiming && model.isAiming && model.target == nil)
        // (47.5, 12) is 23.5 from patch 0 across the middle and 9 + 20 from patch 1 across the seam: patch 1 is nearer.
        let sent = model.aim(at: SIMD2(47.5, 12), patches: patches, size: 48)
        #expect(line(sent) == "power preview bloom id=1 patch=1 gain=1.75 steps=6000" && model.target == .patch(1))
        #expect(PowersModel.nearestPatch(to: SIMD2(47.5, 12), in: patches, size: 0) == 0, "without a torus the plain distance")
        #expect(PowersModel.nearestPatch(to: SIMD2(1, 1), in: [], size: 48) == nil)
    }

    @Test func aClickWithNoPatchInTheWorldDoesNothing() throws {
        var model = PowersModel()
        _ = model.choose(try offer(.blight))
        let sent = model.aim(at: SIMD2(1, 1), patches: [], size: 48)
        #expect(sent == nil && model.phase == .aiming && model.target == nil)
    }

    @Test func aStrikeNamesAColumnAndKeepsItsTargetWhenASliderMoves() throws {
        var model = PowersModel()
        _ = model.choose(try offer(.strike))
        let early = model.set(.radius, to: 10)
        #expect(early == nil, "nothing to preview until a target is aimed")
        #expect(model.values[.radius] == 10)
        let sent = model.aim(at: SIMD2(10.5, 20), patches: patches, size: 48)
        #expect(line(sent) == "power preview strike id=1 x=10.5 y=20 radius=10 kill=0.45")
        _ = model.receive(plan(1, .preview))
        let harder = model.set(.kill, to: 0.6)
        #expect(line(harder) == "power preview strike id=2 x=10.5 y=20 radius=10 kill=0.6")
        // Aiming again, at another place, previews again; the old preview is gone.
        _ = model.receive(plan(2, .preview))
        let elsewhere = model.aim(at: SIMD2(30, 30), patches: patches, size: 48)
        #expect(line(elsewhere) == "power preview strike id=3 x=30 y=30 radius=10 kill=0.6")
        #expect(model.preview == nil && model.isAiming)
    }

    @Test func aClickWhileASentPowerWaitsChangesNothing() throws {
        var model = PowersModel()
        _ = model.choose(try offer(.strike))
        _ = model.aim(at: SIMD2(10, 20), patches: patches, size: 48)
        _ = model.receive(plan(1, .preview))
        _ = model.confirm()
        #expect(!model.isAiming)
        let sent = model.aim(at: SIMD2(30, 30), patches: patches, size: 48)
        #expect(sent == nil && model.phase == .submitted(id: 1))
        let moved = model.set(.radius, to: 20)
        #expect(moved == nil, "no new preview while the sent power waits")
    }

    // MARK: Refusals, escape, the setting

    @Test func aRefusalIsTheWorldsAndTheNextTryStartsAgain() throws {
        var model = PowersModel()
        _ = model.choose(try offer(.strike))
        _ = model.aim(at: SIMD2(10, 20), patches: patches, size: 48)
        _ = model.receive(plan(1, .refused, refused: "Strike is cooling down until step 15,000."))
        #expect(model.phase == .refused(id: 1, reason: "Strike is cooling down until step 15,000.") && model.preview == nil)
        let none = model.confirm()
        #expect(none == nil)
        #expect(model.isAiming, "the map is still the target: a click tries again")
        let retry = model.aim(at: SIMD2(11, 21), patches: patches, size: 48)
        #expect(line(retry)?.hasPrefix("power preview strike id=2") == true)
    }

    @Test func aLateRefusalOfASentPowerIsShownToo() throws {
        var model = PowersModel()
        _ = model.choose(try offer(.food))
        _ = model.receive(plan(1, .preview))
        _ = model.confirm()
        _ = model.receive(plan(1, .refused, refused: "Food is busy."))
        #expect(model.phase == .refused(id: 1, reason: "Food is busy."))
    }

    @Test func escapeCancelsAimingThenDropsThePreviewAndNeverTakesBackASentPower() throws {
        var model = PowersModel()
        let nothing = model.escape()
        #expect(!nothing, "nothing to cancel: the next level of Esc is the caller's")
        _ = model.choose(try offer(.bloom))
        let aimed = model.escape()
        #expect(aimed && model.phase == .idle && model.selected == nil && !model.isAiming, "aiming cancelled")
        _ = model.choose(try offer(.food))
        _ = model.receive(plan(1, .preview))
        let previewed = model.escape()
        #expect(previewed && model.phase == .idle && model.preview == nil, "the preview dropped")
        _ = model.choose(try offer(.food))
        _ = model.receive(plan(2, .preview))
        _ = model.confirm()
        let sent = model.escape()
        #expect(!sent && model.phase == .submitted(id: 2), "sent: the world has it")
    }

    @Test func aBusyPowerIsNotChosen() throws {
        var model = PowersModel()
        let busy = try offer(.strike, states: ["strike": .init(ready: 15_000, busy: "cooling down")])
        let sent = model.choose(busy)
        #expect(sent == nil && model.phase == .idle && model.selected == nil)
    }

    @Test func aCostOneSettingSkipsTheSecondClickAndNeverForADearerPower() throws {
        var model = PowersModel()
        model.skipsConfirmForCostOne = true
        _ = model.choose(try offer(.food))
        let sent = model.receive(plan(1, .preview, cost: 1))
        #expect(line(sent) == "power food id=1 mult=1.3 steps=6000" && model.phase == .submitted(id: 1))
        model.reset()
        _ = model.choose(try offer(.food))
        let held = model.receive(plan(2, .preview, cost: 2))
        #expect(held == nil && model.phase == .previewed(id: 2), "a dearer power is always confirmed")
        model.reset()
        model.skipsConfirmForCostOne = false
        _ = model.choose(try offer(.food))
        let manual = model.receive(plan(3, .preview, cost: 1))
        #expect(manual == nil)
    }

    @Test func aReplyOfAnotherKindOrIdIsIgnored() throws {
        var model = PowersModel()
        _ = model.choose(try offer(.food))
        _ = model.receive(plan(1, .preview, kind: .intent))
        #expect(model.phase == .previewing(id: 1))
        _ = model.receive(plan(7, .preview))
        #expect(model.phase == .previewing(id: 1))
        var idle = PowersModel()
        let reply = idle.receive(plan(1, .preview))
        #expect(reply == nil && idle.phase == .idle)
    }

    @Test func idsGoPastEveryIdTheWindowHasSeen() throws {
        var model = PowersModel()
        var log = PlanLog()
        log.insert(plan(1, .applied))
        log.insert(plan(41, .since))
        log.insert(plan(5, .refused))
        model.see(log)
        let first = model.choose(try offer(.food))
        #expect(line(first)?.contains("id=42") == true, "a resumed world's cards keep their ids")
        model.see(log)  // seeing again never moves it back
        _ = model.receive(plan(42, .preview))
        _ = model.confirm()
        let cancel = model.cancelCommand(.food)
        #expect(line(cancel) == "power cancel food id=43")
        #expect(PowersModel(firstID: 100).phase == .idle)
        var seeded = PowersModel(firstID: 100)
        let seededCancel = seeded.cancelCommand(.bloom)
        #expect(line(seededCancel) == "power cancel bloom id=100")
    }

    // MARK: Influence, effects and cards

    @Test func theUnitsArePipsWithTheStepEachSpentOneComesBack() {
        let pips = InfluenceMeter.pips(.init(have: 4, cap: 6, window: 18_000, freeAt: [15_500, 13_000]))
        #expect(pips == [.init(filled: true, freeAt: nil), .init(filled: true, freeAt: nil), .init(filled: true, freeAt: nil),
                         .init(filled: true, freeAt: nil), .init(filled: false, freeAt: 13_000), .init(filled: false, freeAt: 15_500)])
        let overfull = InfluenceMeter.pips(.init(have: 9, cap: 6, window: 1, freeAt: []))
        #expect(overfull.count == 6 && overfull.allSatisfy { $0.filled })
        #expect(InfluenceMeter.pips(.init(have: 0, cap: 2, window: 1, freeAt: [5])).map(\.freeAt) == [5, nil])
        #expect(InfluenceMeter.pips(.init(have: 0, cap: 0, window: 1, freeAt: [])).isEmpty)
    }

    @Test func theRunningEffectsAreThePlayersTimedOnes() throws {
        var store = try WorldFixtures.store("3d_g3b")
        var header = try #require(store.header)
        header.play?.powers = Self.table
        var fresh = WorldStore()
        fresh.reset(header: header)
        for record in [store.environment.map(StreamRecord.envs)].compactMap({ $0 }) { fresh.apply(record) }
        store = fresh
        #expect(ActiveEffect.running(store) == [.init(name: .food, label: "Food", start: 12_000, end: 18_000)])
        #expect(ActiveEffect.running(store).first?.id == "food@12000")
    }

    @Test func thePreviewCardCarriesTheWorldsWordsAndThePrecedentLine() {
        let precedent = JSONValue.object(["line": .string("Earlier x0.8 pulses (14 in the 3D director runs): the population fell 13%."), "n": .number(14)])
        let card = PowerPreviewCard(plan(1, .preview, precedent: precedent).plan)
        #expect(card.text == "Food x1.3 for 6,000 steps." && card.cautions == ["holds the regen lever until step 18,000"])
        #expect(card.precedent?.hasPrefix("Earlier x0.8 pulses") == true && card.cost == 1 && card.haveAfter == 5 && card.capUnits == 6)
        #expect(card.holds == "regen" && card.clamped.isEmpty)
        #expect(PowerPreviewCard(plan(1, .preview).plan).precedent == nil, "a table with no row")
    }

    @Test func theCardsAreOnePerRequestWithTheirRows() {
        var log = PlanLog()
        log.insert(plan(1, .preview))
        log.insert(plan(2, .refused, refused: "Strike is cooling down."))
        let since = PlanBody.Since(
            from: 12_000, rows: [
                .init(step: 12_000, alive: 42_873, starved: 5_000, killed: 5_100, born: 7_000, storm: 0, catastrophe: 0),
                .init(step: 12_250, alive: 37_310, starved: 8_249, killed: 5_000, born: 7_432, storm: 12, catastrophe: 130),
            ], disc: .init(before: 160, after: 61))
        log.insert(plan(3, .queued))
        log.insert(plan(3, .applied, ends: 18_000, stepApplied: 12_000))
        log.insert(plan(3, .since, since: since, ends: 18_000, stepApplied: 12_000))
        log.insert(plan(9, .preview, kind: .intent))
        let cards = PowerCardModel.cards(log, atStep: 12_300)
        #expect(cards.map(\.id) == [3, 2], "a preview and an intent are not cards")
        #expect(cards[0].status == .applied && cards[0].stepApplied == 12_000 && cards[0].ends == 18_000)
        #expect(cards[0].rows == ["12,000  alive 42,873  born 7,000  starved 5,000  killed 5,100",
                                  "12,250  alive 37,310  born 7,432  starved 8,249  killed 5,000  storm 12  strike 130"])
        #expect(cards[0].disc == "In the column: 160 before, 61 after.")
        #expect(cards[1].status == .refused && cards[1].refusal == "Strike is cooling down.")
        #expect(PowerCardModel.cards(log, atStep: 18_000)[0].status == .ended, "the effect has run out")
        let open = PlanBody.Since(from: 1, rows: [], disc: .init(before: 5, after: nil))
        var other = PlanLog()
        other.insert(plan(4, .since, since: open))
        #expect(PowerCardModel.cards(other, atStep: 0)[0].disc == "In the column: 5 before.")
    }

    @Test func aCancelledRequestIsMarked() {
        var log = PlanLog()
        log.insert(plan(3, .applied))
        log.insert(plan(3, .cancelled))
        #expect(PowerCardModel.cards(log, atStep: 0).first?.status == .cancelled)
        var many = PlanLog()
        for id in 0..<80 { many.insert(plan(id, .applied)) }
        #expect(PowerCardModel.cards(many, atStep: 0).count == PlanLog.capacity)
    }

    @Test func theReservedRowsAreTheStormLevers() {
        let notes = PowerCatalog.reserved.map(\.note)
        #expect(notes.count == 2 && notes.allSatisfy { $0.contains("W4") })
        #expect(PowerCatalog.order.map(\.rawValue) == ["food", "bloom", "blight", "strike"])
        #expect(PowerCatalog.lineKey(ofHeader: "kill_p") == .kill && PowerCatalog.lineKey(ofHeader: "radius") == .radius)
        #expect(PowerCatalog.lineKey(ofHeader: "nope") == nil)
        #expect(PowerCatalog.name(ofAction: "catastrophe", in: Self.table) == .strike && PowerCatalog.name(ofAction: "x", in: Self.table) == nil)
    }
}
