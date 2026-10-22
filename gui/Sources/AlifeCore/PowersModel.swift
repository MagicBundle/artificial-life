import Foundation

// The Powers tab (DESIGN §34.7): the grid of what the player can do, the aim-preview-confirm exchange with the world, the
// units, the effects that run, and the cards of what happened. A pure state machine over the records and the lines, so the
// whole exchange is tested without a window: `power preview NAME id=N ...` and then `power NAME id=N ...` are the only
// lines it writes, the same id and the same values in both, and the world re-checks the second.

/// The powers the window knows by name, in the order of the grid.
package enum PowerCatalog {
    package static let order: [PowerCommand.Name] = [.food, .bloom, .blight, .strike]

    /// One line under each power's name (the server's header carries no description).
    package static let blurbs: [PowerCommand.Name: String] = [
        .food: "Raise or lower how fast food regrows everywhere.",
        .bloom: "Raise the fertility of a food patch.",
        .blight: "Turn a food patch off for a while.",
        .strike: "Kill part of the organisms in a column: a local act.",
    ]

    /// What the grid reserves for the storm levers (W4): shown, greyed, so that the rows are where they will be.
    package static let reserved: [(label: String, note: String)] = [
        ("Seed a storm", "needs the storm levers (W4)"),
        ("Warm a pool", "needs the storm levers (W4)"),
    ]

    /// The line key of a header parameter (the header says `kill_p`, the line says `kill`).
    package static func lineKey(ofHeader name: String) -> PowerCommand.Key? {
        name == "kill_p" ? .kill : PowerCommand.Key(rawValue: name)
    }

    package static func name(ofAction action: String, in powers: [String: PlayHeader.Power]) -> PowerCommand.Name? {
        powers.first { $0.value.action == action }.flatMap { PowerCommand.Name(rawValue: $0.key) }
    }
}

/// One parameter of a power and the values the world accepts for it.
package struct PowerParameter: Equatable, Sendable {
    package var key: PowerCommand.Key
    package var label: String
    /// The accepted values: one range, or two with a hole between them (Food's dead zone).
    package var ranges: [ClosedRange<Double>]
    package var increment: Double
    package var initial: Double

    package var lowest: Double { ranges.first?.lowerBound ?? 0 }
    package var highest: Double { ranges.last?.upperBound ?? 0 }

    /// `value` moved to the nearest accepted value.
    package func clamped(_ value: Double) -> Double {
        guard !ranges.isEmpty else { return value }
        if ranges.contains(where: { $0.contains(value) }) { return value }
        func gap(_ r: ClosedRange<Double>) -> Double { value < r.lowerBound ? r.lowerBound - value : value - r.upperBound }
        let nearest = ranges.min { gap($0) < gap($1) } ?? ranges[0]
        return min(max(value, nearest.lowerBound), nearest.upperBound)
    }
}

/// A power as the grid offers it: the header's bounds and `ENVS`'s state of it.
package struct PowerOffer: Equatable, Identifiable, Sendable {
    package enum Scope: String, Equatable, Sendable { case global, patch, column }

    package var name: PowerCommand.Name
    package var label: String
    package var blurb: String
    package var action: String
    package var cost: Int
    package var cooldown: Int
    package var scope: Scope
    /// The bounds of a timed power's duration, in steps; nil for a one-step power.
    package var steps: ClosedRange<Int>?
    package var parameters: [PowerParameter]
    /// Why it cannot be used now, in the world's words; nil when it can.
    package var busy: String?
    /// The step it can be used, when that is later than now.
    package var readyAt: Int?

    package var id: String { name.rawValue }
    package var usable: Bool { busy == nil }

    /// The fraction of the cooldown still to run at `step`, 0 when ready (the ring around the power).
    package func cooldownFraction(atStep step: Int) -> Double {
        guard let readyAt, readyAt > step else { return 0 }
        return cooldown > 0 ? min(Double(readyAt - step) / Double(cooldown), 1) : 1
    }

    /// The values a power starts at: Food x1.3 for 6,000 steps, the middle of the other ranges.
    package var startingValues: [PowerCommand.Key: Double] {
        var out: [PowerCommand.Key: Double] = [:]
        for parameter in parameters { out[parameter.key] = parameter.initial }
        if let steps { out[.steps] = Double(min(max(6000, steps.lowerBound), steps.upperBound)) }
        return out
    }

    /// The keys a power's line takes, in the order they are sent.
    package var lineKeys: [PowerCommand.Key] {
        switch name {
        case .food: [.mult, .steps]
        case .bloom: [.patch, .gain, .steps]
        case .blight: [.patch, .steps]
        case .strike: [.x, .y, .radius, .kill]
        }
    }

    /// The offers of a world, in the order of the grid; empty for a world that takes no powers.
    package static func offers(header: PlayHeader?, states: [String: PlayEnvironment.PowerState]) -> [PowerOffer] {
        guard let powers = header?.powers else { return [] }
        return PowerCatalog.order.compactMap { name -> PowerOffer? in
            guard let power = powers[name.rawValue] else { return nil }
            let scope = Scope(rawValue: power.scope ?? "") ?? (name == .strike ? .column : name == .food ? .global : .patch)
            var parameters: [PowerParameter] = []
            for (rawKey, bounds) in (power.params ?? [:]).sorted(by: { $0.key < $1.key }) {
                guard let key = PowerCatalog.lineKey(ofHeader: rawKey), let ranges = Self.ranges(bounds), !ranges.isEmpty else { continue }
                parameters.append(parameter(key, ranges))
            }
            let order = parameters.sorted { Self.index(of: $0.key) < Self.index(of: $1.key) }
            let state = states[name.rawValue]
            return PowerOffer(
                name: name, label: power.label, blurb: PowerCatalog.blurbs[name] ?? "", action: power.action, cost: power.cost,
                cooldown: power.cooldown, scope: scope,
                steps: power.duration.flatMap { $0.count == 2 ? $0[0]...$0[1] : nil }, parameters: order,
                busy: state?.busy, readyAt: state?.ready)
        }
    }

    private static func index(of key: PowerCommand.Key) -> Int {
        [PowerCommand.Key.mult, .gain, .radius, .kill].firstIndex(of: key) ?? 9
    }

    /// `[lo, hi]` or `[[lo, hi], [lo, hi]]`.
    private static func ranges(_ value: JSONValue) -> [ClosedRange<Double>]? {
        guard let items = value.array else { return nil }
        if items.count == 2, let lo = items[0].number, let hi = items[1].number, lo <= hi { return [lo...hi] }
        let nested = items.compactMap { item -> ClosedRange<Double>? in
            guard let pair = item.array, pair.count == 2, let lo = pair[0].number, let hi = pair[1].number, lo <= hi else { return nil }
            return lo...hi
        }
        return nested.count == items.count ? nested : nil
    }

    private static func parameter(_ key: PowerCommand.Key, _ ranges: [ClosedRange<Double>]) -> PowerParameter {
        let low = ranges.first!.lowerBound, high = ranges.last!.upperBound
        let middle = ((low + high) / 2 * 10_000).rounded() / 10_000  // (0.3 and 0.6 give 0.45, not 0.44999...)
        var parameter = PowerParameter(key: key, label: key.rawValue, ranges: ranges, increment: 1, initial: low)
        switch key {
        case .mult:
            parameter.label = "Regrowth"
            parameter.increment = 0.05
            parameter.initial = parameter.clamped(1.3)
        case .gain:
            parameter.label = "Fertility"
            parameter.increment = 0.05
            parameter.initial = middle
        case .radius:
            parameter.label = "Radius"
            parameter.increment = max((high - low) / 40, 0.05)
            parameter.initial = middle
        case .kill:
            parameter.label = "Kill chance"
            parameter.increment = 0.05
            parameter.initial = middle
        default:
            break
        }
        return parameter
    }
}

/// Where a power lands.
package enum PowerTarget: Equatable, Sendable {
    case patch(Int)
    case point(x: Double, y: Double)
}

/// The exchange of one power with the world.
package struct PowersModel: Equatable, Sendable {
    package enum Phase: Equatable, Sendable {
        case idle
        /// A power is chosen and the map is waiting for its target.
        case aiming
        /// A preview was asked for.
        case previewing(id: Int)
        /// The world described what the power would do: Confirm sends it.
        case previewed(id: Int)
        /// The world refused the preview or the power: the reason is the world's.
        case refused(id: Int, reason: String)
        /// The power was sent; the world will say `queued` and then `applied`.
        case submitted(id: Int)
    }

    package private(set) var phase = Phase.idle
    package private(set) var offer: PowerOffer?
    package private(set) var values: [PowerCommand.Key: Double] = [:]
    package private(set) var target: PowerTarget?
    package private(set) var preview: PlanBody?
    /// Cost 1 powers go without the second click (a setting; off).
    package var skipsConfirmForCostOne = false
    private var nextID: Int

    package init(firstID: Int = 1) { nextID = max(firstID, 0) }

    package var selected: PowerCommand.Name? { offer?.name }

    /// A preview is in view: `Return` sends the power (and does nothing otherwise).
    package var canConfirm: Bool {
        if case .previewed = phase { return true }
        return false
    }

    /// A patch or column power is chosen and not yet sent: a click on the map is its target.
    package var isAiming: Bool {
        guard let offer, offer.scope != .global else { return false }
        switch phase {
        case .aiming, .previewing, .previewed, .refused: return true
        case .idle, .submitted: return false
        }
    }

    /// The id the next request takes: past every id the window has seen (a resumed world's cards keep theirs).
    package mutating func see(_ log: PlanLog) {
        if let top = log.records.map(\.plan.id).max() { nextID = max(nextID, top + 1) }
    }

    private mutating func allocate() -> Int {
        defer { nextID += 1 }
        return nextID
    }

    // MARK: Choosing and aiming

    /// Chooses a power. A global one (Food) is previewed at once; the others wait for a target. Nil when it cannot be used now.
    package mutating func choose(_ offer: PowerOffer) -> PlayCommand? {
        guard offer.usable else { return nil }
        reset()
        self.offer = offer
        values = offer.startingValues
        if offer.scope == .global { return requestPreview() }
        phase = .aiming
        return nil
    }

    /// A click on the map at the world point `point` (on the floor): the patch nearest to it, or the column over it.
    package mutating func aim(at point: SIMD2<Float>, patches: [PlayEnvironment.Patch], size: Float) -> PlayCommand? {
        guard let offer, offer.scope != .global else { return nil }
        switch phase {
        case .submitted, .idle: return nil
        default: break
        }
        switch offer.scope {
        case .patch:
            guard let index = Self.nearestPatch(to: point, in: patches, size: size) else { return nil }
            target = .patch(index)
        case .column:
            target = .point(x: Double(point.x), y: Double(point.y))
        case .global:
            return nil
        }
        return requestPreview()
    }

    /// The patch nearest to `point` on the torus of side `size`; the lowest index on a tie. Nil with no patch.
    package static func nearestPatch(to point: SIMD2<Float>, in patches: [PlayEnvironment.Patch], size: Float) -> Int? {
        var best: (index: Int, distance: Float)?
        for patch in patches {
            var dx = Float(patch.x) - point.x, dy = Float(patch.y) - point.y
            if size > 0 {
                dx -= size * (dx / size).rounded()
                dy -= size * (dy / size).rounded()
            }
            let d = dx * dx + dy * dy
            if best == nil || d < best!.distance { best = (patch.i, d) }
        }
        return best?.index
    }

    /// Moves a slider. The preview in view is stale from now on, so a new one is asked for when the power has what it needs.
    package mutating func set(_ key: PowerCommand.Key, to value: Double) -> PlayCommand? {
        guard let offer else { return nil }
        let bounded: Double
        if key == .steps, let steps = offer.steps {
            bounded = min(max((value / 250).rounded() * 250, Double(steps.lowerBound)), Double(steps.upperBound))
        } else if let parameter = offer.parameters.first(where: { $0.key == key }) {
            bounded = parameter.clamped(value)
        } else {
            return nil
        }
        values[key] = bounded
        switch phase {
        case .idle, .aiming, .submitted: return nil
        default: break
        }
        return ready ? requestPreview() : nil
    }

    private var ready: Bool {
        guard let offer else { return false }
        return offer.scope == .global || target != nil
    }

    // MARK: Lines

    private mutating func requestPreview() -> PlayCommand? {
        guard offer != nil, ready else { return nil }
        let id = allocate()
        guard let command = line(.preview, id: id) else { return nil }
        phase = .previewing(id: id)
        preview = nil
        return command
    }

    private func line(_ verb: PowerCommand.Verb, id: Int) -> PlayCommand? {
        guard let offer else { return nil }
        var parameters: [PowerCommand.Parameter] = []
        for key in offer.lineKeys {
            switch key {
            case .patch:
                guard case .patch(let index)? = target else { return nil }
                parameters.append(.init(.patch, Double(index)))
            case .x, .y:
                guard case .point(let x, let y)? = target else { return nil }
                parameters.append(.init(key, key == .x ? x : y))
            default:
                guard let value = values[key] else { return nil }
                parameters.append(.init(key, key == .steps ? value.rounded() : value))
            }
        }
        return .power(PowerCommand(verb, offer.name, id: id, parameters))
    }

    /// The second click: sends the power the player saw described. Nil unless a preview is in view.
    package mutating func confirm() -> PlayCommand? {
        guard case .previewed(let id) = phase, let command = line(.apply, id: id) else { return nil }
        phase = .submitted(id: id)
        return command
    }

    /// Ends a running effect early (no refund; the cooldown is that of the planned end).
    package mutating func cancelCommand(_ name: PowerCommand.Name) -> PlayCommand {
        .power(PowerCommand(.cancel, name, id: allocate()))
    }

    // MARK: What the world says

    /// A `PLAN` arrived. A reply to the request in flight moves the exchange along; anything else is none of its business (a
    /// since card, an old request). Returns the line to send when a cost-1 power goes without the second click.
    package mutating func receive(_ plan: PlayPlan) -> PlayCommand? {
        let body = plan.plan
        guard body.kind == .power, let id = currentID, body.id == id else { return nil }
        switch (phase, body.status) {
        case (.previewing, .preview):
            preview = body
            phase = .previewed(id: id)
            if skipsConfirmForCostOne, body.cost <= 1 { return confirm() }
        case (.previewing, .refused), (.submitted, .refused):
            preview = nil
            phase = .refused(id: id, reason: body.refused ?? body.text)
        case (.submitted, .applied), (.submitted, .cancelled):
            reset()
        default:
            break
        }
        return nil
    }

    private var currentID: Int? {
        switch phase {
        case .previewing(let id), .previewed(let id), .submitted(let id): id
        case .refused(let id, _): id
        case .idle, .aiming: nil
        }
    }

    // MARK: Leaving

    /// `Esc`: cancels aiming, else drops the preview. True when it did either (the caller goes on to the next level when not).
    package mutating func escape() -> Bool {
        switch phase {
        case .idle: return false
        case .submitted: return false  // sent: the world has it; the exchange is not the player's to take back
        default:
            reset()
            return true
        }
    }

    package mutating func reset() {
        phase = .idle
        offer = nil
        values = [:]
        target = nil
        preview = nil
    }
}

/// What the preview card shows.
package struct PowerPreviewCard: Equatable, Sendable {
    package var text: String
    package var cautions: [String]
    package var precedent: String?
    package var cost: Int
    package var haveAfter: Int?
    package var capUnits: Int?
    package var clamped: [String]
    package var holds: String?

    package init(_ plan: PlanBody) {
        text = plan.text
        cautions = plan.caution
        precedent = plan.precedent?.object?["line"]?.string
        cost = plan.cost
        haveAfter = plan.influence?.have
        capUnits = plan.influence?.cap
        clamped = plan.clamped
        holds = plan.holds
    }
}

/// The units of influence as pips: the free ones filled, each spent one with the step it comes back.
package struct InfluencePip: Equatable, Sendable {
    package var filled: Bool
    package var freeAt: Int?
}

package enum InfluenceMeter {
    package static func pips(_ influence: PlayEnvironment.Influence) -> [InfluencePip] {
        let cap = max(influence.cap, 0)
        let have = min(max(influence.have, 0), cap)
        let back = influence.freeAt.sorted()
        return (0..<cap).map { k in
            k < have ? InfluencePip(filled: true, freeAt: nil) : InfluencePip(filled: false, freeAt: k - have < back.count ? back[k - have] : nil)
        }
    }
}

/// An effect of the player's that runs now.
package struct ActiveEffect: Equatable, Identifiable, Sendable {
    package var name: PowerCommand.Name
    package var label: String
    package var start: Int
    package var end: Int
    package var id: String { "\(name.rawValue)@\(start)" }

    /// The player's timed effects in the newest `ENVS`, soonest end first.
    package static func running(_ store: WorldStore) -> [ActiveEffect] {
        let powers = store.play?.powers ?? [:]
        return (store.environment?.environment.effects ?? []).compactMap { effect -> ActiveEffect? in
            guard effect.source == "player", let end = effect.end,
                  let name = PowerCatalog.name(ofAction: effect.action, in: powers) else { return nil }
            return ActiveEffect(name: name, label: powers[name.rawValue]?.label ?? effect.action, start: effect.start, end: end)
        }.sorted { $0.end < $1.end }
    }
}

/// The card of one request in the Powers tab: what was asked, what the world did, and the rows since.
package struct PowerCardModel: Equatable, Identifiable, Sendable {
    package enum Status: String, Equatable, Sendable { case queued, applied, cancelled, refused, ended }

    package var id: Int
    package var status: Status
    package var text: String
    package var stepApplied: Int?
    package var ends: Int?
    /// The rows of the card as lines: step, living, deaths and births of the interval.
    package var rows: [String]
    package var disc: String?
    package var refusal: String?

    /// One card per request of the log, newest first. A preview is not a card.
    package static func cards(_ log: PlanLog, atStep step: Int) -> [PowerCardModel] {
        log.cards.compactMap { record -> PowerCardModel? in
            let plan = record.plan
            guard plan.kind == .power else { return nil }
            let status: Status
            switch plan.status {
            case .preview: return nil
            case .queued: status = .queued
            case .refused: status = .refused
            case .cancelled: status = .cancelled
            case .applied, .since:
                status = (plan.ends.map { $0 <= step } ?? false) ? .ended : .applied
            }
            let rows = (plan.since?.rows ?? []).map { row in
                "\(Numbers.grouped(row.step))  alive \(Numbers.grouped(row.alive))  born \(Numbers.grouped(row.born))"
                    + "  starved \(Numbers.grouped(row.starved))  killed \(Numbers.grouped(row.killed))"
                    + (row.storm > 0 ? "  storm \(Numbers.grouped(row.storm))" : "")
                    + (row.catastrophe > 0 ? "  strike \(Numbers.grouped(row.catastrophe))" : "")
            }
            let disc = plan.since?.disc.map { d in
                d.after.map { "In the column: \(Numbers.grouped(d.before)) before, \(Numbers.grouped($0)) after." }
                    ?? "In the column: \(Numbers.grouped(d.before)) before."
            }
            return PowerCardModel(
                id: plan.id, status: status, text: plan.text, stepApplied: plan.stepApplied, ends: plan.ends,
                rows: Array(rows.suffix(6)), disc: disc, refusal: plan.refused)
        }
    }
}
