import Foundation

// What the World window knows of the world it is connected to, reduced from the decoded records (DESIGN §33.4.2). A value
// type with no UI and no clock, so that the reducers are tested alone. A latest-wins record (`FRMA IDNT INTN TERR STRM
// MAP1 MAPH ENVS CLCK FOCS`) replaces the one before it, and is complete by itself (§33.1.1); `EVNT` is queued: every event is
// kept, in order, once.

/// The chronicle: the events of the world in the order of their ids (`step * 1000 + family base + index`), without
/// repeats, newest `capacity` kept. An event the world sends again (a resume, a backlog that overlaps the live feed) is
/// recognised by its id.
package struct Chronicle: Equatable, Sendable {
    package static let capacity = 500

    /// Ascending by `id`.
    package private(set) var events: [WorldEvent] = []
    /// Ids at or below this were trimmed from the ring: an event with such an id is not new.
    private var floor: UInt64 = 0

    package init() {}

    /// The events of `batch` that were not known, in order of arrival; they are now in the chronicle.
    package mutating func insert(_ batch: [WorldEvent]) -> [WorldEvent] {
        var fresh: [WorldEvent] = []
        for event in batch {
            guard event.id > floor, let at = insertionIndex(of: event.id) else { continue }
            events.insert(event, at: at)
            fresh.append(event)
        }
        if events.count > Self.capacity {
            let extra = events.count - Self.capacity
            floor = max(floor, events[extra - 1].id)
            events.removeFirst(extra)
        }
        return fresh
    }

    /// Where an event of `id` goes to keep the ids ascending; nil when one with that id is there.
    private func insertionIndex(of id: UInt64) -> Int? {
        var low = 0, high = events.count
        while low < high {
            let mid = (low + high) / 2
            if events[mid].id == id { return nil }
            if events[mid].id < id { low = mid + 1 } else { high = mid }
        }
        return low
    }

    /// Newest first, as the window lists them.
    package var newestFirst: [WorldEvent] { events.reversed() }
}

/// What a delivery did to one alert of the board.
package struct AlertChange: Equatable, Sendable {
    package enum Kind: Equatable, Sendable {
        /// An id the board had not seen (a first sight of an alert already updated or cleared included).
        case raised
        /// A higher version of a known id whose condition still holds.
        case updated
        /// The end of the alert's condition: the row is removed.
        case cleared
    }

    package var kind: Kind
    package var alert: PlayAlert
}

/// The alerts of the world, by id (DESIGN §34.3, §34.6): the newest `version` of each, in order of their ids, newest
/// `capacity` kept. A cleared alert stays in the ring (so a late older version cannot bring it back) but is not on the stack.
/// Acknowledgement is the player's, and only an explicit one: nothing here acknowledges by itself.
package struct AlertBoard: Equatable, Sendable {
    package static let capacity = 200

    /// Ascending by `id`, the newest version of each, cleared ones included.
    package private(set) var alerts: [PlayAlert] = []
    /// Ids the player acknowledged (the `ack` line was sent for them).
    package private(set) var acknowledged: Set<UInt64> = []
    /// Ids at or below this were trimmed from the ring: an alert with such an id is not new.
    private var floor: UInt64 = 0

    package init() {}

    /// What `batch` changed, in the order of the batch: an unknown id is `raised`, a known one with a higher version is
    /// `updated` (or `cleared`), anything else is ignored.
    @discardableResult
    package mutating func insert(_ batch: [PlayAlert]) -> [AlertChange] {
        var changes: [AlertChange] = []
        for alert in batch where alert.id > floor {
            let (found, at) = position(of: alert.id)
            if found {
                guard alert.version > alerts[at].version else { continue }
                alerts[at] = alert
                changes.append(AlertChange(kind: alert.state == .cleared ? .cleared : .updated, alert: alert))
            } else {
                alerts.insert(alert, at: at)
                changes.append(AlertChange(kind: alert.state == .cleared ? .cleared : .raised, alert: alert))
            }
        }
        if alerts.count > Self.capacity {
            let extra = alerts.count - Self.capacity
            floor = max(floor, alerts[extra - 1].id)
            let dropped = Set(alerts[..<extra].map(\.id))
            alerts.removeFirst(extra)
            acknowledged.subtract(dropped)
        }
        return changes
    }

    private func position(of id: UInt64) -> (found: Bool, index: Int) {
        var low = 0, high = alerts.count
        while low < high {
            let mid = (low + high) / 2
            if alerts[mid].id == id { return (true, mid) }
            if alerts[mid].id < id { low = mid + 1 } else { high = mid }
        }
        return (false, low)
    }

    package func alert(id: UInt64) -> PlayAlert? {
        let (found, at) = position(of: id)
        return found ? alerts[at] : nil
    }

    /// The player's acknowledgement of a known alert; false for an unknown one.
    @discardableResult
    package mutating func acknowledge(id: UInt64) -> Bool {
        guard alert(id: id) != nil else { return false }
        acknowledged.insert(id)
        return true
    }

    /// Whether `alert` is on the stack at `step`: surface `alert`, not cleared, not acknowledged, and not past its `expires`
    /// (an alert with no expiry, the importance-3 ones, stays until it clears or is acknowledged).
    package func isOnStack(_ alert: PlayAlert, atStep step: Int) -> Bool {
        alert.surface == .alert && alert.state != .cleared && !acknowledged.contains(alert.id)
            && alert.expires.map { $0 > step } ?? true
    }

    /// The stack, newest first.
    package func stack(atStep step: Int) -> [PlayAlert] {
        alerts.reversed().filter { isOnStack($0, atStep: step) }
    }

    /// The alerts of surface `log` that are not cleared (chronicle rows and ticker lines), newest first.
    package var logRows: [PlayAlert] {
        alerts.reversed().filter { $0.surface == .log && $0.state != .cleared }
    }

    /// The newest alert on the stack that the player has not acknowledged (`k` acknowledges it).
    package func newestUnacknowledged(atStep step: Int) -> PlayAlert? {
        stack(atStep: step).first
    }

    /// The alerts to toast for `changes`: a newly raised alert of the stack surface at importance 2 or more, and not the
    /// backlog loaded at the start (the history of a re-opened world is not news).
    package static func toasts(from changes: [AlertChange], backlog: Bool) -> [PlayAlert] {
        guard !backlog else { return [] }
        return changes.filter { $0.kind == .raised && $0.alert.surface == .alert && $0.alert.imp >= 2 }.map(\.alert)
    }
}

/// The reports of the chapters (DESIGN §34.4): the newest record per `(chapter, mode)`, the newest `capacity` chapters kept.
package struct ReportShelf: Equatable, Sendable {
    package static let capacity = 24

    package struct Key: Hashable, Sendable {
        package var chapter: Int
        package var mode: Int
    }

    package private(set) var entries: [Key: PlayReport] = [:]

    package init() {}

    /// Keeps `record` unless the shelf holds a newer one of its key; a pending report never replaces a final one. True if
    /// the shelf changed.
    @discardableResult
    package mutating func insert(_ record: PlayReport) -> Bool {
        let key = Key(chapter: record.chapter, mode: record.mode)
        if let held = entries[key] {
            if held.isFinal && !record.isFinal { return false }
            if record.head.step < held.head.step { return false }
            if held == record { return false }
        }
        entries[key] = record
        let chapters = Set(entries.keys.map(\.chapter))
        if chapters.count > Self.capacity, let oldest = chapters.min() {
            entries = entries.filter { $0.key.chapter != oldest }
        }
        return entries[key] != nil
    }

    package func report(chapter: Int, mode: Int) -> PlayReport? { entries[Key(chapter: chapter, mode: mode)] }

    /// The report of the chapter with absolute number `nAbs` (a select report that has arrived, else the template one).
    package func report(nAbs: Int) -> PlayReport? {
        let held = entries.values.filter { $0.report.nAbs == nAbs }
        return held.first { $0.mode == 1 && $0.isFinal } ?? held.first { $0.mode == 0 } ?? held.first { $0.mode == 1 }
    }

    /// The chapters held, newest first.
    package var chapters: [Int] { Set(entries.keys.map(\.chapter)).sorted(by: >) }

    /// The report to show for `chapter`: a select report that has arrived, else the template one, else a pending select
    /// (G3c). The away report (mode 2) is its own thing and is never this.
    package func current(chapter: Int) -> PlayReport? {
        if let select = report(chapter: chapter, mode: 1), select.isFinal { return select }
        return report(chapter: chapter, mode: 0) ?? report(chapter: chapter, mode: 1)
    }

    /// `current` for every chapter held, newest first.
    package var newestFirst: [PlayReport] { chapters.compactMap { current(chapter: $0) } }
}

/// What the world said about the player's powers (DESIGN §34.5): the last `capacity` `PLAN` records in arrival order, and the
/// card of each request, which is the newest record of its `id` (a since card updates in place).
package struct PlanLog: Equatable, Sendable {
    package static let capacity = 50

    package private(set) var records: [PlayPlan] = []

    package init() {}

    package mutating func insert(_ plan: PlayPlan) {
        records.append(plan)
        if records.count > Self.capacity { records.removeFirst(records.count - Self.capacity) }
    }

    /// The newest record of request `id`, nil if it has none in the log.
    package func card(id: Int) -> PlayPlan? { records.last { $0.plan.id == id } }

    /// One card per request, newest request first.
    package var cards: [PlayPlan] {
        var seen: Set<Int> = []
        return records.reversed().filter { seen.insert($0.plan.id).inserted }
    }

    package var newest: PlayPlan? { records.last }
}

package struct WorldStore: Sendable {
    /// What a record changed.
    package struct Changes: OptionSet, Sendable {
        package let rawValue: Int
        package init(rawValue: Int) { self.rawValue = rawValue }
        package static let frame = Changes(rawValue: 1 << 0)
        package static let identities = Changes(rawValue: 1 << 1)
        package static let labels = Changes(rawValue: 1 << 2)
        package static let territory = Changes(rawValue: 1 << 3)
        package static let storms = Changes(rawValue: 1 << 4)
        package static let map = Changes(rawValue: 1 << 5)
        package static let environment = Changes(rawValue: 1 << 6)
        package static let clock = Changes(rawValue: 1 << 7)
        package static let focus = Changes(rawValue: 1 << 8)
        package static let events = Changes(rawValue: 1 << 9)
        package static let header = Changes(rawValue: 1 << 10)
        package static let alerts = Changes(rawValue: 1 << 11)
        package static let reports = Changes(rawValue: 1 << 12)
        package static let plans = Changes(rawValue: 1 << 13)
        package static let material = Changes(rawValue: 1 << 14)
        package static let soil = Changes(rawValue: 1 << 15)
        package static let voice = Changes(rawValue: 1 << 16)
    }

    package private(set) var header: StreamHeader?
    package private(set) var frame: PlayFrame?
    package private(set) var identities: PlayIdentities?
    package private(set) var labels: PlayLabels?
    package private(set) var territory: PlayTerritory?
    package private(set) var storms: PlayStormRecord?
    package private(set) var map: PlayWeatherMap?
    /// A sphere world's weather map (`MAPH`); `map` is nil there, and this is nil in a box world.
    package private(set) var sphereMap: PlaySphereWeather?
    package private(set) var environment: PlayEnvironmentRecord?
    package private(set) var clock: PlayClock?
    package private(set) var focus: PlayFocus?
    /// The feature layers (DESIGN §45): the built material, the farmed soil, the focused organism's voice. Nil without the feature.
    package private(set) var material: PlayCells?
    package private(set) var soil: PlayCells?
    package private(set) var voice: PlayVoice?
    package private(set) var chronicle = Chronicle()
    package private(set) var alerts = AlertBoard()
    package private(set) var reports = ReportShelf()
    package private(set) var plans = PlanLog()
    /// Increase with every record of the kind that arrives (the renderer refreshes a buffer set when its serial is old).
    package private(set) var frameSerial = 0
    package private(set) var mapSerial = 0
    package private(set) var territorySerial = 0
    package private(set) var materialSerial = 0
    package private(set) var soilSerial = 0
    /// Increases with every change other than a frame.
    package private(set) var revision = 0
    /// Records that were replaced by a newer one before anyone looked, by tag (informational).
    package private(set) var replaced: [String: Int] = [:]

    package init() {}

    package var play: PlayHeader? { header?.play }

    /// A new stream: forget everything of the old one.
    package mutating func reset(header: StreamHeader?) {
        self = WorldStore()
        self.header = header
        revision = 1
    }

    /// Applies a latest-wins record, or an `EVNT` batch (use `apply(events:)` for the fresh events).
    @discardableResult
    package mutating func apply(_ record: StreamRecord) -> Changes {
        var changes: Changes
        switch record {
        case .frma(let r):
            if frame != nil { replaced["FRMA", default: 0] += 1 }
            frame = r
            frameSerial += 1
            return .frame  // a frame is not a revision: the renderer reads it by its serial
        case .idnt(let r): identities = r; changes = .identities
        case .intn(let r): labels = r; changes = .labels
        case .terr(let r): territory = r; territorySerial += 1; changes = .territory
        case .strm(let r): storms = r; changes = .storms
        case .map1(let r): map = r; mapSerial += 1; changes = .map
        case .maph(let r): sphereMap = r; mapSerial += 1; changes = .map
        case .envs(let r): environment = r; changes = .environment
        case .clck(let r): clock = r; changes = .clock
        case .focs(let r): focus = r; changes = .focus
        case .strc(let r): material = r; materialSerial += 1; changes = .material
        case .fsol(let r): soil = r; soilSerial += 1; changes = .soil
        case .fsig(let r): voice = r; changes = .voice
        case .evnt(let batch):
            return apply(events: batch).isEmpty ? [] : .events  // (the revision is bumped by `apply(events:)`)
        case .alrt(let batch):
            return apply(alerts: batch).isEmpty ? [] : .alerts  // (likewise)
        case .rprt(let report):
            return apply(report: report) ? .reports : []
        case .plan(let plan):
            apply(plan: plan)
            return .plans
        }
        if !changes.isEmpty { revision += 1 }
        return changes
    }

    /// Adds the events of a batch; returns the ones that were new (a backlog's are new the first time too).
    @discardableResult
    package mutating func apply(events batch: PlayEventBatch) -> [WorldEvent] {
        let fresh = chronicle.insert(batch.events)
        if !fresh.isEmpty { revision += 1 }
        return fresh
    }

    /// Adds the alerts of a batch; returns what each did to the board (a backlog's alerts are new the first time too).
    @discardableResult
    package mutating func apply(alerts batch: PlayAlertBatch) -> [AlertChange] {
        let changes = alerts.insert(batch.alerts)
        if !changes.isEmpty { revision += 1 }
        return changes
    }

    /// Shelves a report; true if the shelf changed.
    @discardableResult
    package mutating func apply(report: PlayReport) -> Bool {
        let changed = reports.insert(report)
        if changed { revision += 1 }
        return changed
    }

    package mutating func apply(plan: PlayPlan) {
        plans.insert(plan)
        revision += 1
    }

    /// The player acknowledged an alert (the window sent `ack`); false for an unknown id.
    @discardableResult
    package mutating func acknowledge(alert id: UInt64) -> Bool {
        guard alerts.acknowledge(id: id) else { return false }
        revision += 1
        return true
    }

    // MARK: Lookups

    /// The voice of the focused organism: the newest `FSIG` when it is the one the newest `FOCS` follows (a voice of another organism, or
    /// of an organism that has gone, is not shown).
    package var focusVoice: PlayVoice? {
        guard let voice, let focus, focus.isAlive, voice.uid == focus.uid else { return nil }
        return voice
    }

    /// The header says the stream carries the layer (the tag is in `play.records`).
    package func carries(_ tag: String) -> Bool { play?.records.contains(tag) ?? false }

    /// The influence the player has, from the newest `ENVS` (nil in a world with no player channel).
    package var influence: PlayEnvironment.Influence? { environment?.environment.influence }
    /// The food patches the powers can target.
    package var patches: [PlayEnvironment.Patch] { environment?.environment.patches ?? [] }
    /// Whether each power is usable now, by name.
    package var powerStates: [String: PlayEnvironment.PowerState] { environment?.environment.powers ?? [:] }
    /// The powers the world offers (the header's), by name; empty for a world with no player channel.
    package var powerSpecs: [String: PlayHeader.Power] { play?.powers ?? [:] }

    /// The identity row of `uid` in the last `IDNT`, nil when it was born since or is not alive.
    package func identity(ofUID uid: Int64) -> IdentityRow? {
        identities?.rows.first { $0.uid == uid }
    }

    /// The listed index of `uid` in the newest frame, scanning from `hint` (the index it had in the frame before, which
    /// is where it usually still is); nil when it is not in the frame (dead).
    package func listedIndex(ofUID uid: Int64, hint: Int = 0) -> Int? {
        guard let frame else { return nil }
        let uids = frame.uid
        let n = uids.count
        guard n > 0 else { return nil }
        let start = min(max(hint, 0), n - 1)
        for k in 0..<n {
            let i = (start + k) % n
            if uids[i] == uid { return i }
        }
        return nil
    }
}
