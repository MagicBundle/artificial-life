import Foundation

// The chronicle tab and the toasts, as pure reducers over the events the world sent (DESIGN §33.4.8). The texts are the
// server's (a closed template table, §33.2.3); the client groups, filters and times them and never writes one.

/// Which families of event the chronicle tab shows.
package struct ChronicleFilter: OptionSet, Sendable, Hashable {
    package let rawValue: Int
    package init(rawValue: Int) { self.rawValue = rawValue }

    package static let storms = ChronicleFilter(rawValue: 1 << 0)
    package static let lineages = ChronicleFilter(rawValue: 1 << 1)
    package static let houses = ChronicleFilter(rawValue: 1 << 2)
    package static let director = ChronicleFilter(rawValue: 1 << 3)
    /// The play loop's own events (checkpoints, a renumbering, extinction) and the focus's.
    package static let system = ChronicleFilter(rawValue: 1 << 4)

    package static let all: ChronicleFilter = [.storms, .lineages, .houses, .director, .system]

    /// The filters in the order the tab lists them, with their labels.
    package static let choices: [(filter: ChronicleFilter, label: String)] = [
        (.storms, "Storms"), (.lineages, "Lineages"), (.houses, "Houses"), (.director, "Director"), (.system, "System"),
    ]

    /// The filter an event belongs to by its family. An unknown family is `system`, so that it is never hidden by accident.
    package static func of(family: String) -> ChronicleFilter {
        switch family {
        case "storm": return .storms
        case "kin": return .lineages
        case "house": return .houses
        case "director": return .director
        default: return .system
        }
    }
}

/// Events within one minute of game time, newest first.
package struct ChronicleGroup: Equatable, Identifiable, Sendable {
    package var minute: Int
    package var steps: Range<Int>
    package var events: [WorldEvent]

    package var id: Int { minute }
    package var title: String { "Minute \(minute)" }
    package var subtitle: String { "steps \(steps.lowerBound) to \(steps.upperBound - 1)" }
}

package enum ChronicleReducer {
    /// The steps in a minute of game time at the base rate, `60·B`: 864 at 14.4. Game time does not depend on the rate the
    /// player has set, so a minute of the chronicle is the same stretch of the world at 0.25x and at 2x.
    package static func stepsPerMinute(baseRate: Double) -> Int {
        max(1, Int((60 * baseRate).rounded()))
    }

    /// The minutes of game time of `step`.
    package static func minute(ofStep step: Int, baseRate: Double) -> Int {
        step / stepsPerMinute(baseRate: baseRate)
    }

    /// A kin event of the first design: one for every split and merge of a group (kinds 1 to 4). A world of G3b tells threads
    /// instead (kinds 18 and 19, DESIGN §34.1), so these are a developer's view of a log that still holds some.
    package static func isRawKin(_ event: WorldEvent) -> Bool {
        event.fam == "kin" && (EventKind.kinBorn...EventKind.kinDissolved).contains(event.kind)
    }

    /// The chronicle newest first, grouped by minute, with only the families of `filter`. Empty groups are not listed. With
    /// `hidingRawKin` the raw kin events are left out (a G3b world; the developer's filter shows them).
    package static func groups(_ chronicle: Chronicle, filter: ChronicleFilter = .all, baseRate: Double,
                               hidingRawKin: Bool = false) -> [ChronicleGroup] {
        let span = stepsPerMinute(baseRate: baseRate)
        var out: [ChronicleGroup] = []
        for event in chronicle.newestFirst
        where filter.contains(ChronicleFilter.of(family: event.fam)) && !(hidingRawKin && isRawKin(event)) {
            let minute = event.step / span
            if let last = out.last, last.minute == minute {
                out[out.count - 1].events.append(event)
            } else {
                out.append(ChronicleGroup(minute: minute, steps: (minute * span)..<((minute + 1) * span), events: [event]))
            }
        }
        return out
    }

    /// The events a delivery raises a toast for: importance 2, and not the backlog loaded at the start (the history of a
    /// re-opened world is not news).
    package static func toasts(from fresh: [WorldEvent], backlog: Bool) -> [WorldEvent] {
        backlog ? [] : fresh.filter { $0.imp >= 2 }
    }
}

/// The toasts on screen: each lasts `duration` seconds from the moment it is raised, at most `capacity` are shown (a burst
/// keeps the newest), and a toast is shown once however often it is pushed. The clock is the caller's. A toast comes from an
/// alert of the stack surface at importance 2 or more (DESIGN §34.7); an event toasts only in a world with no alert engine.
package struct ToastQueue: Equatable, Sendable {
    package struct Toast: Equatable, Identifiable, Sendable {
        /// The id of the event or the alert (the two never clash: §34.6).
        package var id: UInt64
        package var text: String
        /// What a click does: fly to where it happened and select what it is about.
        package var action: ChronicleAction?
        package var expires: Double
    }

    package static let duration = 3.0
    package static let capacity = 3

    /// Oldest first.
    package private(set) var toasts: [Toast] = []
    private var seen: Set<UInt64> = []

    package init() {}

    /// Raises toasts for `events` at time `now`.
    package mutating func push(_ events: [WorldEvent], at now: Double) {
        add(events.map { ($0.id, $0.text, ChronicleAction.of($0)) }, at: now)
    }

    /// Raises toasts for `alerts` at time `now`.
    package mutating func push(alerts: [PlayAlert], at now: Double) {
        add(alerts.map { ($0.id, $0.title.isEmpty ? $0.text : $0.title + ": " + $0.text, ChronicleAction.of($0)) }, at: now)
    }

    private mutating func add(_ items: [(id: UInt64, text: String, action: ChronicleAction?)], at now: Double) {
        for item in items where !seen.contains(item.id) {
            seen.insert(item.id)
            toasts.append(Toast(id: item.id, text: item.text, action: item.action, expires: now + Self.duration))
        }
        if toasts.count > Self.capacity { toasts.removeFirst(toasts.count - Self.capacity) }
        if seen.count > 4 * Chronicle.capacity { seen = Set(toasts.map(\.id)) }  // (an id this old does not come back)
    }

    /// Drops the toasts that ended by `now`; true if any did.
    @discardableResult
    package mutating func expire(at now: Double) -> Bool {
        let before = toasts.count
        toasts.removeAll { $0.expires <= now }
        return toasts.count != before
    }

    /// The time the next toast ends; nil with none on screen.
    package var nextExpiry: Double? { toasts.map(\.expires).min() }

    package mutating func dismiss(id: UInt64) {
        toasts.removeAll { $0.id == id }
    }

    package mutating func clear() {
        toasts = []
    }
}

/// The one line at the bottom of the map (DESIGN §34.7): the newest thing the world told, for `duration` seconds. It carries the
/// log-surface alerts, the told events of the chronicle (a power taking effect included) and the newest line of a live report's
/// ticker; no filler, and nothing when nothing happened. A click flies to the place, as a chronicle row does.
package struct Ticker: Equatable, Sendable {
    package struct Line: Equatable, Identifiable, Sendable {
        package var id: UInt64
        package var text: String
        package var action: ChronicleAction?
        package var expires: Double
    }

    package static let duration = 17.0

    package private(set) var line: Line?
    private var lastID: UInt64 = 0

    package init() {}

    /// Shows `text` for `duration` seconds from `now`, replacing the line on screen. A repeated `id` changes nothing (an event
    /// the world sends again, a backlog); an `id` of 0 is always new (a report's line has none).
    package mutating func show(id: UInt64, text: String, action: ChronicleAction? = nil, at now: Double) {
        guard !text.isEmpty else { return }
        if id != 0 {
            guard id != lastID else { return }
            lastID = id
        }
        line = Line(id: id, text: text, action: action, expires: now + Self.duration)
    }

    /// The newest of a delivery: its last event that is worth a line.
    package mutating func show(events: [WorldEvent], at now: Double) {
        guard let event = events.last(where: Self.isTold) else { return }
        show(id: event.id, text: event.text, action: ChronicleAction.of(event), at: now)
    }

    package mutating func show(alerts: [PlayAlert], at now: Double) {
        guard let alert = alerts.last(where: { $0.surface == .log && $0.state != .cleared }) else { return }
        show(id: alert.id, text: alert.text, action: ChronicleAction.of(alert), at: now)
    }

    /// An event the ticker tells: importance 1 or more, in a family of the world (not a checkpoint or a hand-off of the focus).
    package static func isTold(_ event: WorldEvent) -> Bool {
        event.imp >= 1 && event.fam != "system" && event.fam != "focus"
    }

    @discardableResult
    package mutating func expire(at now: Double) -> Bool {
        guard let line, line.expires <= now else { return false }
        self.line = nil
        return true
    }

    package var nextExpiry: Double? { line?.expires }

    package mutating func clear() {
        line = nil
        lastID = 0
    }
}
