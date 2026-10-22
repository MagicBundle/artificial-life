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

    /// The chronicle newest first, grouped by minute, with only the families of `filter`. Empty groups are not listed.
    package static func groups(_ chronicle: Chronicle, filter: ChronicleFilter = .all, baseRate: Double) -> [ChronicleGroup] {
        let span = stepsPerMinute(baseRate: baseRate)
        var out: [ChronicleGroup] = []
        for event in chronicle.newestFirst where filter.contains(ChronicleFilter.of(family: event.fam)) {
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
/// keeps the newest), and an event is shown once however often it is pushed. The clock is the caller's.
package struct ToastQueue: Equatable, Sendable {
    package struct Toast: Equatable, Identifiable, Sendable {
        package var event: WorldEvent
        package var expires: Double
        package var id: UInt64 { event.id }
    }

    package static let duration = 3.0
    package static let capacity = 3

    /// Oldest first.
    package private(set) var toasts: [Toast] = []
    private var seen: Set<UInt64> = []

    package init() {}

    /// Raises toasts for `events` at time `now`.
    package mutating func push(_ events: [WorldEvent], at now: Double) {
        for event in events where !seen.contains(event.id) {
            seen.insert(event.id)
            toasts.append(Toast(event: event, expires: now + Self.duration))
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
