import Foundation

// What the World window knows of the world it is connected to, reduced from the decoded records (DESIGN §33.4.2). A value
// type with no UI and no clock, so that the reducers are tested alone. A latest-wins record (`FRMA IDNT INTN TERR STRM
// MAP1 ENVS CLCK FOCS`) replaces the one before it, and is complete by itself (§33.1.1); `EVNT` is queued: every event is
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
    }

    package private(set) var header: StreamHeader?
    package private(set) var frame: PlayFrame?
    package private(set) var identities: PlayIdentities?
    package private(set) var labels: PlayLabels?
    package private(set) var territory: PlayTerritory?
    package private(set) var storms: PlayStormRecord?
    package private(set) var map: PlayWeatherMap?
    package private(set) var environment: PlayEnvironmentRecord?
    package private(set) var clock: PlayClock?
    package private(set) var focus: PlayFocus?
    package private(set) var chronicle = Chronicle()
    /// Increase with every record of the kind that arrives (the renderer refreshes a buffer set when its serial is old).
    package private(set) var frameSerial = 0
    package private(set) var mapSerial = 0
    package private(set) var territorySerial = 0
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
        case .envs(let r): environment = r; changes = .environment
        case .clck(let r): clock = r; changes = .clock
        case .focs(let r): focus = r; changes = .focus
        case .evnt(let batch):
            return apply(events: batch).isEmpty ? [] : .events  // (the revision is bumped by `apply(events:)`)
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

    // MARK: Lookups

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
