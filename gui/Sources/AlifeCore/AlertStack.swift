import Foundation

// The alert stack over the map (DESIGN §34.7): the alerts of the stack surface that nobody has acknowledged, most important
// first, at most a few rows, the rest under a count. A pure layout over the `AlertBoard`, so what the stack shows is tested
// without a window. Nothing here acknowledges, hides or pauses by itself: the only way a row leaves is its condition clearing,
// its expiry, or the player's explicit acknowledgement (the check button, `k`).

/// The kin groups the player watches (`watch` / `unwatch`, at most three; with none the world watches the three largest). The
/// lineage alerts read this set on the world's side; the client keeps it to draw the stars and to know what is full.
package struct WatchSet: Equatable, Sendable {
    package static let capacity = 3

    /// In the order they were added.
    package private(set) var ids: [Int] = []

    package init() {}

    package func contains(_ id: Int) -> Bool { ids.contains(id) }

    package var isFull: Bool { ids.count >= Self.capacity }

    /// Watches `id`, or stops watching it when it is watched. The line to send; nil when the set is full and `id` is not in it.
    package mutating func toggle(_ id: Int) -> PlayCommand? {
        if let at = ids.firstIndex(of: id) {
            ids.remove(at: at)
            return .unwatch(Int64(id))
        }
        guard !isFull, id >= 0 else { return nil }
        ids.append(id)
        return .watch(Int64(id))
    }

    /// `watch off`: back to the three largest. Nil when nothing was watched.
    package mutating func clear() -> PlayCommand? {
        guard !ids.isEmpty else { return nil }
        ids = []
        return .watch(nil)
    }
}

/// How an alert is coloured: the importance, in four tones (the view maps them to colours).
package enum AlertTone: Int, Equatable, Sendable {
    case quiet = 0, notice, important, urgent

    package init(importance: Int) { self = AlertTone(rawValue: min(max(importance, 0), 3)) ?? .quiet }
}

/// One row of the stack.
package struct AlertRow: Equatable, Identifiable, Sendable {
    package var id: UInt64
    package var title: String
    package var text: String
    package var tone: AlertTone
    /// The alert stopped the world (or would have, with its class armed): it stays until acknowledged or cleared.
    package var pauses: Bool
    package var step: Int
    /// Where the fly-to arrow goes and what it selects; `nil` for an alert with no place and no entity.
    package var action: ChronicleAction?
    /// The kin group the star watches (the alert's `ref` of kind kin).
    package var kin: Int?
    package var watched: Bool
    /// Whether a measurement backs the class's threshold (shown as a small mark when it does not).
    package var calibrated: Bool
}

package struct AlertStackLayout: Equatable, Sendable {
    package var rows: [AlertRow]
    /// Rows on the stack beyond `rows`.
    package var hidden: Int
}

package enum AlertStack {
    /// Rows shown in a roomy dock, and in a tight one (the card, the inspector and three rows do not fit a 700 point window,
    /// §34.7: the rows give way first).
    package static let roomyRows = 3
    package static let tightRows = 2

    /// The stack at `step`: surface `alert`, not cleared, not acknowledged, not expired; most important first, the newest first
    /// among equals, so that a pausing alert is never the one hidden under the count.
    ///
    /// `excluding` are classes left off: the chapter's alert while the card (or its notice) announces it, so the card, the strip
    /// and the stack do not say "Chapter 3 complete" three times in a dock that is already tight.
    package static func layout(_ board: AlertBoard, step: Int, watch: WatchSet = WatchSet(), maxRows: Int = roomyRows,
                               excluding: Set<String> = []) -> AlertStackLayout {
        let all = board.stack(atStep: step).filter { !excluding.contains($0.alertClass) }.sorted { a, b in
            if a.imp != b.imp { return a.imp > b.imp }
            return a.id > b.id
        }
        let rows = all.prefix(max(maxRows, 0)).map { row(for: $0, watch: watch) }
        return AlertStackLayout(rows: Array(rows), hidden: max(all.count - rows.count, 0))
    }

    package static func row(for alert: PlayAlert, watch: WatchSet = WatchSet()) -> AlertRow {
        let kin: Int? = alert.ref.flatMap { $0.kind == .kin && $0.id >= 0 && $0.id <= Int64(Int.max) ? Int($0.id) : nil }
        return AlertRow(
            id: alert.id, title: alert.title, text: alert.text, tone: AlertTone(importance: alert.imp), pauses: alert.pause,
            step: alert.step, action: ChronicleAction.of(alert), kin: kin,
            watched: alert.watched || (kin.map { watch.contains($0) } ?? false), calibrated: alert.calibrated)
    }
}

extension ChronicleAction {
    /// What a click on an alert does: fly to where it is (its own place, else the entity it names) and select the entity.
    package static func of(_ alert: PlayAlert) -> ChronicleAction? {
        var at: SIMD2<Float>?
        if let x = alert.x, let y = alert.y { at = SIMD2(Float(x), Float(y)) }
        var select = WorldSelection.none
        if let ref = alert.ref, ref.id >= 0, ref.id <= Int64(Int.max) {
            switch ref.kind {
            case .storm: select = .storm(id: Int(ref.id))
            case .kin: select = .kin(id: Int(ref.id))
            case .organism: select = .organism(uid: ref.id)
            case .house, .world, .power: break
            }
        }
        if at == nil, select == .none { return nil }
        return ChronicleAction(at: at, select: select, z: alert.z.map { Float($0) })
    }

    /// What a click on a sentence of a report does: the first of its `refs` that names an entity the report knows.
    package static func of(refs: [String], entities: [String: ReportBody.Entity]) -> ChronicleAction? {
        for key in refs {
            guard let entity = entities[key], entity.id >= 0, entity.id <= Int64(Int.max) else { continue }
            var at: SIMD2<Float>?
            if let x = entity.x, let y = entity.y { at = SIMD2(Float(x), Float(y)) }
            let z = entity.z.map { Float($0) }
            var select = WorldSelection.none
            switch entity.kind {
            case "storm": select = .storm(id: Int(entity.id))
            case "kin": select = .kin(id: Int(entity.id))
            case "organism": select = .organism(uid: entity.id)
            default: break
            }
            if at != nil || select != .none { return ChronicleAction(at: at, select: select, z: z) }
        }
        return nil
    }
}

extension AlertBoard {
    /// The newest alert that stopped the world and is still on the stack: the one the clock's "alert" reason is about.
    package func pausing(atStep step: Int) -> PlayAlert? {
        stack(atStep: step).filter(\.pause).max { $0.id < $1.id }
    }

    /// The alerts raised in `(start, end]` on the stack surface, most important first (the card's "three most important").
    package func raised(in range: Range<Int>, excluding classes: Set<String> = ["chapter"]) -> [PlayAlert] {
        alerts.filter { $0.surface == .alert && $0.state != .cleared && range.contains($0.step) && !classes.contains($0.alertClass) }
            .sorted { $0.imp != $1.imp ? $0.imp > $1.imp : $0.id > $1.id }
    }
}
