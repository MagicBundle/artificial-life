import Foundation

/// The multiple of the base rate a world runs at.
package enum PlayRate: Equatable, Hashable, Sendable {
    /// `M` times the base rate, M in [0.05, 16]; 0 pauses.
    case multiplier(Double)
    /// Unthrottled.
    case max

    /// The rungs the window offers (0.25x to 2x and max); the `rate` line accepts any multiplier in `range`.
    package static let rungs: [PlayRate] = [.multiplier(0.25), .multiplier(0.5), .multiplier(1), .multiplier(2), .max]
    package static let range = 0.05...16.0

    /// The label of a rung or a requested rate: "0.25x", "1x", "max".
    package var label: String {
        switch self {
        case .max: return "max"
        case .multiplier(let m): return PlayCommand.decimal(m) + "x"
        }
    }

    /// The rate `CLCK.rate` reports: 0 paused, -1 max, else the multiplier; nil for a value that is not one.
    package init?(clock rate: Float) {
        if rate < 0 {
            self = .max
        } else if rate.isFinite {
            self = .multiplier(Double(rate))
        } else {
            return nil
        }
    }
}

/// What an `autopause` line sets (DESIGN §34.2): every pause at once, the pause at a chapter end, or the pause of one alert
/// class. It changes what the world presents, never the trajectory.
package enum AutopauseSetting: Equatable, Sendable {
    case all(Bool)
    case chapter(Bool)
    case alert(String, Bool)
}

/// How many saves a world keeps (DESIGN §34.10): `all` the last few (3) and the origin, `newest` only the newest and the origin. The
/// world's `saves` line sets it; it changes what stays on disk, never the trajectory.
package enum SaveMode: String, CaseIterable, Equatable, Hashable, Sendable {
    case all, newest
}

/// A `power` line (DESIGN §34.5): `power preview NAME id=N ...` dry-runs a power and answers one `PLAN`, `power NAME id=N
/// ...` queues it and `power cancel NAME id=N` ends a running one. `id` is the client's request number, echoed in every
/// `PLAN` of the exchange. The grammar checks the shape of a value; its range is the power's to clamp or refuse.
package struct PowerCommand: Equatable, Sendable {
    package enum Verb: Equatable, Sendable { case apply, preview, cancel }

    package enum Name: String, CaseIterable, Sendable { case food, bloom, blight, strike }

    /// `steps` and `patch` are whole numbers, the others decimals.
    package enum Key: String, CaseIterable, Sendable {
        case mult, steps, patch, gain, x, y, radius, kill
        var isWhole: Bool { self == .steps || self == .patch }
    }

    package struct Parameter: Equatable, Sendable {
        package var key: Key
        package var value: Double
        package init(_ key: Key, _ value: Double) {
            self.key = key
            self.value = value
        }
    }

    package var verb: Verb
    package var name: Name
    package var id: Int
    /// In the order they are sent.
    package var parameters: [Parameter]

    package init(_ verb: Verb, _ name: Name, id: Int, _ parameters: [Parameter] = []) {
        self.verb = verb
        self.name = name
        self.id = id
        self.parameters = parameters
    }
}

/// A control line for the stdin of `alife play` (DESIGN §33.2.6, §34.6). The world applies lines one at a time between
/// steps; the presentation-only ones (`focus follow idnt labels color every autopause watch unwatch ack`, a `power
/// preview` and a refused `power`) change what is written and when, never the trajectory; `saves` changes what stays on disk. An applied `power` or `power
/// cancel` is the one line that does: the world logs it (`director.jsonl`) so a played world replays.
///
/// ```text
/// command = "pause" | "run" | "step" [N] | "rate" ( M | "max" ) | "color" MODE | "every" ( K | "auto" )
///         | "focus" ( UID | "off" ) | "follow" ( UID | "off" ) | "idnt" | "labels" ( "on" | "off" ) | "quit"
///         | "autopause" ( "on" | "off" | "chapter" ( "on" | "off" ) | "alert" CLASS ( "on" | "off" ) )
///         | "watch" ( KIN_ID | "off" ) | "unwatch" KIN_ID | "ack" ALERT_ID
///         | "power" [ "preview" | "cancel" ] NAME "id=" N { KEY "=" VALUE } | "intent" TEXT
///         | "saves" ( "all" | "newest" )
/// ```
package enum PlayCommand: Equatable, Sendable {
    case pause
    /// Resumes stepping at the rate the world holds (`rate` is the command that changes it).
    case run
    /// Pauses, runs `count` >= 1 steps unthrottled and writes one frame after the last.
    case step(count: Int)
    /// `rate M` sets the rate, and unpauses when M > 0; `rate 0` pauses.
    case rate(PlayRate)
    /// The colour mode of later frames (a name from the header's `colorModes`); writes a frame now.
    case color(String)
    /// The frame period in steps, 1 to 64; nil restores the automatic period.
    case every(Int?)
    /// Reports one organism (`FOCS`); nil stops. An unknown or dead uid answers with a record whose alive flag is 0.
    case focus(Int64?)
    /// `focus` plus the hand-off at death; nil cancels the follow and keeps the focus.
    case follow(Int64?)
    /// Sends `IDNT INTN TERR ENVS MAP1 STRM` now (after a gap, or for a new window).
    case idnt
    /// Starts or stops `INTN` (bandwidth in fast-forward).
    case labels(Bool)
    case quit
    /// Arms or disarms the pauses the world takes by itself (the chapter end, an alert class).
    case autopause(AutopauseSetting)
    /// Adds a kin group (by thread) to the watch set the lineage alerts read; nil clears it.
    case watch(Int64?)
    case unwatch(Int64)
    /// Acknowledges an alert by its id.
    case ack(Int64)
    case power(PowerCommand)
    /// Reserved (G4): the world answers "intent is not available in this build".
    case intent(String)
    /// Keeps the last saves (`all`) or only the newest one (`newest`) from the next save on; nothing is deleted at once (§34.10).
    case saves(SaveMode)

    /// A command the grammar cannot express (the world would drop the line with a note on stderr).
    package struct Invalid: Error, Equatable {
        package let reason: String
    }

    /// The longest line the world accepts, not counting the line feed.
    package static let maxLineBytes = 4096

    /// A decimal without an exponent: "0.25", "1", "2.5".
    package static func decimal(_ value: Double) -> String {
        var text = String(format: "%.4f", value)
        while text.hasSuffix("0") { text.removeLast() }
        if text.hasSuffix(".") { text.removeLast() }
        return text
    }

    /// The command's line without its line feed.
    package func line() throws -> String {
        switch self {
        case .pause: return "pause"
        case .run: return "run"
        case .idnt: return "idnt"
        case .quit: return "quit"
        case .step(let count):
            guard count >= 1 else { throw Invalid(reason: "step needs a count of at least 1, not \(count)") }
            return count == 1 ? "step" : "step \(count)"
        case .rate(.max): return "rate max"
        case .rate(.multiplier(let m)):
            guard m.isFinite, m == 0 || PlayRate.range.contains(m) else {
                throw Invalid(reason: "rate needs 0 or a multiplier in [0.05, 16], not \(m)")
            }
            return "rate " + Self.decimal(m)
        case .color(let mode):
            guard mode.range(of: #"^[a-z][a-z0-9_]*$"#, options: .regularExpression) != nil,
                  mode.utf8.count <= Self.maxLineBytes - 6 else {
                throw Invalid(reason: "'\(mode)' is not a colour mode name (lower case letters, digits, '_')")
            }
            return "color \(mode)"
        case .every(let k):
            guard let k else { return "every auto" }
            guard (1...64).contains(k) else { throw Invalid(reason: "every needs a period in 1...64, not \(k)") }
            return "every \(k)"
        case .focus(let uid): return try "focus " + Self.target(uid)
        case .follow(let uid): return try "follow " + Self.target(uid)
        case .labels(let on): return on ? "labels on" : "labels off"
        case .autopause(let setting): return try Self.line(of: setting)
        case .watch(let id): return try "watch " + (id.map { try Self.identifier($0, "a kin id") } ?? "off")
        case .unwatch(let id): return try "unwatch " + Self.identifier(id, "a kin id")
        case .ack(let id): return try "ack " + Self.identifier(id, "an alert id")
        case .power(let power): return try Self.line(of: power)
        case .saves(let mode): return "saves " + mode.rawValue
        case .intent(let text):
            let trimmed = text.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty, !trimmed.contains(where: { $0.isNewline }) else {
                throw Invalid(reason: "intent needs a text on one line")
            }
            guard trimmed.utf8.count <= Self.maxLineBytes - "intent ".utf8.count else {
                throw Invalid(reason: "intent is longer than a line may be")
            }
            return "intent " + trimmed
        }
    }

    private static func identifier(_ id: Int64, _ what: String) throws -> String {
        guard id >= 0 else { throw Invalid(reason: "\(what) is not negative, not \(id)") }
        return String(id)
    }

    private static func line(of setting: AutopauseSetting) throws -> String {
        func word(_ on: Bool) -> String { on ? "on" : "off" }
        switch setting {
        case .all(let on): return "autopause " + word(on)
        case .chapter(let on): return "autopause chapter " + word(on)
        case .alert(let name, let on):
            guard name.range(of: #"^[a-z][a-z0-9_]*$"#, options: .regularExpression) != nil else {
                throw Invalid(reason: "'\(name)' is not an alert class name (lower case letters, digits, '_')")
            }
            return "autopause alert \(name) " + word(on)
        }
    }

    private static func line(of power: PowerCommand) throws -> String {
        guard power.id >= 0 else { throw Invalid(reason: "a power id is not negative, not \(power.id)") }
        var parts = ["power"]
        switch power.verb {
        case .apply: break
        case .preview: parts.append("preview")
        case .cancel: parts.append("cancel")
        }
        parts += [power.name.rawValue, "id=\(power.id)"]
        if power.verb == .cancel, !power.parameters.isEmpty {
            throw Invalid(reason: "power cancel takes only an id")
        }
        var seen: Set<PowerCommand.Key> = []
        for p in power.parameters {
            guard seen.insert(p.key).inserted else { throw Invalid(reason: "power names \(p.key.rawValue) twice") }
            guard p.value.isFinite, p.value >= 0, !p.key.isWhole || p.value == p.value.rounded() else {
                throw Invalid(reason: "\(p.key.rawValue) needs \(p.key.isWhole ? "a whole" : "a") non-negative number, not \(p.value)")
            }
            if p.key.isWhole {
                // `Int64(_:)` traps at 2^63 and above: a value the line cannot hold is refused, not written
                guard let whole = Int64(exactly: p.value) else {
                    throw Invalid(reason: "\(p.key.rawValue) needs a whole number that fits in 64 bits, not \(p.value)")
                }
                parts.append(p.key.rawValue + "=" + String(whole))
            } else {
                parts.append(p.key.rawValue + "=" + decimal(p.value))
            }
        }
        return parts.joined(separator: " ")
    }

    private static func target(_ uid: Int64?) throws -> String {
        guard let uid else { return "off" }
        guard uid >= 0 else { throw Invalid(reason: "a uid is not negative, not \(uid)") }
        return String(uid)
    }

    /// The bytes to write to the world's stdin: the line and its line feed.
    package func encoded() throws -> Data { Data((try line() + "\n").utf8) }
}
