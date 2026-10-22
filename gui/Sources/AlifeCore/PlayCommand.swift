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

/// A control line for the stdin of `alife play` (DESIGN §33.2.6). The world applies lines one at a time between steps;
/// the presentation-only ones (`focus follow idnt labels color every`) change what is written and when, never the
/// trajectory.
///
/// ```text
/// command = "pause" | "run" | "step" [N] | "rate" ( M | "max" ) | "color" MODE | "every" ( K | "auto" )
///         | "focus" ( UID | "off" ) | "follow" ( UID | "off" ) | "idnt" | "labels" ( "on" | "off" ) | "quit"
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
        }
    }

    private static func target(_ uid: Int64?) throws -> String {
        guard let uid else { return "off" }
        guard uid >= 0 else { throw Invalid(reason: "a uid is not negative, not \(uid)") }
        return String(uid)
    }

    /// The bytes to write to the world's stdin: the line and its line feed.
    package func encoded() throws -> Data { Data((try line() + "\n").utf8) }
}
