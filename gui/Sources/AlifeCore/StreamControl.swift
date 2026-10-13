import Foundation

/// A control line for the stdin of `alife stream` (DESIGN §27.11, "Control grammar"). The stream applies lines one at
/// a time between steps; a command changes only what is written and when, never the trajectory.
///
/// ```text
/// command = "pause" | "run" | "step" [N] | "every" K | "color" MODE
///         | "fields" ( "on" | "off" ) | "fields" "z" K | "quit"
/// ```
package enum StreamCommand: Equatable, Sendable {
    /// Stops stepping and frames, and cancels pending `step` batches.
    case pause
    /// Resumes stepping (and cancels pending batches).
    case run
    /// Pauses, runs `count` ≥ 1 steps and writes one frame after the last, whatever `every` is. A `step` received
    /// while another batch is queued has a frame of its own.
    case step(count: Int)
    /// The frame period, K ≥ 1 steps.
    case every(Int)
    /// The colour mode of later frames (a name from the header's `colorModes`); writes a frame of the current
    /// state now.
    case color(String)
    /// Includes or omits the field section; writes a frame now.
    case fields(Bool)
    /// The field layer, 0 ≤ K < FZ (the stream refuses a layer outside the header's); writes a frame now if the
    /// field section is on.
    case fieldLayer(Int)
    /// Ends the stream, as does EOF of stdin.
    case quit

    /// A command the grammar cannot express (the stream would drop the line with a note on stderr).
    package struct Invalid: Error, Equatable {
        package let reason: String
    }

    /// The longest line the stream accepts, not counting the line feed.
    package static let maxLineBytes = 4096

    /// The command's line without its line feed.
    package func line() throws -> String {
        switch self {
        case .pause: return "pause"
        case .run: return "run"
        case .step(let count):
            guard count >= 1 else { throw Invalid(reason: "step needs a count of at least 1, not \(count)") }
            return count == 1 ? "step" : "step \(count)"
        case .every(let k):
            guard k >= 1 else { throw Invalid(reason: "every needs a period of at least 1, not \(k)") }
            return "every \(k)"
        case .color(let mode):
            guard mode.range(of: #"^[a-z][a-z0-9_]*$"#, options: .regularExpression) != nil,
                  mode.utf8.count <= Self.maxLineBytes - 6 else {
                throw Invalid(reason: "'\(mode)' is not a colour mode name (lower case letters, digits, '_')")
            }
            return "color \(mode)"
        case .fields(let on): return on ? "fields on" : "fields off"
        case .fieldLayer(let k):
            guard k >= 0 else { throw Invalid(reason: "fields z needs a layer of at least 0, not \(k)") }
            return "fields z \(k)"
        case .quit: return "quit"
        }
    }

    /// The bytes to write to the stream's stdin: the line and its line feed.
    package func encoded() throws -> Data { Data((try line() + "\n").utf8) }
}
