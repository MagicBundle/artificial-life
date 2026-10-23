import Foundation

/// Numbers as the world's own sentences write them: digits grouped by thousands with a comma, whatever the machine's locale.
/// A locale separator is a full stop in much of Europe, and "6.216 born" reads as a decimal (G3a's Mac day, DESIGN §34.7).
package enum Numbers {
    /// 6216 becomes "6,216"; -1234 "-1,234"; below a thousand the digits alone.
    package static func grouped(_ value: Int) -> String {
        let digits = String(value.magnitude)
        var out = ""
        for (offset, character) in digits.reversed().enumerated() {
            if offset > 0, offset % 3 == 0 { out.append(",") }
            out.append(character)
        }
        return (value < 0 ? "-" : "") + String(out.reversed())
    }

    package static func grouped(_ value: Int64) -> String { grouped(Int(clamping: value)) }

    /// "step 468,000": a step as the dock and the chronicle write it. `Text("step \(n)")` is a localized string key and groups
    /// with the machine's separator ("468.000"); the views show this one verbatim.
    package static func stepLabel(_ step: Int) -> String { "step " + grouped(step) }
    package static func stepLabel(_ step: UInt64) -> String { stepLabel(Int(clamping: step)) }
}
