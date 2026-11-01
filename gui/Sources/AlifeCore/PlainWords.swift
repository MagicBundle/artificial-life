import Foundation

// The words the engine-side checks use, and the words the page uses instead (docs/APP.md §2.5). Kept apart from `WorldsGuide` so that
// `NameAuditTests` can scan the page's own strings for the retired words without finding them here, where they are what is replaced.
extension WorldsGuide {
    /// The words of the planet form's messages: the form's checks and the launcher's notes speak of runs, checkpoints, spin-ups and sources;
    /// the page says world, save point, warm-up and starting world (docs/APP.md §2.5). Anything not listed gets the same word-for-word swaps.
    package static func plain(_ text: String) -> String {
        if text == "Choose a source run." { return "Choose a world to start from." }
        if text.hasPrefix("'runs/"), text.hasSuffix("' exists.") {
            return "A world named " + text.dropFirst(6).dropLast(9) + " is already here."
        }
        let table: [(String, String)] = [
            ("The checkpoint step is not negative.", "The save point is not negative."),
            ("A source whose weather is on is forked as it is: no spin-up.", "A world whose weather is already on is copied as it is: no weather warm-up."),
            ("Turning the weather on needs a spin-up of at least 1 step.", "Turning the weather on needs a warm-up of at least 1 step."),
            ("The rule tier needs the director compiled into the world (director.enabled).",
             "The rule-based autopilot needs a world that takes powers. Turn that on under Advanced."),
        ]
        if let hit = table.first(where: { $0.0 == text }) { return hit.1 }
        return text.replacingOccurrences(of: "checkpoint", with: "save point").replacingOccurrences(of: "Checkpoint", with: "Save point")
            .replacingOccurrences(of: "spin-up", with: "warm-up").replacingOccurrences(of: "Spin-up", with: "Warm-up")
            .replacingOccurrences(of: "source", with: "starting world").replacingOccurrences(of: "Source", with: "Starting world")
    }
}
