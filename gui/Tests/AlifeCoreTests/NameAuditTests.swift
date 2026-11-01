import Foundation
import Testing

@testable import AlifeCore

/// One word per action (docs/APP.md §2.5), for the pages of the alife window that are new: every label that triggers an action is in
/// the table with the action it triggers; no label triggers two actions and no action has two labels; and none of the words §2.5 retires
/// is in a string the pages show. (The Lab page keeps today's words until PR-D and is not scanned; PR-D extends this to the set-up page.)
@Suite struct NameAuditTests {
    /// (the label on the button or menu item, the action it triggers). The pages' sources are checked to contain each label.
    static let actions: [(label: String, action: String)] = [
        ("Play", "play-world"),
        ("Make planet", "make-planet"),
        ("Make Planet…", "make-planet"),  // the File menu's item: same action, the menu's own capitalisation
        ("Close World", "close-world"),
        ("Watch", "show-watch"),
        ("Go to Worlds", "show-worlds"),
        ("Back to Worlds", "show-worlds"),
        ("Show Activity", "show-activity"),
        ("Open the Lab", "show-lab"),
        ("Overview", "camera-overview"),
        ("Set up engine", "set-up-engine"),
        ("Check again", "check-engine"),
        ("Stop", "stop-run"),
        ("Force stop", "force-stop"),
        ("Cancel", "cancel"),
        ("Dismiss", "dismiss"),
    ]

    /// The page sources that are new in PR-B (the Lab page is `LabPage.swift`).
    static let sources = ["WorldsPage.swift", "ActivityPage.swift", "WatchPage.swift", "GeneralSettingsView.swift", "AppShellView.swift",
                          "AlifeApp.swift"]
    static let coreSources = ["WorldsGuide.swift", "ActivityGuide.swift", "WatchCue.swift", "ShellPage.swift"]

    /// The words §2.5 retires or reserves, and the engine's own names, which a page of the new window does not show to the user.
    static let banned = ["Resume", "Checkpoint", "Fork", "Home", "spin-up", "Spin-up", "director.enabled", "Storm Worlds", "Source",
                         "another window", "branch", "Repository", "repository"]

    private func read(_ file: String, core: Bool) throws -> String {
        let dir = core ? "AlifeCore" : "AlifeControl"
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/\(dir)/\(file)")
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// The text of the string literals of a source (single-line ones), without interpolations' contents.
    private func literals(_ source: String) -> [String] {
        var out: [String] = []
        for line in source.split(separator: "\n") {
            let text = line.trimmingCharacters(in: .whitespaces)
            if text.hasPrefix("//") || text.hasPrefix("///") { continue }
            var rest = Substring(text)
            while let open = rest.firstIndex(of: "\"") {
                let after = rest.index(after: open)
                guard let close = rest[after...].firstIndex(of: "\"") else { break }
                out.append(String(rest[after..<close]))
                rest = rest[rest.index(after: close)...]
            }
        }
        return out
    }

    @Test func noLabelTriggersTwoActionsAndNoActionHasTwoLabelsOnOneSurface() {
        var byLabel: [String: Set<String>] = [:]
        for (label, action) in Self.actions { byLabel[label, default: []].insert(action) }
        for (label, actions) in byLabel { #expect(actions.count == 1, "\(label) triggers \(actions)") }
        // two different labels for one action are allowed only where a menu repeats a button (the table says which)
        let twins = Dictionary(grouping: Self.actions, by: \.action).filter { $0.value.count > 1 }.mapValues { $0.map(\.label).sorted() }
        #expect(twins == ["make-planet": ["Make Planet…", "Make planet"], "show-worlds": ["Back to Worlds", "Go to Worlds"]])
    }

    @Test func everyLabelOfTheTableIsInThePagesAndTheRetiredWordsAreNot() throws {
        var literalsOfPages: [String] = []
        for file in Self.sources { literalsOfPages += literals(try read(file, core: false)) }
        for file in Self.coreSources { literalsOfPages += literals(try read(file, core: true)) }
        let all = Set(literalsOfPages)
        for (label, _) in Self.actions { #expect(all.contains(label), "'\(label)' is not a string of the new pages") }
        for text in literalsOfPages {
            for word in Self.banned where text.contains(word) {
                // identifiers and the engine's own handles are not words the user reads
                let handle = text.hasPrefix("world.") || text.hasPrefix("shell.") || text.contains(".swift")
                #expect(handle, "'\(word)' in the page string \"\(text)\"")
            }
        }
    }

    @Test func theLabelsOfTheMenusAndTheSidebarAreThePageTitles() {
        #expect(ShellPage.allCases.map(\.title) == ["Worlds", "Watch", "Activity", "Lab"])
    }
}
