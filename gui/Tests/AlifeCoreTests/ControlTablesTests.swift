import Foundation
import Testing

@testable import AlifeCore

/// The two key tables of Watch (docs/APP.md §3.4): they are never merged, every key the engine of the keys maps is in the table that is shown,
/// no key is shown that does nothing, and the menu names its keys as text.
@Suite struct ControlTablesTests {
    @Test func everyKeyOfTheWorldsTableIsInThePlayedTableAndNothingElse() {
        #expect(ControlTables.keys(.played) == Set(WorldKeys.characters))
    }

    @Test func everyKeyOfTheObserversTableIsInTheResearchTableAndNothingElse() {
        #expect(ControlTables.keys(.research) == Set(ViewerKeys.characters))
        // the list is pinned to the `switch`: no other printable character does anything
        let printable = (32...126).compactMap { UnicodeScalar($0) }.map { String($0) }.filter { $0 == $0.lowercased() }  // (capitals are the same keys)
        let mapped = printable.filter { ViewerKeys.action(for: $0) != nil }
        #expect(Set(mapped) == Set(ViewerKeys.characters.filter { $0 != "\u{1B}" }))
        let mappedWorld = printable.filter { WorldKeys.action(for: $0) != nil }
        #expect(Set(mappedWorld).isSubset(of: Set(WorldKeys.characters)))
    }

    @Test func theKeyOfAnEntryDoesWhatItsMenuItemDoes() {
        for entry in ControlTables.played {
            for key in entry.keys {
                let action = WorldKeys.action(for: key)
                #expect(action != nil, "played '\(entry.title)': '\(key)' does nothing")
                if case .world(let expected)? = entry.action { #expect(action == expected, "played '\(entry.title)': '\(key)'") }
            }
        }
        for entry in ControlTables.research {
            for key in entry.keys {
                let action = ViewerKeys.action(for: key)
                #expect(action != nil, "research '\(entry.title)': '\(key)' does nothing")
                if case .observer(let expected)? = entry.action { #expect(action == expected, "research '\(entry.title)': '\(key)'") }
            }
        }
    }

    @Test func aMenuEntryHasOneActionOfItsOwnFormAndAnyKeyOfItIsItsHint() {
        for form in [WatchForm.played, .research] {
            for entry in ControlTables.menu(form) {
                switch (form, entry.action) {
                case (.played, .world?), (.research, .observer?): break
                default: Issue.record("\(form) '\(entry.title)' has the other form's action or none")
                }
                #expect(entry.menuTitle == "\(entry.title)  (\(entry.hint))")
                #expect(!entry.keys.isEmpty && !entry.hint.isEmpty)
            }
        }
    }

    @Test func eachFormsMenuLeavesOutNothingThatHasASingleAction() {
        // the rows without a menu item are the ones that stay in the dock's View menu or in the help only
        let played = Set(ControlTables.menu(.played).map(\.title))
        #expect(played == ["Pause or Run", "Step", "Slower", "Faster", "Continue", "Overview", "Follow the selection", "North up", "Powers",
                           "Next alert", "Acknowledge the alert", "Next colour"])
        let research = Set(ControlTables.menu(.research).map(\.title))
        #expect(research == ["Pause or Run", "Step", "Slower", "Faster", "Next colour", "Field slice", "Snapshot", "Orbit or Fly", "Reset view"])
    }

    /// The hint is what a person reads in the menu and in Help; it must name keys the row maps, and name the row's first key.
    @Test func theHintOfARowNamesOnlyKeysItMapsAndItsFirstKey() {
        func keys(ofToken token: String) -> [String] {
            switch token {
            case "Space": return [" "]
            case "Esc": return ["\u{1B}"]
            case "Return": return ["\r", "\u{3}"]
            default: return [token.lowercased()]
            }
        }
        for form in [WatchForm.played, .research] {
            for entry in ControlTables.entries(form) {
                let tokens = entry.hint.split(whereSeparator: { $0 == "," || $0 == " " }).map(String.init).filter { $0 != "or" }
                #expect(!tokens.isEmpty, "\(form) '\(entry.title)'")
                for token in tokens {
                    #expect(keys(ofToken: token).contains { entry.keys.contains($0) }, "\(form) '\(entry.title)': the hint names '\(token)', which the row does not map (\(entry.keys))")
                }
                // every key of the row is named, but the shifted twins (the same key with Shift held: "=" for "+", "_" for "-"), the second field-slice
                // key ("4"), the second snapshot key ("p") and the other Return: those are aliases the hint does not repeat
                let aliases: Set<String> = ["=", "_", "4", "p", "\u{3}"]
                for key in entry.keys where !aliases.contains(key) {
                    #expect(tokens.contains { keys(ofToken: $0).contains(key) }, "\(form) '\(entry.title)': the hint \"\(entry.hint)\" leaves out '\(key)'")
                }
                let first = entry.keys[0]
                #expect(tokens.contains { keys(ofToken: $0).contains(first) }, "\(form) '\(entry.title)': the hint \"\(entry.hint)\" leaves out the row's first key '\(first)'")
            }
        }
    }

    @Test func theTwoTablesConflictOnFSOAndRSoTheyAreNeverMerged() {
        // the same letter, two meanings: this is why each form has its own table and the menu follows the form in view
        #expect(WorldKeys.action(for: "f") == .followSelection && ViewerKeys.action(for: "f") == .toggleFields)
        #expect(WorldKeys.action(for: "s") == .viewPreset(.side) && ViewerKeys.action(for: "s") == .snapshot)
        #expect(WorldKeys.action(for: "o") == .viewPreset(.oblique) && ViewerKeys.action(for: "o") == .toggleCameraMode)
        #expect(WorldKeys.action(for: "[") == .slower && ViewerKeys.action(for: "[") == .fieldLayerDown)
        let played = ControlTables.keys(.played), research = ControlTables.keys(.research)
        #expect(!played.intersection(research).isEmpty)
        for key in played.intersection(research) {
            let mine = ControlTables.played.first { $0.keys.contains(key) }
            let theirs = ControlTables.research.first { $0.keys.contains(key) }
            if key == " " || key == "." || key == "c" || key == "1" || key == "2" { continue }  // (the same meaning in both)
            #expect(mine?.title != theirs?.title, "'\(key)' is listed with one meaning in two tables: \(String(describing: mine?.title))")
        }
    }

    @Test func noRowIsListedTwiceInATable() {
        for form in [WatchForm.played, .research] {
            let entries = ControlTables.entries(form)
            #expect(Set(entries.map(\.title)).count == entries.count, "\(form)")
            let keys = entries.flatMap(\.keys)
            #expect(Set(keys).count == keys.count, "\(form): a key in two rows of one table")
        }
    }

    // MARK: The hint on the map

    @Test func theOldAppsHintIsWordForWord() {
        #expect(ControlTables.playedHint(overview: false, planet: false)
            == "Drag moves the map. Two-finger scroll pans; pinch or a mouse wheel zooms (Magic Mouse: View > Scroll zooms); zoom out for the planet, zoom in for the map. H home (the planet), N north up, I minimap, C colour. T O L S: top, oblique, low and side views; Option-drag tilts and turns.")
        #expect(ControlTables.playedHint(overview: false, planet: true)
            == "Drag turns the planet. Two-finger scroll pans; pinch or a mouse wheel zooms (View > Scroll zooms swaps them); zoom out for the whole planet, in for the horizon. H home, N north up, I minimap, C colour, T O L S: top, oblique, low and side views; Option-drag tilts and turns. Click a cell, an organism or a storm to see it.")
    }

    @Test func theNewWindowsHintSaysOverviewAndNeverHome() {
        for planet in [false, true] {
            let hint = ControlTables.playedHint(overview: true, planet: planet)
            #expect(hint.contains("H overview") && !hint.contains("home") && !hint.contains("Home"), "\(hint)")
        }
    }

    @Test func thePointerLinesAreWrittenForBothForms() {
        #expect(ControlTables.pointer(.played).joined().contains("Option-drag tilts"))
        #expect(ControlTables.pointer(.research).joined().contains("Orbit") && ControlTables.pointer(.research).joined().contains("Fly"))
    }
}
