import Foundation
import Testing

@testable import AlifeCore

/// One word per action (docs/APP.md §2.5), for the pages of the alife window that are new: every label that triggers an action is in
/// the table with the action it triggers; no label triggers two actions and no action has two labels; and none of the words §2.5 retires
/// is in a string the pages show. PR-D extends it to the set-up page, the pipeline's bar and the cards' new buttons, and to the Lab's and
/// the launcher's words, which no page of the window has any more.
@Suite struct NameAuditTests {
    /// (the label on the button or menu item, the action it triggers). The pages' sources are checked to contain each label.
    static let actions: [(label: String, action: String)] = [
        ("Play", "play-world"),
        ("Make planet", "make-planet"),  // the sheet's button: starts the planet
        ("Make planet…", "open-planet-form"),  // a card's button: opens the form (the ellipsis says a form follows)
        ("Make Planet…", "open-planet-form"),  // the File menu's item: same action, the menu's own capitalisation
        ("Make again with a new name", "make-again"),
        ("Reveal in Finder", "reveal"),
        ("Clear the search", "clear-search"),
        ("Show all worlds", "show-all-worlds"),
        ("Close World", "close-world"),
        ("Watch", "show-watch"),
        ("Go to Worlds", "show-worlds"),
        ("Back to Worlds", "show-worlds"),
        ("Show Activity", "show-activity"),
        ("New world", "new-world"),
        ("New World…", "new-world"),  // the File menu's item
        ("Create a small planet", "create-small-planet"),
        ("Choose something else…", "new-world"),
        ("Create", "start-create"),
        ("Branch", "start-branch"),
        ("Branch…", "open-branch-form"),
        ("Keep growing", "start-keep-growing"),
        ("Keep growing…", "open-keep-growing-form"),
        ("Make planet from here", "make-planet"),  // the bar's button for a world that was stopped: the same action as Make planet
        ("Done", "pipeline-done"),
        ("Show log", "show-activity"),  // the log is on the Activity page
        ("Try again with a new name", "retry-new-name"),
        ("Show progress", "show-setup"),
        ("Open in window", "open-setup-window"),
        ("Reset all", "reset-remembered"),
        ("Copy command", "copy-command"),
        ("Watch in 3D viewer", "watch-3d-viewer"),
        ("Open Classic 2D Viewer", "open-classic-viewer"),
        ("Check world", "check-world"),
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
                          "AlifeApp.swift", "WorldCards.swift", "WorldLibrary.swift", "SetupPage.swift", "SetupModel.swift", "PipelineViews.swift",
                          "PipelineCoordinator.swift", "RunNumbersView.swift"]
    static let coreSources = ["WorldsGuide.swift", "ActivityGuide.swift", "WatchCue.swift", "ShellPage.swift", "WorldLibraryState.swift",
                              "LibraryScanner.swift", "Recipe.swift", "PipelinePlan.swift", "SetupPlan.swift", "SetupSettings.swift",
                              "DirectoryGuard.swift"]

    /// The words §2.5 retires or reserves, and the engine's own names, which a page of the new window does not show to the user.
    static let banned = ["Resume", "Checkpoint", "Fork", "Home", "spin-up", "Spin-up", "director.enabled", "Storm Worlds", "Source",
                         "another window", "Repository", "repository", "its seed", "the Lab", "Lab's", "Open the Lab", "Go to the Lab", "Mode:", "Director"]

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
        #expect(twins == [
            "open-planet-form": ["Make Planet…", "Make planet…"], "show-worlds": ["Back to Worlds", "Go to Worlds"],
            "new-world": ["Choose something else…", "New World…", "New world"],
            "make-planet": ["Make planet", "Make planet from here"],
            "show-activity": ["Show Activity", "Show log"],
        ])
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

    /// "seed" is the random seed field and nothing else: the world a planet is grown from is "the grown world" (one word, one meaning).
    @Test func seedMeansOnlyTheRandomSeedAndBranchOnlyWhatBranchDoes() throws {
        var literalsOfPages: [String] = []
        for file in Self.sources { literalsOfPages += literals(try read(file, core: false)) }
        for file in Self.coreSources { literalsOfPages += literals(try read(file, core: true)) }
        let seedUses = ["Seed \\(", "New seed", "Seed must", "The seed must", "The same seed", "Seed: the preset's", "MadeWorld("]
        for text in literalsOfPages where text.localizedCaseInsensitiveContains("seed") {
            #expect(text == "Seed" || text == "seed" || seedUses.contains { text.contains($0) }, "'seed' means the random seed only: \"\(text)\"")
        }
        // "branch" is the action of copying a world from its newest save point and changing it; these are the places it is said
        let branchUses = ["Branch", "-branch", "A branch keeps", "you branch from", "branched from", "branch one"]
        for text in literalsOfPages where text.localizedCaseInsensitiveContains("branch") {
            #expect(branchUses.contains { text.contains($0) }, "'branch' is used for something else: \"\(text)\"")
        }
    }

    @Test func theLabelsOfTheMenusAndTheSidebarAreThePageTitles() {
        #expect(ShellPage.allCases.map(\.title) == ["Worlds", "Watch", "Activity", "New world"])
    }

    // MARK: The library's words (PR-C)

    @Test func theStatesOfAWorldAreTheNamesOfSection2_5AndNoStateIsAnAction() {
        #expect(WorldState.allCases.map(\.title) == ["Saved", "Live", "Paused", "Starting", "Growing", "Saving", "In use elsewhere", "Unfinished"])
        let labels = Set(Self.actions.map(\.label))
        for state in WorldState.allCases { #expect(!labels.contains(state.title), "'\(state.title)' is a state and a button") }
        #expect(WorldKind.allCases.map(\.title) == ["Played", "Research"])
        #expect(Set(WorldKind.allCases.map(\.title)).isDisjoint(with: labels.union(WorldState.allCases.map(\.title))))
        #expect([WorldShape.flat, .box, .planet].map(\.title) == ["Flat", "3D box", "Planet"])
        #expect(LibraryFilter.allCases.map(\.title) == ["All", "Played", "Research", "Unfinished"])
    }

    @Test func noWordOfAStateOrAReasonIsOneOfTheRetiredOnes() {
        var texts = WorldState.allCases.flatMap { [$0.title, $0.meaning] }
        let cadence = PlayWorldFile.Cadence(metricsEvery: 1, heavyMetricsEvery: 1, checkpointEvery: 1, keepCheckpoints: 1)
        let file = PlayWorldFile(playVersion: 1, baseRate: 14.4, cadence: cadence, tick: 1, weather: nil, planet: nil)
        let world = PlayWorld(name: "w", file: file, newestStep: 1, status: nil, modified: Date(), locked: true)
        let facts = WorldsGuide.Facts(folderExists: true, engineReady: true, catalog: PlayCatalog(), saving: ["w"])
        texts.append(WorldsGuide.reason(forPlaying: world, in: facts) ?? "")
        for status in [nil, "interrupted", "violation", "extinct", "running"] {
            for mark in [nil, "planet"] {
                for steps in [[], [1]] {
                    var r = LibraryRecord(
                        name: "u", play: nil, steps: steps, status: status, parent: nil, dims: 3, geometry: "sphere", worldSize: nil, planetRadius: nil,
                        planetShape: nil, weatherOn: false, hasEvents: false, directoryModified: Date(), lastSaved: Date(), hasContent: true,
                        looksUnfinished: true, buildMark: mark)
                    r.status = status
                    texts.append(LibraryState.unfinishedReason(r))
                    texts.append(LibraryState.unfinishedReason(r, sessionNote: "You cancelled it"))
                }
            }
        }
        for status in ["extinct", "violation", "running"] { texts.append(LibraryState.endNote(status) ?? "") }
        for filter in LibraryFilter.allCases { texts.append(WorldsGuide.emptyList(filter: filter, query: "", counts: LibraryCounts())) }
        for text in texts {
            for word in Self.banned { #expect(!text.contains(word), "'\(word)' in \"\(text)\"") }
            #expect(!text.localizedCaseInsensitiveContains("checkpoint"), "\(text)")
        }
    }

    @Test func theMessagesOfThePlanetSheetAreInThePagesWords() {
        // the sheet shows what `PlanetSpec.validate` and `weatherObstacle` say, through `WorldsGuide.plain`
        var spec = PlanetSpec(source: "", name: "")
        var texts: [String] = []
        func take() { texts += spec.validate(existing: ["taken"]).map(WorldsGuide.plain) }
        take()
        spec.source = "x"; spec.name = "bad name"; take()
        spec.name = "taken"; spec.sourceStep = -1; take()
        spec.sourceStep = nil; spec.storm = true; spec.weatherSpinup = 0; spec.sourceDims = 2; take()
        spec.sourceDims = 3; spec.storm = false; spec.weatherSpinup = 4; take()
        spec.director = .rules; spec.directorEnabled = false; spec.seed = -1; take()
        texts.append(WorldsGuide.plain(PlanetSpec.weatherObstacle(dims: 2) ?? ""))
        texts.append(WorldsGuide.plain(PlanetSpec.weatherObstacle(dims: 3, geometry: "sphere", shape: nil) ?? ""))
        #expect(texts.count >= 10 && !texts.contains(""))
        for text in texts {
            for word in Self.banned + ["source", "checkpoint", "spin-up", "runs/", "director"] { #expect(!text.contains(word), "'\(word)' in \"\(text)\"") }
        }
    }
}
