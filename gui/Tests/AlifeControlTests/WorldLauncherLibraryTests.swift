import AlifeCore
import Foundation
import Testing

@testable import AlifeControl

/// The planet form fed by the library instead of by its own scan (docs/APP.md PR-C): a catalog taken from elsewhere, the form opened from a
/// world's card, and an unfinished build made again under a new name.
@MainActor
@Suite struct WorldLauncherLibraryTests {
    private func catalog(_ runs: URL) throws -> PlayCatalog {
        try LibraryTestFolders.research(runs, "seed", steps: [10_000, 30_000])
        try LibraryTestFolders.research(runs, "other_seed", steps: [5_000])
        try LibraryTestFolders.played(runs, "planet_a", source: "runs/seed:30000")
        try LibraryTestFolders.unfinished(runs, "planet_b", parent: "runs/seed:10000")
        try LibraryTestFolders.unfinished(runs, "planet_c", parent: nil)
        return LibraryState.resolve(LibraryScanner(inline: true).scan(runsDirectory: runs)).catalog
    }

    private func launcher(_ launches: LaunchLog = LaunchLog()) -> (WorldLauncherModel, LaunchLog) {
        (WorldLauncherModel(runsDirectory: { nil }, launch: { launches.requests.append($0) }), launches)
    }

    final class LaunchLog {
        var requests: [PlayRequest] = []
    }

    @Test func aCatalogFromTheLibraryFillsTheFormAndTheSavingSet() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, _) = launcher()
        model.apply(catalog: try catalog(scratch.url), saving: ["planet_a"])
        #expect(Set(model.catalog.sources.map(\.name)) == ["seed", "other_seed", "planet_a"])
        #expect(model.saving == ["planet_a"] && !model.planet.source.isEmpty && !model.planet.name.isEmpty)
        #expect(model.obstacle(to: try #require(model.catalog.worlds.first)) == "saving…")
    }

    @Test func theFormOpenedFromACardStartsAtThatWorldsNewestSavePointWithAFreeName() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, _) = launcher()
        model.apply(catalog: try catalog(scratch.url), saving: [])
        model.planet.name = "planet_a"  // taken
        model.prepareForm(from: "other_seed")
        #expect(model.planet.source == "other_seed" && model.planet.sourceStep == 5_000)
        #expect(!model.catalog.existingNames.contains(model.planet.name) && model.planetErrors.isEmpty)
        model.prepareForm(from: "no_such_world")
        #expect(model.planet.source == "other_seed", "a name that is not a source changes nothing")
    }

    @Test func aBuildMadeAgainFromThisSessionKeepsEverythingItWasMadeWith() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, launches) = launcher()
        model.apply(catalog: try catalog(scratch.url), saving: [])
        model.selectSource("seed")
        model.planet.name = "planet_b"
        model.planet.weatherSpinup = 4_321
        model.planet.houses = .clade
        model.planet.name = "fresh"
        model.startPlanet()
        #expect(launches.requests.count == 1 && model.startedSpecs["fresh"]?.weatherSpinup == 4_321)
        // "fresh" is now an unfinished directory in this folder
        let build = UnfinishedRow(name: "fresh", reason: "x", lastTouched: Date(), source: nil)
        model.selectSource("other_seed")
        model.prepareAgain(build)
        #expect(model.planet.source == "seed" && model.planet.weatherSpinup == 4_321 && model.planet.houses == .clade)
        #expect(model.planet.name == "fresh-2", "a new name: \(model.planet.name)")
    }

    @Test func aBuildFromAnEarlierSessionIsMadeAgainFromItsSourceAndSavePoint() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, launches) = launcher()
        model.apply(catalog: try catalog(scratch.url), saving: [])
        let build = UnfinishedRow(name: "planet_b", reason: "x", lastTouched: Date(), source: SourceRef(name: "seed", step: 10_000))
        model.prepareAgain(build)
        #expect(model.planet.source == "seed" && model.planet.sourceStep == 10_000)
        #expect(model.planet.name != "planet_b" && model.planetErrors.isEmpty)
        #expect(launches.requests.isEmpty, "nothing starts: the form is filled in and the player presses Make planet")
    }

    @Test func theLauncherOfTheOldAppsStillScansForItself() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = scratch.url.appendingPathComponent("runs")
        _ = try catalog(runs)
        let model = WorldLauncherModel(runsDirectory: { runs }, launch: { _ in })
        model.refresh(saving: ["x"])
        #expect(model.catalog.worlds.map(\.name) == ["planet_a"] && model.saving == ["x"])
    }

    @Test func aNameThatDiffersOnlyInCapitalsOrIsALinkToADriveIsTaken() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let runs = scratch.url.appendingPathComponent("runs")
        _ = try catalog(runs)
        try FileManager.default.createSymbolicLink(atPath: runs.appendingPathComponent("archived_a").path, withDestinationPath: "/Volumes/Gone/runs/archived_a")
        let (model, _) = launcher()
        model.apply(catalog: LibraryState.resolve(LibraryScanner(inline: true).scan(runsDirectory: runs)).catalog, saving: [])
        model.selectSource("seed")
        model.planet.name = "archived_a"
        #expect(model.planetMessages == ["A world named archived_a is already here."], "a link to an archive drive holds its name")
        model.planet.name = "PLANET_A"
        #expect(model.planetMessages.count == 1 && model.planetMessages[0].contains("planet_a") && model.planetMessages[0].contains("capitals"))
        model.planet.name = "planet_b2"
        #expect(model.planetErrors.isEmpty)
        // the suggestions never pick a name that a capital letter would make the same folder
        let free = model.freeName(after: "planet_a")
        #expect(free != "planet_a" && free != "PLANET_A", "planet_a is taken")
        #expect(!model.catalog.existingNames.map { $0.lowercased() }.contains(free.lowercased()))
        #expect(model.freeName(after: "Planet_A") != "Planet_A", "and so is the same name with capitals")
    }

    @Test func aSpecIsRememberedUnderTheNameTheDirectoryHas() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, _) = launcher()
        model.apply(catalog: try catalog(scratch.url), saving: [])
        model.selectSource("seed")
        model.planet.name = "  spaced  "
        model.planet.weatherSpinup = 777
        model.startPlanet()
        #expect(model.startedSpecs["spaced"]?.weatherSpinup == 777, "the directory is `spaced`, not `  spaced  `")
        #expect(model.startedSpecs["  spaced  "] == nil)
    }

    @Test func theFormsOwnWordsAreThePagesWords() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (model, _) = launcher()
        model.apply(catalog: try catalog(scratch.url), saving: [])
        model.planet.source = ""
        model.planet.name = ""
        var texts = model.planetMessages
        model.planet.source = "planet_a"
        model.planet.name = "planet_a"
        model.planet.sourceStep = -3
        texts += model.planetMessages
        model.planet.sourceDims = 2
        model.planet.storm = true
        texts += model.planetMessages
        texts += [model.weatherNote ?? "", model.housesMessage ?? ""]
        #expect(texts.count >= 5)
        for text in texts {
            for word in ["source", "Source", "checkpoint", "Checkpoint", "spin-up", "chunks", "runs/"] {
                #expect(!text.contains(word), "'\(word)' in \"\(text)\"")
            }
        }
    }
}
