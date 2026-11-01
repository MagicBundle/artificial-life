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
        return LibraryState.resolve(LibraryScanner().scan(runsDirectory: runs)).catalog
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
        #expect(model.obstacle(to: model.catalog.worlds[0]) == "saving…")
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
}
