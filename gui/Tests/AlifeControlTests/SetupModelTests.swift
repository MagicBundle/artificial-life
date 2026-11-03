import AlifeCore
import Foundation
import Testing

@testable import AlifeControl

/// The set-up page's model without a process: what the button says, why it is grey, the starting points, the goals (docs/APP.md §3.2, §3.5).
@MainActor
@Suite("The set-up page's model", .serialized)
struct SetupModelTests {
    private func rig() throws -> PipelineRig { try PipelineRig(mode: "ok:ok") }

    @Test func theFirstFormIsTheSmallPlanetToPlayWithAFreeNameAndNothingToExplain() async throws {
        let rig = try rig()
        defer { rig.finish() }
        await rig.scan()
        let m = rig.setup
        m.openCreate()
        #expect(m.startingPoint == .recipe("small-planet") && m.goal == .play && m.name == "planet-1" && m.growSteps == "30000")
        #expect(m.reasons.isEmpty, "\(m.reasons)")
        #expect(m.changes.isEmpty && m.otherChanges.isEmpty && m.rememberedShown.isEmpty)
        #expect(m.intent.buttonLabel == "Create")
        #expect(m.growLine == "about 1 min 50 s")
        #expect(m.plan.madeSentence == "This makes 2 worlds: planet-1 (you play it) and planet-1-grow (the grown world it starts from).")
    }

    @Test func aFlatWorldIsForResearchOnlyAndSaysWhy() async throws {
        let rig = try rig()
        defer { rig.finish() }
        await rig.scan()
        let m = rig.setup
        m.openCreate(.recipe(Recipes.flatWorld.id))
        #expect(m.goalsAllowed == [.research] && m.goal == .research)
        #expect(m.goalReason == "A flat world has no weather, so it is for research only.")
        #expect(m.name == "flat-1" && m.growSteps == "20000" && m.plan.stages.count == 1)
        m.name = "mine"
        m.choose(.recipe(Recipes.biggerPlanet.id))
        #expect(m.name == "mine", "a typed name does not follow the starting point")
        m.choose(.recipe(Recipes.flatWorld.id))
        #expect(m.reasons.isEmpty, "\(m.reasons)")
        m.choose(.recipe(Recipes.smallPlanet.id))
        #expect(m.goalsAllowed == [.play, .research] && m.goalReason == nil)
    }

    @Test func anyPresetIsAStartingPointWithItsOwnStepsAndGoals() async throws {
        let rig = try rig()
        defer { rig.finish() }
        await rig.scan()
        let m = rig.setup
        m.openCreate()
        for preset in Preset.allCases where Recipes.recipe(for: preset) == nil {
            m.choose(.preset(preset))
            #expect(Parse.integer(m.growSteps) != nil, "\(preset.rawValue): steps are always a number")
            #expect(m.controller.settings.preset == preset)
        }
        m.choose(.preset(.k64_3s))   // a planet of the third calibrated shape
        #expect(m.goalsAllowed == [.play, .research])
        m.choose(.preset(.tiny))   // a flat preset
        #expect(m.goalsAllowed == [.research])
        m.choose(.preset(.tiny3d))   // a 3D box: weather has its overlay
        #expect(m.goalsAllowed == [.play, .research])
    }

    @Test func aPlanetWhoseShapeTheWeatherIsNotCalibratedForIsToldBeforeAnythingGrows() async throws {
        let rig = try rig()
        defer { rig.finish() }
        await rig.scan()
        let m = rig.setup
        m.openCreate(.recipe(Recipes.smallPlanet.id))
        #expect(m.reasons.isEmpty)
        rig.controller.settings.levers.world3D.sphere.radius = "40"   // not a calibrated planet
        let reasons = m.reasons
        #expect(reasons.count == 1 && reasons[0].contains("weather") && !reasons[0].contains("checkpoint"), "\(reasons)")
        m.goal = .research
        #expect(m.reasons.isEmpty, "a research world has no weather to calibrate")
    }

    @Test func aPlanetCannotHaveLighterBrainsWhichTheEngineWouldRefuse() async throws {
        let rig = try rig()
        defer { rig.finish() }
        await rig.scan()
        let m = rig.setup
        m.openCreate(.recipe(Recipes.smallPlanet.id))
        #expect(m.lighterBrainsObstacle?.contains("planet") == true)
        rig.controller.settings.fast = true   // remembered from a flat world
        m.dropLighterBrainsIfRefused()
        #expect(!rig.controller.settings.fast, "--overlay fast must not reach a planet's command")
        m.openCreate(.recipe(Recipes.flatWorld.id))
        #expect(m.lighterBrainsObstacle == nil, "a flat world can have lighter brains")
    }

    @Test func theLengthIsAlwaysANumberAndTheLineSaysVeryShortUnderFiveThousand() async throws {
        let rig = try rig()
        defer { rig.finish() }
        await rig.scan()
        let m = rig.setup
        m.openCreate()
        m.setGrowSteps("")
        #expect(m.reasons.contains("Grow for must be a whole number of steps, 1 or more."))
        m.setGrowSteps("abc")
        #expect(!m.reasons.isEmpty)
        m.setGrowSteps("3000")
        #expect(m.reasons.isEmpty && m.growLine?.hasSuffix("Very short: the world will be mostly unevolved") == true)
        #expect(m.controller.settings.steps == "3000", "the number is the engine's --steps")
    }

    @Test func aWrongSettingIsToldInPlainWordsAndTheButtonIsGrey() async throws {
        let rig = try rig()
        defer { rig.finish() }
        await rig.scan()
        let m = rig.setup
        m.openCreate()
        rig.controller.settings.levers.seasonAmplitude = "2"
        #expect(m.reasons.contains("Season strength must be a number from 0 to 1."))
        #expect(m.fieldError("fields.season_amplitude") == "Season strength must be a number from 0 to 1.")
        #expect(m.fieldError("fields.patch_drift") == nil)
        rig.controller.settings.levers.seasonAmplitude = "0"
        #expect(m.reasons.isEmpty)
    }

    @Test func theNameIsCheckedAtEveryKeystrokeAndAFreeOneIsOffered() async throws {
        let rig = try rig()
        defer { rig.finish() }
        try LibraryTestFolders.research(rig.runs, "planet-1")
        await rig.scan()
        let m = rig.setup
        m.openCreate()
        #expect(m.name == "planet-2", "the suggestion skips the name that is taken")
        m.name = "planet-1"
        #expect(m.nameProblems == ["A world called planet-1 is already here."] && m.freeAlternative == "planet-2")
        m.name = "Planet-1"
        #expect(m.nameProblems.first?.contains("differ only in capitals") == true)
        m.name = ""
        #expect(m.nameProblems == ["Give the world a name."])
        m.name = "a b"
        #expect(m.nameProblems.count == 1 && m.reasons.contains(m.nameProblems[0]))
    }

    @Test func aSuggestedNameThatTheLibraryTakesLaterIsSuggestedAgainButATypedOneIsNeverChanged() async throws {
        let rig = try rig()
        defer { rig.finish() }
        let m = rig.setup
        m.openCreate()   // a launch opens this page before the first scan has finished: nothing is known yet
        #expect(m.name == "planet-1")
        try LibraryTestFolders.research(rig.runs, "planet-1-grow")
        await rig.scan()
        m.libraryChanged()   // what the page does when the library's worlds change
        #expect(m.name == "planet-2" && m.nameProblems.isEmpty, "\(m.name) \(m.nameProblems)")
        m.name = "mine"
        try LibraryTestFolders.research(rig.runs, "mine")
        await rig.scan()
        m.libraryChanged()
        #expect(m.name == "mine" && !m.nameProblems.isEmpty, "a name the person typed is theirs, and the form says it is taken")
    }

    @Test func advancedChoicesAreInTheCommandAndInTheChanges() async throws {
        let rig = try rig()
        defer { rig.finish() }
        await rig.scan()
        let m = rig.setup
        m.openCreate()
        rig.controller.settings.seed = "5"
        rig.controller.settings.fast = true
        m.withoutPowers = true
        let args = m.plan.stages.flatMap(\.arguments)
        #expect(args.contains("--seed") && args.contains("5") && args.contains("fast"))
        #expect(!args.contains("director.enabled=true"), "a world that takes no powers is made without the director")
        #expect(m.otherChanges.contains("Seed 5") && m.otherChanges.contains("Lighter brains") && m.otherChanges.contains("No powers in the planet"))
    }

    @Test func aHalfFilledFormSurvivesThePageAppearingAgainAndOnlyANewWorldEndsIt() async throws {
        let rig = try rig()
        defer { rig.finish() }
        try LibraryTestFolders.research(rig.runs, "src")
        await rig.scan()
        let m = rig.setup
        m.pageAppeared()   // the page restored at launch: a Create
        #expect(m.intent == .create)
        m.name = "mine"
        m.pageAppeared()
        #expect(m.name == "mine", "a Create form is kept when the page is shown again")
        m.openMakePlanet(from: "src")
        #expect(m.intent == .makePlanet(from: "src"))
        m.pageAppeared()   // a card asked for the page; and then the window of its own, and the sidebar's row
        m.pageAppeared()
        #expect(m.intent == .makePlanet(from: "src"), "an unfinished Make planet is not turned into a Create by the page appearing")
        m.openCreate()     // File > New World, the Worlds page's button, the form's own "New world instead"
        #expect(m.intent == .create)
    }

    @Test func theLengthOfAPresetWithoutOneIsTheEnginesAndSaysSo() async throws {
        let rig = try rig()
        defer { rig.finish() }
        await rig.scan()
        let m = rig.setup
        m.openCreate()
        #expect(m.stepsOrigin == "This is the recipe's own length.")
        m.choose(.preset(.k64_3s))
        #expect(m.growSteps == "100000" && m.stepsOrigin.contains("engine's usual length"), "\(m.growSteps) \(m.stepsOrigin)")
        m.choose(.preset(.tiny))
        #expect(m.growSteps != "30000" || m.stepsOrigin.contains("preset's own"))
    }

    @Test func theTimeLineSaysWhoseMachineMeasuredItAndThatTheFirstWorldIsSlower() async throws {
        let rig = try rig()
        defer { rig.finish() }
        await rig.scan()
        let m = rig.setup
        m.openCreate()
        #expect(m.timeSentence == "About 1 min 50 s on a recent Mac (measured on an M5 Pro), then about 10 s to make the planet.", "\(m.timeSentence ?? "")")
        #expect(m.timeNote?.contains("The first world after installing alife takes longer") == true)
        m.goal = .research
        #expect(m.timeSentence == "About 1 min 50 s on a recent Mac (measured on an M5 Pro).")
        m.setGrowSteps("3000")
        #expect(m.timeSentence?.contains("Very short") == true && m.timeSentence?.hasSuffix(".") == true && m.timeSentence?.contains("..") == false, "\(m.timeSentence ?? "")")
        m.openBranch(from: "x")
        #expect(m.timeSentence == nil && m.timeNote == nil)
    }

    @Test func familyLinesAndThePlanetsWordsAreInPlainWordsAndShownAsChanges() async throws {
        let rig = try rig()
        defer { rig.finish() }
        await rig.scan()
        let m = rig.setup
        m.openCreate()
        #expect(PlayHouses.allCases.map(\.plainTitle).sorted() == ["Automatically", "Exactly", "Roughly"])
        #expect(!m.otherChanges.contains { $0.contains("Family lines") })
        m.houses = .clade
        #expect(m.otherChanges.contains("Family lines rebuilt roughly"), "houses is remembered across forms, so it is shown: \(m.otherChanges)")
    }

    @Test func eachIntentShowsWhatItWillDoInTheCommand() async throws {
        let rig = try rig()
        defer { rig.finish() }
        try LibraryTestFolders.research(rig.runs, "src", steps: [10_000, 20_000])
        await rig.scan()
        let m = rig.setup
        m.openBranch(from: "src")
        #expect(m.commandBlocks.count == 1 && m.commandBlocks[0].command.contains("--init-from runs/src"))
        m.openKeepGrowing(world: "src")
        #expect(m.commandBlocks[0].command.contains("alife resume runs/src"))
        m.openMakePlanet(from: "src")
        #expect(m.commandBlocks[0].command.contains("play --init-from runs/src:20000"))
        m.openCreate()
        #expect(m.commandBlocks.count == 2)
    }
}
