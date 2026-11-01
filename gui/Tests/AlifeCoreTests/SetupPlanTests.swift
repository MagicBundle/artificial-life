import Foundation
import Testing

@testable import AlifeCore

/// What one press of the set-up page's button makes (docs/APP.md §3.2): the commands are the existing builders', the worlds are the table's,
/// the names never collide.
@Suite struct SetupPlanTests {
    /// The form of a recipe as the page fills it, with the preset's real 3D values.
    func form(_ recipe: Recipe, name: String = "planet-1") -> RunSettings {
        var s = RunSettings()
        s.mode = .new
        s.preset = recipe.preset
        s.seed = "1"
        s.steps = String(recipe.growSteps)
        s.name = name
        if let w = World3DLevers.preset(recipe.preset.rawValue, repo: Repo.root) {
            s.presetWorld3D = w
            s.levers = Levers(world3D: w)
        }
        return s
    }

    func planet(_ recipe: Recipe, name: String = "planet-1") -> PlanetSpec {
        let w = World3DLevers.preset(recipe.preset.rawValue, repo: Repo.root) ?? World3DLevers()
        let shape = w.sphere.isSphere ? PlanetShape(radius: Double(w.sphere.radius)!, height: Double(w.height)!, nside: Int(w.sphere.nside)!, layers: Int(w.sphere.layers)!) : nil
        let size = PresetFile.preset(recipe.preset.rawValue, repo: Repo.root)?.double("world", "size")
        return PlanetSpec(
            source: "", name: name, weatherSpinup: PlanetSpec.defaultSpinup(forGeometry: w.sphere.geometry, shape: shape, worldSize: size),
            sourceWorldSize: size, sourceDims: w.dims, sourceGeometry: w.sphere.geometry, sourceShape: shape)
    }

    // MARK: The recipes

    @Test func smallPlanetIsTwoStagesWithTheCommandsTheSpecGives() {
        let plan = SetupPlan.make(intent: .create, goal: .play, name: "planet-1", settings: form(Recipes.smallPlanet), planet: planet(Recipes.smallPlanet), existing: [])
        #expect(plan.stages.count == 2)
        #expect(plan.stages[0].arguments == ["run", "--preset", "tiny3s", "--seed", "1", "--steps", "30000", "--out", "runs/planet-1-grow"])
        #expect(plan.stages[1].arguments == [
            "play", "--init-from", "runs/planet-1-grow", "--overlay", "stormsph16", "--set", "director.enabled=true",
            "--weather-spinup", "10000", "--houses", "auto", "--out", "runs/planet-1", "--rate", "1",
        ])
        #expect(plan.stages.map(\.world) == ["planet-1-grow", "planet-1"] && plan.stages[1].reads == "planet-1-grow")
        #expect(plan.stages.map(\.kind) == [.grow, .makePlanet])
        #expect(plan.pipelineMode == .growThenPlay)
    }

    @Test func biggerPlanetUsesItsOwnWeatherAndWarmUp() {
        let plan = SetupPlan.make(intent: .create, goal: .play, name: "p", settings: form(Recipes.biggerPlanet, name: "p"), planet: planet(Recipes.biggerPlanet, name: "p"), existing: [])
        #expect(plan.stages[0].arguments == ["run", "--preset", "mini3s", "--seed", "1", "--steps", "100000", "--out", "runs/p-grow"])
        #expect(Array(plan.stages[1].arguments.dropFirst(3).prefix(6)) == ["--overlay", "stormsph22", "--set", "director.enabled=true", "--weather-spinup", "8000"])
    }

    @Test func flatWorldIsOneResearchStageAndCannotBeAPlanet() {
        #expect(Recipes.flatWorld.goals == [.research])
        let plan = SetupPlan.make(intent: .create, goal: .research, name: "flat-1", settings: form(Recipes.flatWorld, name: "flat-1"), planet: nil, existing: [])
        #expect(plan.stages.count == 1 && plan.made.count == 1 && plan.madeSentence == nil)
        #expect(plan.stages[0].arguments == ["run", "--preset", "dev", "--seed", "1", "--steps", "20000", "--out", "runs/flat-1"])
        #expect(plan.pipelineMode == .growOnly)
        // the engine refuses the weather on a flat world; the spec says so before anything is grown
        #expect(planet(Recipes.flatWorld).sourceDims == 2)
        #expect(PlanetSpec.weatherObstacle(dims: planet(Recipes.flatWorld).sourceDims, geometry: "box") != nil)
    }

    @Test func aResearchGoalOnAPlanetRecipeIsOneWorld() {
        let plan = SetupPlan.make(intent: .create, goal: .research, name: "r", settings: form(Recipes.smallPlanet, name: "r"), planet: planet(Recipes.smallPlanet), existing: [])
        #expect(plan.stages.count == 1 && plan.stages[0].world == "r")
        #expect(plan.made == [MadeWorld(name: "r", kind: .research, role: "you study it")])
    }

    @Test func theRecipesAreDataThatMatchThePresetsTheyName() {
        for recipe in Recipes.all {
            #expect(PresetFile.preset(recipe.preset.rawValue, repo: Repo.root) != nil, "\(recipe.id)")
            #expect(recipe.growSteps >= 5_000, "a recipe is not a very short world")
            #expect(!recipe.goals.isEmpty && !recipe.title.isEmpty && !recipe.blurb.isEmpty)
        }
        #expect(Recipes.all.map(\.title) == ["Small planet", "Bigger planet", "Flat world"])
        #expect(Recipes.all.map(\.growSteps) == [30_000, 100_000, 20_000])
        #expect(Recipes.all.map(\.preset) == [.tiny3s, .mini3s, .dev])
        #expect(Set(Recipes.all.map(\.id)).count == 3)
        #expect(StartingPoint.initial.recipe == Recipes.smallPlanet)
    }

    @Test func theMeasuredNumbersAreSane() {
        let m = Recipes.smallPlanet.measured
        #expect(m != nil && m!.stepsPerSecond > 100 && m!.startSeconds < 60)
        let line = GrowEstimate.line(steps: 30_000, stepsPerSecond: m!.stepsPerSecond, startSeconds: m!.startSeconds)
        #expect(line == "about 1 min 50 s")
        #expect(GrowEstimate.line(steps: 3_000, stepsPerSecond: 300)?.hasSuffix("Very short: the world will be mostly unevolved") == true)
        #expect(GrowEstimate.line(steps: nil, stepsPerSecond: 300) == nil && GrowEstimate.line(steps: 5, stepsPerSecond: nil) == nil)
    }

    // MARK: Show command, with two blocks

    @Test func showCommandHasOneBlockPerStageAndCopyJoinsThemWithAndAnd() {
        let plan = SetupPlan.make(intent: .create, goal: .play, name: "planet-1", settings: form(Recipes.smallPlanet), planet: planet(Recipes.smallPlanet), existing: [])
        let blocks = plan.commandBlocks(folder: "/Users/me/alife")
        #expect(blocks.map(\.title) == ["Stage 1", "Stage 2"])
        #expect(blocks[0].command.hasPrefix("cd /Users/me/alife && .venv/bin/alife run --preset tiny3s"))
        #expect(blocks[1].command.contains(".venv/bin/alife play --init-from runs/planet-1-grow"))
        let script = plan.copyScript(folder: "/Users/me/alife")
        #expect(script.components(separatedBy: " && ").count == 3, "cd, stage 1, stage 2")
        #expect(script.hasPrefix("cd /Users/me/alife && .venv/bin/alife run") && script.contains(" && .venv/bin/alife play "))
        #expect(plan.commandBlocks(folder: "/a b").first?.command.hasPrefix("cd '/a b' &&") == true, "a folder with a space is quoted")
        #expect(plan.commandBlocks(folder: "").first?.command.hasPrefix("cd <alife folder> &&") == true)
        let single = SetupPlan.make(intent: .create, goal: .research, name: "r", settings: form(Recipes.flatWorld, name: "r"), planet: nil, existing: [])
        #expect(single.commandBlocks(folder: "/x").map(\.title) == ["Command"])
    }

    @Test func oneClickMakesTwoWorldsAndSaysSo() {
        let plan = SetupPlan.make(intent: .create, goal: .play, name: "planet-1", settings: form(Recipes.smallPlanet), planet: planet(Recipes.smallPlanet), existing: [])
        #expect(plan.made == [MadeWorld(name: "planet-1", kind: .played, role: "you play it"), MadeWorld(name: "planet-1-grow", kind: .research, role: "the grown world it starts from")])
        #expect(plan.madeSentence == "This makes 2 worlds: planet-1 (you play it) and planet-1-grow (the grown world it starts from).")
    }

    // MARK: The other intents use the builders as they are

    @Test func branchKeepGrowingAndMakePlanetAreTheBuildersOwnCommands() {
        var fork = RunSettings()
        fork.mode = .fork
        fork.sourceRun = "src"
        fork.seed = ""
        fork.name = "src-branch-1"
        var changed = fork.levers
        changed.rateMin = "0.002"
        fork.levers = changed
        let b = SetupPlan.make(intent: .branch(from: "src"), goal: .research, name: "src-branch-1", settings: fork, planet: nil, existing: ["src"])
        #expect(b.stages[0].arguments == fork.arguments(existingRuns: ["src"]))
        #expect(b.stages[0].arguments.prefix(4).elementsEqual(["run", "--init-from", "runs/src", "--out"]))
        #expect(b.stages[0].arguments.contains("mutation.rate_min=0.002") && b.stages[0].kind == .branch && b.stages[0].reads == "src")
        #expect(b.made == [MadeWorld(name: "src-branch-1", kind: .research, role: "branched from src")])

        var resume = RunSettings()
        resume.mode = .resume
        resume.sourceRun = "src"
        resume.steps = "5000"
        let k = SetupPlan.make(intent: .keepGrowing(world: "src"), goal: .research, name: "src", settings: resume, planet: nil, existing: ["src"])
        #expect(k.stages[0].arguments == ["resume", "runs/src", "--steps", "5000"] && k.stages[0].kind == .keepGrowing && k.made.isEmpty)
        #expect(k.pipelineMode == .keepGrowing)

        var spec = PlanetSpec(source: "src", sourceStep: 20_000, name: "ignored", sourceDims: 3, sourceGeometry: "sphere",
                              sourceShape: PlanetShape(radius: 36, height: 16, nside: 8, layers: 8))
        spec.weatherSpinup = 10_000
        let p = SetupPlan.make(intent: .makePlanet(from: "src"), goal: .play, name: "planet-9", settings: RunSettings(), planet: spec, existing: ["src"])
        var named = spec
        named.name = "planet-9"
        #expect(p.stages[0].arguments == PlayLaunch.arguments(planet: named))
        #expect(p.stages[0].arguments.contains("runs/src:20000") && p.stages.count == 1 && p.pipelineMode == .makePlanet)
    }

    // MARK: Names never collide

    @Test func aNameMustBeValidAndFreeForBothWorlds() {
        func problems(_ n: String, _ goal: Goal = .play, taken: Set<String> = []) -> [String] {
            SetupPlan.nameProblems(n, goal: goal, intent: .create, taken: taken)
        }
        #expect(problems("planet-1").isEmpty)
        #expect(problems("  ") == ["Give the world a name."])
        #expect(problems("bad name").count == 1 && problems("-x").count == 1 && problems("a/b").count == 1)
        #expect(problems("planet-1", taken: ["planet-1"]) == ["A world called planet-1 is already here."])
        // the seed takes a name too
        #expect(problems("planet-1", taken: ["planet-1-grow"]) == ["A world called planet-1-grow is already here."])
        #expect(problems("planet-1", .research, taken: ["planet-1-grow"]).isEmpty, "a research world has no seed")
        // capitals: one folder on a case-insensitive volume
        #expect(problems("Planet-1", taken: ["planet-1"]).first?.contains("differ only in capitals") == true)
        #expect(problems("planet-1", taken: ["PLANET-1-GROW"]).first?.contains("differ only in capitals") == true)
        // a keep-growing has no name to check
        #expect(SetupPlan.nameProblems("", goal: .research, intent: .keepGrowing(world: "w"), taken: []).isEmpty)
    }

    @Test func aSuggestedNameIsFreeWithItsSeedAndWithOfflineLinksAndCapitals() {
        #expect(SetupPlan.suggestName(base: "planet", goal: .play, taken: []) == "planet-1")
        #expect(SetupPlan.suggestName(base: "planet", goal: .play, taken: ["planet-1"]) == "planet-2")
        #expect(SetupPlan.suggestName(base: "planet", goal: .play, taken: ["planet-1-grow"]) == "planet-2", "its seed is taken")
        #expect(SetupPlan.suggestName(base: "planet", goal: .research, taken: ["planet-1-grow"]) == "planet-1")
        #expect(SetupPlan.suggestName(base: "planet", goal: .play, taken: ["PLANET-1", "Planet-2-Grow"]) == "planet-3")
        // a link to a drive that is not connected takes its name too (the catalog's other names include it)
        var catalog = PlayCatalog()
        catalog.other = ["planet-1"]
        #expect(SetupPlan.suggestName(base: "planet", goal: .play, taken: catalog.existingNames) == "planet-2")
        for _ in 0..<5 {
            let n = SetupPlan.suggestName(base: "planet", goal: .play, taken: Set((1...50).map { "planet-\($0)" }))
            #expect(n == "planet-51" && SetupPlan.nameProblems(n, goal: .play, intent: .create, taken: []).isEmpty)
        }
    }

    // MARK: The intents' words

    @Test func eachIntentHasItsOwnTitleAndButton() {
        #expect(SetupIntent.create.buttonLabel == "Create" && SetupIntent.branch(from: "a").buttonLabel == "Branch")
        #expect(SetupIntent.keepGrowing(world: "a").buttonLabel == "Keep growing" && SetupIntent.makePlanet(from: "a").buttonLabel == "Make planet")
        #expect(SetupIntent.create.title == "New world" && SetupIntent.branch(from: "a").title == "Branch from a")
        #expect(SetupIntent.create.world == nil && SetupIntent.keepGrowing(world: "w").world == "w")
        let labels = [SetupIntent.create, .branch(from: "a"), .keepGrowing(world: "a"), .makePlanet(from: "a")].map(\.buttonLabel)
        #expect(Set(labels).count == 4, "four actions, four words")
        for intent in [SetupIntent.create, .branch(from: "a"), .keepGrowing(world: "a"), .makePlanet(from: "a")] {
            for word in ["Resume", "Fork", "Checkpoint", "the Lab", "Home"] {
                #expect(!intent.summary.contains(word) && !intent.title.contains(word))
            }
        }
    }

    @Test func timeIsSaidInRoundNumbers() {
        #expect(TimeWords.about(5) == "a few seconds" && TimeWords.about(12) == "about 10 s" && TimeWords.about(59) == "about 60 s")
        #expect(TimeWords.about(60) == "about 1 min" && TimeWords.about(106) == "about 1 min 50 s" && TimeWords.about(482) == "about 8 min")
        #expect(TimeWords.about(700) == "about 12 min" && TimeWords.about(3_700) == "about 1 h" && TimeWords.about(5_400) == "about 1 h 30 min")
        #expect(TimeWords.clock(65) == "1:05" && TimeWords.clock(-3) == "0:00" && TimeWords.clock(754) == "12:34")
    }
}
