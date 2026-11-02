import Foundation
import Testing

@testable import AlifeCore

/// The Changes list, the remembered values and the plain messages of the set-up page (docs/APP.md §3.5).
@Suite struct SetupSettingsTests {
    func tiny3s() -> RunSettings {
        var s = RunSettings()
        s.mode = .new
        s.preset = .tiny3s
        s.name = "x"
        let w = World3DLevers.preset("tiny3s", repo: Repo.root)!
        s.presetWorld3D = w
        s.levers = Levers(world3D: w)
        return s
    }

    // MARK: Levers by key

    @Test func everyKeyReadsAndWritesItsField() {
        for key in LeverKeys.allKeys {
            var l = Levers()
            let before = LeverKeys.text(key, in: l)
            #expect(before != nil, "\(key)")
            #expect(LeverKeys.title(key) != nil, "\(key)")
            #expect(LeverKeys.set(key, to: "7", in: &l), "\(key)")
            #expect(LeverKeys.text(key, in: l) == "7", "\(key)")
        }
        var l = Levers()
        #expect(!LeverKeys.set("not.a.key", to: "1", in: &l) && LeverKeys.text("not.a.key", in: l) == nil)
        #expect(LeverKeys.allKeys.count == Set(LeverKeys.allKeys).count)
    }

    @Test func theKeysAreTheOnesTheCommandPasses() {
        // every key `setArguments` can pass for a 3D planet with the feature levers is a key the page knows (the tied buoyancy keys excepted)
        var s = tiny3s()
        var l = s.levers
        for key in LeverKeys.allKeys where FeatureLevers.spec(key) == nil {
            guard let now = LeverKeys.text(key, in: l) else { continue }
            if key == "fields.season_mode" { LeverKeys.set(key, to: "per_patch", in: &l); continue }
            LeverKeys.set(key, to: Parse.decimal(now).map { String($0 + 1) } ?? now, in: &l)
        }
        s.levers = l
        let args = l.setArguments(baseline: Levers(world3D: s.presetWorld3D))
        var passed: [String] = []
        var i = 0
        while i + 1 < args.count { passed.append(String(args[i + 1].prefix { $0 != "=" })); i += 2 }
        for key in passed where !["mutation.buoy_init", "mutation.buoy_sigma"].contains(key) {
            #expect(LeverKeys.title(key) != nil, "\(key) is passed but has no plain name")
        }
    }

    // MARK: The Changes list

    @Test func theChangesAreExactlyWhatTheCommandWillSetWithWhatTheyWere() {
        var s = tiny3s()
        #expect(ChangeList.levers(s).isEmpty && ChangeList.sentence([]) == nil)
        s.levers.world3D.gravity = "0.05"
        s.levers.seasonPeriod = "5000"
        s.levers.seasonAmplitude = "0.3"
        let changes = ChangeList.levers(s)
        let was = LeverKeys.text("physics.gravity", in: LeverKeys.baseline(of: s))!
        #expect(changes.map(\.key).sorted() == ["fields.season_amplitude", "fields.season_period", "physics.gravity"])
        #expect(changes.first { $0.key == "physics.gravity" }?.sentence == "Gravity \(was) to 0.05")
        let sentence = ChangeList.sentence(changes)
        #expect(sentence?.hasPrefix("3 settings changed: ") == true)
        // the list and the command agree
        let command = s.effectiveLevers.setArguments(baseline: LeverKeys.baseline(of: s))
        for change in changes { #expect(command.contains { $0.hasPrefix("\(change.key)=") }, "\(change.key)") }
        #expect(ChangeList.sentence([changes[0]])?.hasPrefix("1 setting changed: ") == true)
    }

    @Test func aValueTypedAsTheSameNumberIsNotAChange() {
        var s = tiny3s()
        s.levers.rateMin = "0.0010"
        s.levers.seasonAmplitude = "0.0"
        #expect(ChangeList.levers(s).isEmpty)
    }

    @Test func aBranchIsComparedWithItsSourceNotWithTheEngineDefaults() {
        var s = RunSettings()
        s.mode = .fork
        s.sourceRun = "src"
        var base = Levers()
        base.rateMin = "0.01"
        s.leverBaseline = base
        s.levers = base
        #expect(ChangeList.levers(s).isEmpty, "the source's own value is the starting point")
        s.levers.rateMin = "0.001"
        #expect(ChangeList.levers(s).map(\.sentence) == ["Mutation rate floor 0.01 to 0.001"])
    }

    @Test func twoTextsAreOneValueWhenTheyAreTheSameNumberAFlagOrAWord() {
        #expect(LeverKeys.same("mutation.rate_min", "0.0010", "0.001") && LeverKeys.same("mutation.rate_min", "5e-3", "0.005"))
        #expect(!LeverKeys.same("mutation.rate_min", "0.002", "0.001"))
        #expect(LeverKeys.same("arms.enabled", "true", "true") && !LeverKeys.same("arms.enabled", "true", "false"))
        #expect(LeverKeys.same("fields.season_mode", "global", "global") && !LeverKeys.same("fields.season_mode", "global", "per_patch"))
        #expect(LeverKeys.same("physics.gravity", " 0.03 ", "0.03"))
    }

    @Test func aResumeHasNoChanges() {
        var s = RunSettings()
        s.mode = .resume
        s.levers.rateMin = "0.5"
        #expect(ChangeList.levers(s).isEmpty)
    }

    @Test func theOtherChoicesAreInTheListToo() {
        var s = tiny3s()
        s.seed = "1"
        #expect(ChangeList.others(s).isEmpty)
        s.seed = "7"
        s.fast = true
        s.timerEnabled = true
        s.timerHours = 2
        s.timerMinutes = 30
        s.director.mode = .live
        s.featureOverlays = ["arms"]
        let others = ChangeList.others(s)
        #expect(others.contains("Seed 7") && others.contains("Lighter brains") && others.contains("Stops after 2 h 30 min") && others.contains("Autopilot on"))
        #expect(others.contains { $0.hasPrefix("Add-ons: ") })
    }

    // MARK: Remembered values

    @Test func whatWasEditedIsRememberedAndComesBackOnTheNextForm() {
        var s = tiny3s()
        s.levers.rateMin = "0.002"
        s.levers.world3D.gravity = "0.09"
        let remembered = RememberedLevers.capture(s)
        #expect(remembered.values == ["mutation.rate_min": "0.002", "physics.gravity": "0.09"])
        var next = tiny3s()
        let applied = remembered.apply(to: &next)
        #expect(applied.keys.sorted() == ["mutation.rate_min", "physics.gravity"] && applied.skipped.isEmpty)
        #expect(next.levers.rateMin == "0.002" && next.levers.world3D.gravity == "0.09")
        // …and they are shown: the changes list names them
        #expect(ChangeList.levers(next).count == 2)
    }

    @Test func anEditNeverRidesIntoAWorldItDoesNotFit() {
        var remembered = RememberedLevers()
        remembered.values = ["physics.gravity": "0.5", "world.radius": "50", "mutation.rate_min": "0.004"]
        var flat = RunSettings()   // a 2D world
        var applied = remembered.apply(to: &flat)
        #expect(applied.keys == ["mutation.rate_min"] && applied.skipped.sorted() == ["physics.gravity", "world.radius"])
        #expect(flat.levers.world3D.gravity == "0" && flat.levers.rateMin == "0.004")
        // a branch keeps the shape of its source
        var fork = tiny3s()
        fork.mode = .fork
        fork.leverBaseline = Levers(world3D: fork.presetWorld3D)
        applied = remembered.apply(to: &fork)
        #expect(applied.skipped == ["world.radius"] && applied.keys.contains("physics.gravity"))
        // a box world has no planet radius
        var box = RunSettings()
        box.presetWorld3D = World3DLevers.preset("tiny3d", repo: Repo.root)!
        box.levers = Levers(world3D: box.presetWorld3D)
        applied = remembered.apply(to: &box)
        #expect(applied.skipped == ["world.radius"])
    }

    @Test func rememberedValuesSurviveStorage() {
        var r = RememberedLevers()
        r.values = ["mutation.rate_min": "0.002", "arms.enabled": "true"]
        #expect(RememberedLevers(data: r.data) == r)
        #expect(RememberedLevers(data: nil).isEmpty && RememberedLevers(data: Data("nonsense".utf8)).isEmpty)
        #expect(RememberedLevers(values: [:]).isEmpty)
    }

    @Test func nothingEditedRemembersNothing() {
        #expect(RememberedLevers.capture(tiny3s()).isEmpty)
    }

    // MARK: Plain messages

    @Test func theDefaultPathsMessagesAreInPlainWords() {
        var s = tiny3s()
        s.levers.rateMin = "2"
        s.levers.sigmaMin = "0"
        s.levers.immigrationFloor = "x"
        s.levers.seasonPeriod = "-1"
        s.levers.seasonAmplitude = "3"
        s.levers.world3D.gravity = "abc"
        s.seed = "q"
        s.steps = "0"
        s.name = ""
        let messages = s.validate(existingRuns: []).map(SetupMessages.plain)
        for message in messages {
            for key in LeverKeys.allKeys { #expect(!message.contains(key), "'\(key)' in \"\(message)\"") }
            #expect(!message.contains("must be a plain decimal"), "\(message)")
        }
        #expect(messages.contains("Mutation rate floor must be above 0 and below 1."))
        #expect(messages.contains("Mutation size floor must be a number above 0."))
        #expect(messages.contains("Newcomers when the population is below must be a whole number, 0 or more."))
        #expect(messages.contains("Season length must be a whole number of steps, 0 or more (0 is no seasons)."))
        #expect(messages.contains("Season strength must be a number from 0 to 1."))
        #expect(messages.contains("Gravity must be a number, 0 or more."))
        #expect(messages.contains("The seed must be a whole number, 0 or more."))
        #expect(messages.contains("Grow for must be a whole number of steps, 1 or more."))
        #expect(messages.contains("Give the world a name."))
    }

    @Test func aMessageWithACondition() {
        var s = tiny3s()
        s.levers.immigrationFloor = "5"
        s.levers.immigrantsPerStep = "0"
        s.levers.seasonAmplitude = "0.5"
        s.levers.seasonPeriod = "0"
        let messages = s.validate(existingRuns: []).map(SetupMessages.plain)
        #expect(messages.contains("Newcomers per step must be at least 1 while newcomers are added below a population floor."))
        #expect(messages.contains("Season length must be at least 1 when the seasons have a strength."))
    }

    @Test func aKeyThePageHasNoWordsForKeepsTheEnginesMessage() {
        let text = "arms.kappa must be a plain decimal number in [0, inf]."
        #expect(SetupMessages.plain(text) == text && SetupMessages.key(of: text) == nil)
        #expect(SetupMessages.key(of: "physics.gravity must be a plain decimal number >= 0.") == "physics.gravity")
        #expect(SetupMessages.plain("something else entirely") == "something else entirely")
    }

    @Test func everyMessageTheDefaultPathCanMakeHasPlainWords() {
        // every lever the default path reaches has a keyed message, and none of them uses a retired word
        let keys = (LeverKeys.dynamics + LeverKeys.world3D).map(\.key).filter { $0 != "fields.season_mode" }
        for key in keys {
            let probe = "\(key) must be x."
            let plain = SetupMessages.plain(probe)
            #expect(plain != probe, "\(key) has no plain message")
            for word in ["Resume", "Fork", "Checkpoint", "the Lab", "Source", "spin-up"] { #expect(!plain.contains(word), "\(plain)") }
        }
    }
}
