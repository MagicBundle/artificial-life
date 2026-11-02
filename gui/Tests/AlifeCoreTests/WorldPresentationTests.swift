import Foundation
import Testing

@testable import AlifeCore

@Suite struct WorldPresentationTests {
    // MARK: The player's pauses

    private let world = PlayHeader.Autopause(chapter: true, alert: ["pop_fall": true, "storm_named": false, "pop_surge": false])
    private let known: Set<String> = ["pop_fall", "storm_named", "pop_surge"]

    @Test func theWorldsOwnDefaultsNeedNoLine() {
        #expect(AutopausePrefs().commands(against: world, classes: known).isEmpty)
        #expect(AutopausePrefs(chapter: true, alert: ["pop_fall": true, "storm_named": false]).commands(against: world, classes: known).isEmpty)
    }

    @Test func aChoiceThatDiffersFromTheHeaderBecomesALine() {
        let prefs = AutopausePrefs(chapter: false, alert: ["storm_named": true, "pop_fall": false, "pop_surge": false])
        #expect(prefs.commands(against: world, classes: known)
                == [.autopause(.chapter(false)), .autopause(.alert("pop_fall", false)), .autopause(.alert("storm_named", true))])
    }

    @Test func aClassTheWorldDoesNotKnowIsLeftOut() {
        let prefs = AutopausePrefs(alert: ["never_heard_of": true, "pop_fall": false])
        #expect(prefs.commands(against: world, classes: known) == [.autopause(.alert("pop_fall", false))])
        #expect(AutopausePrefs(alert: ["x": true]).commands(against: nil, classes: []).isEmpty)
    }

    @Test func aWorldWithoutADefaultBlockIsTheDocumentedDefault() {
        // chapter on, every class off
        #expect(AutopausePrefs(chapter: false).commands(against: nil, classes: []) == [.autopause(.chapter(false))])
        #expect(AutopausePrefs(chapter: true, alert: ["a": false]).commands(against: nil, classes: ["a"]).isEmpty)
        #expect(AutopausePrefs().chapterOn(default: nil) && !AutopausePrefs().alertOn("a", default: nil))
        #expect(!AutopausePrefs(chapter: false).chapterOn(default: world) && AutopausePrefs().alertOn("pop_fall", default: world))
        #expect(AutopausePrefs(alert: ["pop_fall": false]).alertOn("pop_fall", default: world) == false)
    }

    @Test func thePrefsSurviveTheirStorageAndGarbageIsNothing() {
        let prefs = AutopausePrefs(chapter: false, alert: ["pop_fall": false, "storm_named": true])
        #expect(AutopausePrefs(json: prefs.json) == prefs)
        #expect(AutopausePrefs(json: nil) == AutopausePrefs() && AutopausePrefs(json: "not json") == AutopausePrefs())
        #expect(AutopausePrefs(json: "{}") == AutopausePrefs(), "an empty object is an empty choice")
    }

    // MARK: Attention

    @Test func aPauseCallsTheDockOnceAndOnlyWhenTheWindowIsNotInFront() {
        var tracker = AttentionTracker()
        let looking = tracker.paused(atStep: 6000, windowIsKey: true)
        let first = tracker.paused(atStep: 12_000, windowIsKey: false)
        let repeated = tracker.paused(atStep: 12_000, windowIsKey: false)
        let next = tracker.paused(atStep: 12_350, windowIsKey: false)
        #expect(!looking, "the player is looking")
        #expect(first)
        #expect(!repeated, "the same stop, announced again by the next clock")
        #expect(next, "a new stop")
    }

    @Test func theBadgeCountsTheAlertsRaisedWhileAwayAndGoesWhenTheWindowIsKey() {
        var tracker = AttentionTracker()
        #expect(tracker.badge == nil)
        tracker.raised(2, windowIsKey: true)
        #expect(tracker.badge == nil, "in front: nothing is unread")
        tracker.raised(2, windowIsKey: false)
        tracker.raised(0, windowIsKey: false)
        tracker.raised(1, windowIsKey: false)
        #expect(tracker.unread == 3 && tracker.badge == "3")
        tracker.windowBecameKey()
        #expect(tracker.unread == 0 && tracker.badge == nil)
        tracker.raised(1, windowIsKey: false)
        tracker.reset()
        #expect(tracker.badge == nil)
        let fresh = tracker.paused(atStep: 5, windowIsKey: false)
        #expect(fresh, "a new world starts afresh")
    }

    // MARK: The title

    @Test func theTitleIsNotTheRunsHandle() throws {
        let preset = try WorldFixtures.store("2d").header?.play  // (no source run: a preset or adopted world)
        let planet = try WorldFixtures.store("3d").header?.play  // (forked from runs/example:500000)
        #expect(WorldTitle.heading(world: "r", header: preset) == "World r")
        #expect(WorldTitle.heading(world: nil, header: nil) == "No world" && WorldTitle.heading(world: "  ", header: nil) == "No world")
        #expect(WorldTitle.heading(world: "p0", header: planet) == "Planet p0")
        #expect(WorldTitle.subtitle(header: planet) == "forked from runs/example:500000" && WorldTitle.subtitle(header: preset) == nil)
        #expect(WorldTitle.window(world: nil, header: nil) == "World" && WorldTitle.window(world: "p0", header: planet) == "Planet p0")
    }

    @Test func theNewWindowsSubtitleNamesTheGrownWorldNotTheDirectoryOrTheEnginesWord() throws {
        let planet = try WorldFixtures.store("3d").header?.play  // (made from runs/example:500000)
        let preset = try WorldFixtures.store("2d").header?.play
        #expect(WorldTitle.plainSubtitle(world: "p0", header: planet) == "made from example")
        #expect(WorldTitle.plainSubtitle(world: "example", header: planet) == "made from its grown world" || WorldTitle.plainSubtitle(world: "example", header: planet) == "made from example")
        #expect(WorldTitle.plainSubtitle(world: "p0", header: preset) == nil && WorldTitle.plainSubtitle(world: nil, header: nil) == nil)
        for text in [WorldTitle.plainSubtitle(world: "p0", header: planet) ?? ""] { #expect(!text.contains("runs/") && !text.contains("forked") && !text.contains(":")) }
    }

    @Test func aPlanetMadeFromItsOwnGrownWorldSaysSo() {
        // (the world "planet-1" made from "runs/planet-1-grow:30000": the default path's planet)
        #expect(WorldTitle.plainSubtitle(world: "planet-1", source: "runs/planet-1-grow:30000") == "made from its grown world")
        #expect(WorldTitle.plainSubtitle(world: "PLANET-1", source: "runs/planet-1-grow:30000") == "made from its grown world", "capitals do not tell worlds apart")
        // another planet's grown world, and a world the player named, are named as they are
        #expect(WorldTitle.plainSubtitle(world: "planet-2", source: "runs/planet-1-grow:30000") == "made from the grown world of planet-1")
        #expect(WorldTitle.plainSubtitle(world: "p", source: "runs/tiny3s_seed1") == "made from tiny3s_seed1")
        #expect(WorldTitle.plainSubtitle(world: " ", source: "runs/planet-1-grow") == "made from the grown world of planet-1")
        #expect(WorldTitle.plainSubtitle(world: "p", source: nil) == nil && WorldTitle.plainSubtitle(world: "p", source: "") == nil)
    }
}
