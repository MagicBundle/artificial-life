import AlifeCore
import AppKit
import SwiftUI
import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeControl

/// A hand-cranked clock for the model.
fileprivate final class Clock {
    var t: Double = 1000
}

fileprivate struct Rig {
    let model: WorldModel
    let play: PlayController
    let commands: URL
    let clock: Clock
    var sent: [String] { ((try? String(contentsOf: commands, encoding: .utf8)) ?? "").split(separator: "\n").map(String.init) }
}

/// A live world playing the 3D fixture that records the control lines it receives, with the camera at the region level
/// over the middle of the world.
@MainActor
fileprivate func makeRig(_ scratch: Scratch, kind: String = "3d") async throws -> Rig {
    let commands = scratch.url.appendingPathComponent("commands.txt")
    let script = try scratch.script("alife-live", "cat \"$1\"\nwhile IFS= read -r line; do echo \"$line\" >> \"$2\"; done")
    let play = PlayController()
    play.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url,
               arguments: [Repo.playFixtureURL(kind).path, commands.path])
    #expect(await eventually { play.store.header != nil && play.store.clock != nil && play.store.focus != nil })
    let model = WorldModel(play: play)
    let clock = Clock()
    model.now = { clock.t }
    model.setFreeRect(CGRect(x: 0, y: 0, width: 900, height: 700))
    model.goHome()
    model.edit {
        $0.distance = 30
        $0.target = SIMD3(24, 24, 0)
    }
    return Rig(model: model, play: play, commands: commands, clock: clock)
}

/// What a click, a key, a hand-off and a toast do to the World's model, against a played fixture (L = 48, height 24, seven
/// alive organisms at step 104): the commands that reach the world, the camera, the selection. No renderer, no window.
@MainActor
@Suite struct WorldInteractionTests {
    private let uid0: Int64 = 5_000_000_000
    private let free = CGRect(x: 0, y: 0, width: 900, height: 700)

    private func screen(_ rig: Rig, _ world: SIMD3<Float>) throws -> CGPoint {
        try #require(rig.model.projector(size: free.size).screen(world))
    }

    /// The point of the view over the isolated organism (uid0 + 147, row 21 at (40, 5, 9)).
    private func loner(_ rig: Rig) throws -> CGPoint {
        let frame = try #require(rig.play.store.frame)
        let index = try #require(rig.play.store.listedIndex(ofUID: uid0 + 147))
        return try screen(rig, frame.position(index))
    }

    /// Runs the camera's own motion until it arrives (or `frames` pass), one sixtieth of a second a frame.
    @discardableResult
    private func settle(_ rig: Rig, frames: Int = 1500) -> Int {
        for n in 1...frames {
            rig.clock.t += 1.0 / 60
            if !rig.model.advance() { return n }
        }
        return frames
    }

    private func sent(_ rig: Rig, _ line: String) async -> Bool {
        await eventually { rig.sent.contains(line) }
    }

    // MARK: Clicks

    @Test func aClickOnAnOrganismSelectsAndFollowsIt() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        let revision = rig.model.overlay.revision
        rig.model.click(at: try loner(rig), shift: false, clicks: 1)
        #expect(rig.model.selection == .organism(uid: uid0 + 147))
        #expect(rig.model.followTarget == .organism(uid: uid0 + 147), "D10: a single click follows")
        #expect(await sent(rig, "follow \(uid0 + 147)"))
        #expect(rig.model.overlay.revision > revision)
        #expect(rig.model.inspectorCard?.title == "Organism \(uid0 + 147)")
    }

    @Test func aShiftClickSelectsWithoutFollowing() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.click(at: try loner(rig), shift: true, clicks: 1)
        #expect(rig.model.selection == .organism(uid: uid0 + 147) && rig.model.followTarget == .none)
        #expect(await sent(rig, "focus \(uid0 + 147)"))
        #expect(!rig.sent.contains { $0.hasPrefix("follow") })
    }

    @Test func aClickOnNothingClearsTheSelectionAndStopsTheReports() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.click(at: try loner(rig), shift: false, clicks: 1)
        let empty = try screen(rig, SIMD3(40, 40, 0))
        #expect(WorldPicker.pick(at: empty, projector: rig.model.projector(size: free.size), store: rig.play.store).isNone)
        rig.model.click(at: empty, shift: false, clicks: 1)
        #expect(rig.model.selection.isNone && rig.model.followTarget == .none && rig.model.inspectorCard == nil)
        #expect(await sent(rig, "focus off"))
    }

    @Test func theSecondClickOfADoubleClickFliesIn() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        let at = try loner(rig)
        rig.model.click(at: at, shift: false, clicks: 1)
        #expect(!rig.model.isFlying)
        rig.model.click(at: at, shift: false, clicks: 2)
        #expect(rig.model.isFlying && rig.model.level == .region)
        settle(rig)
        #expect(rig.model.level == .organism, "it flew in to the organism level")
        let distance = rig.model.flyDistance(.organism)
        #expect(abs(rig.model.camera.distance - distance) < 0.3, "\(rig.model.camera.distance) against \(distance)")
        #expect(!rig.model.isFlying)
        #expect(simd_length(rig.model.camera.target - SIMD3(40, 5, 9)) < 1, "on the organism: \(rig.model.camera.target)")
    }

    @Test func aClickOnAStormSelectsItAndFliesThere() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.click(at: try screen(rig, SIMD3(4.5, 20.25, 0)), shift: false, clicks: 1)
        #expect(rig.model.selection == .storm(id: 7) && rig.model.followTarget == .none)
        #expect(rig.model.isFlying)
        settle(rig)
        let t = rig.model.camera.target
        #expect(abs(Torus.wrapDelta(t.x - 4.5, size: 48)) < 0.1 && abs(t.y - 20.25) < 0.1, "\(t)")
        #expect(rig.model.inspectorCard?.title == "Merrow")
        #expect(!rig.sent.contains { $0.hasPrefix("focus") || $0.hasPrefix("follow") }, "a storm is not reported by the world")
    }

    @Test func aDoubleClickOnAStormFollowsIt() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        let at = try screen(rig, SIMD3(4.5, 20.25, 0))
        rig.model.click(at: at, shift: false, clicks: 1)
        rig.model.click(at: at, shift: false, clicks: 2)
        #expect(rig.model.followTarget == .storm(id: 7))
    }

    @Test func aClickOnATerritoryCellSelectsItsKinGroupAndChangesNothingElse() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        let before = rig.model.camera
        let colour = rig.play.store.frame?.colorIndex
        // Cell (11, 2) is owned by slot 1, kin group 21 (see WorldPickerTests); no storm, pool or organism is there.
        let at = try screen(rig, SIMD3(17.2, 0.8, 0))
        rig.model.click(at: at, shift: false, clicks: 1)
        #expect(rig.model.selection == .kin(id: 21))
        #expect(rig.model.camera == before && !rig.model.isFlying && rig.model.followTarget == .none)
        #expect(rig.play.store.frame?.colorIndex == colour)
        #expect(rig.sent.isEmpty, "the global colour mode and the world are untouched")
    }

    @Test func aWarmPoolCanBeClickedAndFlownTo() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.click(at: try screen(rig, SIMD3(24, 12, 0)), shift: false, clicks: 1)
        #expect(rig.model.selection == .pool(index: 0))
        settle(rig)
        #expect(abs(rig.model.camera.target.y - 12) < 0.2)
        #expect(rig.model.inspectorCard?.title == "Warm pool")
    }

    // MARK: Follow

    @Test func theCameraFollowsASelectedOrganismAndADragStopsIt() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.click(at: try loner(rig), shift: false, clicks: 1)
        let start = rig.model.camera.target
        settle(rig)
        let end = rig.model.camera.target
        #expect(simd_length(end - SIMD3(40, 5, 9)) < 0.1 && simd_length(end - start) > 5, "it travelled to the organism: \(end)")
        // The organism moves; the camera goes with it.
        rig.play.store.apply(.frma(TestFrames.frame([SIMD3(30, 12, 5)], step: 110, uids: [uid0 + 147])))
        settle(rig)
        #expect(simd_length(rig.model.camera.target - SIMD3(30, 12, 5)) < 0.1)
        // A drag cancels the follow, not the selection.
        rig.model.drag(dx: 5, dy: 5, orbit: false)
        #expect(rig.model.followTarget == .none && rig.model.selection == .organism(uid: uid0 + 147))
        let after = rig.model.camera.target
        rig.play.store.apply(.frma(TestFrames.frame([SIMD3(10, 40, 5)], step: 120, uids: [uid0 + 147])))
        settle(rig)
        #expect(rig.model.camera.target == after, "it no longer follows")
    }

    @Test func afterADeathTheCameraHoldsThePointUntilTheHandOff() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.click(at: try loner(rig), shift: false, clicks: 1)
        settle(rig)
        let held = rig.model.camera.target
        rig.play.store.apply(.frma(TestFrames.frame([SIMD3(5, 5, 5)], step: 110, uids: [uid0 + 1])))  // it is not in the frame
        settle(rig)
        #expect(simd_length(rig.model.camera.target - held) < 0.05, "it stays at the death point")
        #expect(rig.model.inspectorCard?.alive == false)
        // The hand-off arrives: the camera goes to the successor.
        rig.model.handle(events: [handoff(from: uid0 + 147, to: uid0 + 1)], backlog: false)
        settle(rig)
        #expect(rig.model.selection == .organism(uid: uid0 + 1))
        #expect(simd_length(rig.model.camera.target - SIMD3(5, 5, 5)) < 0.1)
    }

    // MARK: Hand-off

    private func handoff(from: Int64, to: Int64, id: UInt64 = 111_400) -> WorldEvent {
        WorldEvent(id: id, step: 111, kind: EventKind.focusHandoff, fam: "focus", a: Int(from), b: Int(to), x: 1, y: 2, z: nil, imp: 1,
                   text: "Follow passed to a child.")
    }

    @Test func aHandOffFromTheSelectedOrganismMovesTheSelectionAndTheFollow() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.selectOrganism(uid0 + 7)
        rig.model.handle(events: [handoff(from: uid0 + 7, to: uid0 + 14)], backlog: false)
        #expect(rig.model.selection == .organism(uid: uid0 + 14))
        #expect(rig.model.followTarget == .organism(uid: uid0 + 14))
        #expect(rig.model.successors[uid0 + 7] == uid0 + 14)
        // The world already follows the successor (`follow` continues on the new uid): no command is needed.
        #expect(rig.sent.filter { $0.hasPrefix("follow") }.count <= 1)
    }

    @Test func aLateHandOffDoesNotOverrideANewerClick() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.selectOrganism(uid0 + 7)
        rig.model.selectOrganism(uid0 + 21)  // the player clicked another organism
        rig.model.handle(events: [handoff(from: uid0 + 7, to: uid0 + 14)], backlog: false)
        #expect(rig.model.selection == .organism(uid: uid0 + 21) && rig.model.followTarget == .organism(uid: uid0 + 21))
    }

    @Test func theBacklogChangesNothing() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.selectOrganism(uid0 + 7)
        let toast = WorldEvent(id: 5000, step: 5, kind: EventKind.stormDissolved, fam: "storm", a: 1, b: 0, x: nil, y: nil, z: nil, imp: 2, text: "old")
        rig.model.handle(events: [handoff(from: uid0 + 7, to: uid0 + 14), toast], backlog: true)
        #expect(rig.model.selection == .organism(uid: uid0 + 7) && rig.model.toasts.isEmpty)
    }

    // MARK: Toasts

    private func toastEvent(_ id: UInt64, imp: Int = 2) -> WorldEvent {
        WorldEvent(id: id, step: 200, kind: EventKind.stormDissolved, fam: "storm", a: 7, b: 0, x: 4.5, y: 20.25, z: nil, imp: imp, text: "Storm Merrow dissolved")
    }

    @Test func anImportantLiveEventRaisesAToastForThreeSeconds() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.handle(events: [toastEvent(1000), toastEvent(2000, imp: 1)], backlog: false)
        #expect(rig.model.toasts.map(\.id) == [1000])
        rig.clock.t += 2.9
        rig.model.expireToasts()
        #expect(rig.model.toasts.count == 1)
        rig.clock.t += 0.2
        rig.model.expireToasts()
        #expect(rig.model.toasts.isEmpty)
    }

    @Test func aBurstShowsTheNewestThreeAndAClickOnAToastGoesThere() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.handle(events: (1...6).map { toastEvent(UInt64($0) * 1000) }, backlog: false)
        #expect(rig.model.toasts.map(\.id) == [4000, 5000, 6000])
        let toast = try #require(rig.model.toasts.last)
        rig.model.activate(toast.action)
        rig.model.dismissToast(id: toast.id)
        #expect(rig.model.toasts.map(\.id) == [4000, 5000] && rig.model.selection == .storm(id: 7))
    }

    @Test func aToastExpiresByItselfWhenNobodyCallsBack() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.now = { Date().timeIntervalSince1970 }
        rig.model.handle(events: [toastEvent(1000)], backlog: false)
        #expect(rig.model.toasts.count == 1)
        #expect(await eventually(timeout: .seconds(6)) { rig.model.toasts.isEmpty }, "the model schedules its own expiry")
    }

    // MARK: Keys

    @Test func spaceTogglesPauseAndRunAccordingToTheClock() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        // The fixture's last CLCK says "user pause": space resumes.
        #expect(rig.model.clockStrip.paused)
        rig.model.perform(.togglePause)
        #expect(await sent(rig, "run"))
        // Asking for a rate unpauses; then space pauses.
        rig.play.setRate(.multiplier(1))
        #expect(!rig.model.clockStrip.paused)
        rig.model.perform(.togglePause)
        #expect(await sent(rig, "pause"))
    }

    @Test func theRateKeysWalkTheRungsOfARunningWorld() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.perform(.faster)  // paused: no rate to change
        rig.model.perform(.slower)
        #expect(rig.sent.isEmpty)
        rig.play.setRate(.multiplier(1))
        rig.model.perform(.faster)
        rig.model.perform(.faster)
        rig.model.perform(.faster)
        rig.model.perform(.faster)  // max is the end
        rig.model.perform(.slower)
        #expect(await eventually { rig.sent == ["rate 1", "rate 2", "rate max", "rate 2"] }, "\(rig.sent)")
    }

    @Test func theStepKeyStepsOnceAndTheColourKeyCyclesTheModes() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.perform(.step)
        #expect(await sent(rig, "step"))
        // The fixture's newest frame is in colour mode 3 ("house"); the next is "kin".
        let store = rig.play.store
        let modes = try #require(store.header?.colorModes)
        let current = try #require(store.frame.flatMap { store.header?.colorMode(index: $0.colorIndex) })
        rig.model.perform(.cycleColor)
        let next = try #require(WorldKeys.nextColorMode(after: current, in: modes))
        #expect(await sent(rig, "color \(next)") && next != current)
    }

    @Test func theLayerKeysToggleTheViewMenusSwitches() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        #expect(rig.model.level == .region)
        rig.model.perform(.toggleAgents)
        #expect(!rig.model.display.agents && !rig.model.display.agentPoints, "at the region level the key is the instances")
        rig.model.perform(.toggleBonds)
        rig.model.perform(.toggleTerritory)
        #expect(!rig.model.display.bonds && !rig.model.display.territory)
        rig.model.edit { $0.distance = 75 }
        #expect(rig.model.level == .planet)
        rig.model.perform(.toggleAgents)
        #expect(rig.model.display.agentPoints, "at the planet level it is the 1 px points")
    }

    @Test func escapeClearsTheSelectionAndTheFollowAndResetPutsTheCameraHome() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.click(at: try loner(rig), shift: false, clicks: 1)
        rig.model.perform(.escape)
        #expect(rig.model.selection.isNone && rig.model.followTarget == .none)
        #expect(await sent(rig, "focus off"))
        rig.model.selectStorm(7, fly: true, follow: true)
        rig.model.perform(.resetCamera)
        #expect(rig.model.cameraIsHome && rig.model.followTarget == .none && !rig.model.isFlying)
        #expect(rig.model.selection == .storm(id: 7), "a reset moves the camera, not the selection")
    }

    @Test func fFollowsTheSelectionWhateverItIs() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.perform(.followSelection)
        #expect(rig.model.followTarget == .none && rig.model.note != nil, "nothing selected: it says so")
        rig.model.setSelection(.storm(id: 7))
        rig.model.perform(.followSelection)
        #expect(rig.model.followTarget == .storm(id: 7))
        rig.model.setSelection(.kin(id: 17))
        rig.model.perform(.followSelection)
        #expect(rig.model.followTarget == .lineage(kinID: 17))
        rig.model.selectOrganism(uid0 + 147, follow: true, camera: false)
        rig.model.perform(.followSelection)
        #expect(rig.model.followTarget == .organism(uid: uid0 + 147) && rig.model.isFlying)
    }

    // MARK: Chronicle clicks, the cast, hover

    @Test func aChronicleClickFliesToTheEventAndSelectsItsEntity() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        let kinBorn = try #require(rig.play.store.chronicle.events.first { $0.kind == EventKind.kinBorn })  // kin group 17 at (6, 24)
        rig.model.activate(kinBorn)
        #expect(rig.model.selection == .kin(id: 17) && rig.model.isFlying)
        settle(rig)
        #expect(rig.model.camera.target.x < 12, "went west towards the group: \(rig.model.camera.target)")
        let named = try #require(rig.play.store.chronicle.events.first { $0.kind == EventKind.stormNamed })
        rig.model.activate(named)
        #expect(rig.model.selection == .storm(id: 7))
    }

    @Test func aChronicleClickOnAGoneStormStillFliesToWhereItHappened() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        let gone = WorldEvent(id: 9000, step: 9, kind: EventKind.stormDissolved, fam: "storm", a: 99, b: 0, x: 40, y: 10, z: nil, imp: 2, text: "x")
        rig.model.activate(gone)
        #expect(rig.model.selection == .storm(id: 99))
        settle(rig)
        #expect(abs(rig.model.camera.target.x - 40) < 0.2 && abs(rig.model.camera.target.y - 10) < 0.2)
        #expect(rig.model.inspectorCard?.alive == false)
    }

    @Test func aCastRowFliesToAndSelectsAKinGroup() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        let colour = rig.play.store.frame?.colorIndex
        rig.model.selectKin(id: 21, fly: true, follow: false)  // kin group 21: uid0 + 14 and + 21, near (11, 11)
        #expect(rig.model.selection == .kin(id: 21) && rig.model.isFlying)
        settle(rig)
        let t = rig.model.camera.target
        #expect(abs(t.x - 11) < 1.5 && abs(t.y - 11) < 1.5, "\(t)")
        #expect(rig.play.store.frame?.colorIndex == colour && rig.sent.isEmpty, "the global colour mode is not changed")
        #expect(rig.model.inspectorCard?.title == "Brennar kin")
    }

    @Test func hoveringNamesWhatIsUnderThePointerAndLeavingClearsIt() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.hover(at: try loner(rig))
        let info = try #require(rig.model.overlay.hover)
        guard case .organism(let uid, _) = info.pick else {
            Issue.record("expected an organism, got \(info.pick)")
            return
        }
        #expect(uid == uid0 + 147 && info.text.contains("organism \(uid0 + 147)"))
        rig.clock.t += 1
        rig.model.hover(at: try screen(rig, SIMD3(40, 40, 0)))
        #expect(rig.model.overlay.hover == nil)
        rig.clock.t += 1
        rig.model.hover(at: try screen(rig, SIMD3(4.5, 20.25, 0)))
        #expect(rig.model.overlay.hover?.text == "Merrow")
        rig.model.hover(at: nil)
        #expect(rig.model.overlay.hover == nil)
    }

    @Test func hoveringIsThrottledToSixtyAPick() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.hover(at: try loner(rig))
        #expect(rig.model.overlay.hover != nil)
        rig.model.hover(at: try screen(rig, SIMD3(40, 40, 0)))  // within a sixtieth of a second: skipped
        #expect(rig.model.overlay.hover != nil)
        rig.clock.t += WorldModel.hoverPeriod * 1.01
        rig.model.hover(at: try screen(rig, SIMD3(40, 40, 0)))
        #expect(rig.model.overlay.hover == nil)
    }

    @Test func aNewWorldForgetsTheSelectionTheFollowAndTheToasts() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.click(at: try loner(rig), shift: false, clicks: 1)
        rig.model.handle(events: [toastEvent(1000)], backlog: false)
        rig.model.deliver(.header)
        #expect(rig.model.selection.isNone && rig.model.followTarget == .none && rig.model.toasts.isEmpty)
        #expect(rig.model.successors.isEmpty && !rig.model.isFlying)
    }

    @Test func aChronicleClickOnADeadOrganismFliesToWhereItDiedAndFollowsNothing() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        let gone = uid0 + 999  // not in the newest frame
        let death = WorldEvent(id: 9100, step: 9, kind: EventKind.focusDeath, fam: "focus", a: Int(gone), b: 0, x: 5, y: 6, z: nil, imp: 1,
                               text: "Organism died.")
        rig.model.activate(death)
        #expect(rig.model.selection == .organism(uid: gone) && rig.model.isFlying)
        #expect(rig.model.followTarget == .none, "a dead organism has no position to follow")
        #expect(await sent(rig, "focus \(gone)"))
        let frames = settle(rig)
        #expect(frames < 1500 && !rig.model.isFlying, "the fly ended: \(frames)")
        #expect(abs(rig.model.camera.target.x - 5) < 0.2 && abs(rig.model.camera.target.y - 6) < 0.2)
        #expect(rig.model.inspectorCard?.alive == false)
    }

    @Test func aFlyWhileFollowingSomethingNotYetInAFrameGoesWhereItWasSent() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.setFollow(.organism(uid: uid0 + 999))  // (a successor the frame does not hold yet)
        rig.model.flyTo(SIMD3(10, 10, 0), distance: nil)
        #expect(rig.model.advanceCamera(dt: 1.0 / 60), "it moves")
        settle(rig)
        #expect(abs(rig.model.camera.target.x - 10) < 0.2 && !rig.model.isFlying)
    }

    @Test func aWideWindowsHomePoseIsNotSnappedInByTheFirstDrag() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.setFreeRect(CGRect(x: 0, y: 0, width: 1100, height: 460))  // a bottom dock: 2.39 to 1
        rig.model.goHome()
        let home = rig.model.camera.distance
        rig.model.drag(dx: 1, dy: 0, orbit: false)
        #expect(abs(rig.model.camera.distance - home) < 1e-3, "\(home) became \(rig.model.camera.distance)")
        rig.model.zoom(by: 1.5)
        #expect(rig.model.camera.distance == home, "and it cannot be zoomed out past it")
    }

    @Test func aWeatherWorldOpensAtThePlanetLevelAndADryOneDoesNot() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let wet = try await makeRig(scratch)
        defer { wet.play.shutdown() }
        #expect(wet.play.store.map != nil)
        wet.model.goHome()
        #expect(wet.model.level == .planet)
        wet.model.display.map = false  // no map in view, nothing for the planet level to protect
        #expect(wet.model.level == .region)
        wet.model.display.map = true
        #expect(wet.model.level == .planet)
        let flat = try Scratch()
        defer { flat.cleanup() }
        let dry = try await makeRig(flat, kind: "2d")
        defer { dry.play.shutdown() }
        dry.model.goHome()
        #expect(dry.play.store.map == nil)
        #expect(dry.model.level == .region, "the organisms of a 2D world are in view at home")
        #expect(WorldLevels.agentLayer(at: dry.model.level, display: dry.model.display) == .instances)
    }

    @Test func zoomingOutAtAnObliquePitchTiltsTheViewBackSoNoEdgeOfTheMapShows() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        let size = rig.model.worldSize
        rig.model.edit {
            $0.pitch = -30 * .pi / 180
            $0.distance = 100 * size
        }
        let camera = rig.model.camera
        #expect(camera.pitch < -30 * .pi / 180, "steeper than asked")
        let seen = try #require(camera.farthestFloorDistance(aspect: rig.model.aspect))
        #expect(seen <= Camera.tiledReach(size: size))
    }

    @Test func aWorldThatIsStoppedOrReplacedForgetsWhatWasSelectedAtOnce() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.click(at: try loner(rig), shift: false, clicks: 1)
        #expect(!rig.model.selection.isNone)
        rig.play.stop()  // (no header of a new world has arrived: the old selection must not outlive the old store)
        #expect(rig.model.selection.isNone && rig.model.followTarget == .none && rig.model.inspectorCard == nil)
    }

    @Test func theRequestedRateStaysUntilTheClockShowsItOrASecondHasPassed() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        var now = 100.0
        rig.play.now = { now }
        let clock = try #require(rig.play.store.clock)
        func shows(_ rate: Float) -> PlayClock {
            var c = clock
            c.rate = rate
            return c
        }
        rig.play.setRate(.max)
        #expect(rig.play.requestedRate == .max)
        rig.play.settleRequest(after: shows(0))  // a periodic CLCK sent before the world read the line
        #expect(rig.play.requestedRate == .max, "it does not flip the strip back to the old state")
        rig.play.settleRequest(after: shows(-1))
        #expect(rig.play.requestedRate == nil, "confirmed")
        rig.play.pause()
        #expect(rig.play.requestedRate == .multiplier(0))
        rig.play.settleRequest(after: shows(2))
        #expect(rig.play.requestedRate == .multiplier(0))
        now += 1.5  // a clock this long after the asking is the world's answer, whatever it says
        rig.play.settleRequest(after: shows(2))
        #expect(rig.play.requestedRate == nil)
        rig.play.setRate(.multiplier(0.25))
        rig.play.resume()  // a later intent replaces the pending one
        #expect(rig.play.requestedRate == nil)
        #expect(PlayController.confirms(0.25, .multiplier(0.25)) && !PlayController.confirms(-1, .multiplier(0)))
        #expect(PlayController.confirms(0, .multiplier(0)) && PlayController.confirms(-1, .max) && !PlayController.confirms(1, .max))
    }

    @Test func theLabelsSwitchIsTheWorldsAndIsOnAgainInANewWorld() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        #expect(rig.play.labelsOn)
        rig.play.send(.labels(false))
        #expect(!rig.play.labelsOn)
        rig.play.stop()
        #expect(rig.play.labelsOn)
    }

    @Test func aZoomCancelsTheDistanceOfAFlyButNotItsPlace() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.selectStorm(7, fly: true, follow: false)
        #expect(rig.model.flyDistance != nil || rig.model.isFlying)
        rig.model.zoom(by: 0.5)
        #expect(rig.model.flyDistance == nil)
        settle(rig)
        #expect(abs(rig.model.camera.target.y - 20.25) < 0.2, "the fly still ends at the storm")
    }
}

extension WorldInteractionTests {
    // (more of the same suite: the cases the mutation checks asked for)

    private func idleRig(_ scratch: Scratch) async throws -> Rig { try await makeRig(scratch) }

    @Test func aShiftClickOnAnotherOrganismStopsTheCameraFollowingTheFirst() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await idleRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.selectOrganism(uid0 + 7)
        #expect(rig.model.followTarget == .organism(uid: uid0 + 7))
        rig.model.selectOrganism(uid0 + 14, follow: false, camera: false)
        #expect(rig.model.selection == .organism(uid: uid0 + 14) && rig.model.followTarget == .none)
        #expect(await sent(rig, "focus \(uid0 + 14)"))
    }

    @Test func aFlyNeverGoesOutToReachAnOrganismOrAStorm() async throws {
        // In a world of 320 the organism level is 25.6 away and a region fly ends 85 away. A camera closer than that stays.
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await idleRig(scratch)
        defer { rig.play.shutdown() }
        var header = try #require(rig.play.store.header)
        header.size = 320
        header.height = 160
        var store = rig.play.store
        store.reset(header: header)
        store.apply(.frma(TestFrames.frame([SIMD3(100, 100, 5)], uids: [uid0 + 1])))
        rig.play.store = store
        rig.model.goHome()
        rig.model.edit { $0.distance = 12 }
        let organism = rig.model.flyDistance(.organism)
        let region = rig.model.flyDistance(.region)
        #expect(organism > 12 && region > organism, "\(organism), \(region)")
        rig.model.selectOrganism(uid0 + 1, fly: true)
        #expect(!rig.model.isFlying && rig.model.flyDistance == nil, "12 is closer than the organism level's \(organism)")
        rig.model.edit { $0.distance = 200 }
        rig.model.selectOrganism(uid0 + 1, fly: true)
        #expect(rig.model.isFlying && rig.model.flyDistance == organism, "from far away it flies in")
        rig.model.cancelCameraMotion()
        // The same for a storm: from 12 it keeps 12, from 200 it comes to the region distance.
        rig.model.edit { $0.distance = 12 }
        rig.model.flyTo(SIMD3(100, 100, 0), distance: min(rig.model.camera.distance, region))
        #expect(rig.model.flyDistance == 12)
        rig.model.cancelCameraMotion()
        rig.model.edit { $0.distance = 200 }
        rig.model.flyTo(SIMD3(100, 100, 0), distance: min(rig.model.camera.distance, region))
        #expect(rig.model.flyDistance == region)
    }

    @Test func aStormFlyKeepsACloserCameraAndBringsAFarOneToTheRegionDistance() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await idleRig(scratch)
        defer { rig.play.shutdown() }
        let region = rig.model.flyDistance(.region)
        rig.model.edit { $0.distance = region * 0.6 }
        let close = rig.model.camera.distance
        rig.model.selectStorm(7, fly: true, follow: false)
        #expect(rig.model.flyDistance == close, "closer than the region distance: kept")
        rig.model.cancelCameraMotion()
        rig.model.edit { $0.distance = region * 3 }
        rig.model.selectStorm(7, fly: true, follow: false)
        #expect(rig.model.flyDistance == region)
        settle(rig)
        #expect(abs(rig.model.camera.distance - region) < 0.3, "arrived at the region distance: \(rig.model.camera.distance)")
    }

    @Test func aChronicleClickOnAHandOffFollowsTheSuccessor() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await idleRig(scratch)
        defer { rig.play.shutdown() }
        let event = handoff(from: uid0 + 7, to: uid0 + 14)
        rig.model.activate(event)
        #expect(rig.model.selection == .organism(uid: uid0 + 14) && rig.model.followTarget == .organism(uid: uid0 + 14))
        #expect(await sent(rig, "follow \(uid0 + 14)"))
    }

    @Test func aChronicleClickOnADirectorEntryFliesToWhereItActed() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await idleRig(scratch)
        defer { rig.play.shutdown() }
        let entry = WorldEvent(id: 7000, step: 7, kind: EventKind.directorEntry, fam: "director", a: 0, b: 0, x: 40, y: 10, z: nil, imp: 1,
                               text: "Director: bloom")
        rig.model.activate(entry)
        #expect(rig.model.selection.isNone && rig.model.isFlying)
        settle(rig)
        #expect(abs(rig.model.camera.target.x - 40) < 0.2 && abs(rig.model.camera.target.y - 10) < 0.2)
        // An event with no place (a checkpoint) goes nowhere.
        let before = rig.model.camera
        rig.model.activate(WorldEvent(id: 8000, step: 8, kind: EventKind.checkpoint, fam: "system", a: 0, b: 0, x: nil, y: nil, z: nil, imp: 0, text: "c"))
        #expect(!rig.model.isFlying && rig.model.camera == before)
    }

    @Test func aFlyIsQuickerThanTheFollowsGlide() async throws {
        // 20 units in one second: a fly (a time constant of 0.35 s) is within a quarter of the way, a follow (1 s) is not.
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await idleRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.flyTo(SIMD3(24, 4, 0), distance: nil)
        for _ in 0..<60 {
            rig.clock.t += 1.0 / 60
            _ = rig.model.advance()
        }
        let flown = abs(rig.model.camera.target.y - 4)
        #expect(flown < 5, "a fly after a second: \(flown) units from the place (it started 20 away)")
        rig.model.cancelCameraMotion()
        rig.model.edit { $0.target = SIMD3(24, 24, 0) }
        rig.model.setFollow(.storm(id: 7))  // storm 7 is at (4.5, 20.25): 20 units west
        for _ in 0..<60 {
            rig.clock.t += 1.0 / 60
            _ = rig.model.advance()
        }
        let followed = simd_length(rig.model.camera.target - SIMD3(4.5, 20.25, 0))
        #expect(followed > 8, "a follow after a second is still on its way: \(followed)")
    }

    @Test func hoveringNamesTheOrganismAtTheCentreOfTheView() async throws {
        // (The model asks nothing of the world to hover: the same hit test as a click, over the records it holds.)
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await idleRig(scratch)
        defer { rig.play.shutdown() }
        rig.model.edit { $0.target = SIMD3(40, 5, 9) }
        rig.model.hover(at: CGPoint(x: 450, y: 350))
        #expect(rig.model.overlay.hover?.text.contains("organism") == true)
    }
}

/// The view's own handling of the pointer and the keys, driven by synthesised events (no window, no renderer): a press and a
/// release is a click unless the pointer travelled; the pointer moving names what it is over; the keys are `WorldKeys`.
@MainActor
@Suite struct WorldViewInputTests {
    private let uid0: Int64 = 5_000_000_000

    private func view(for rig: Rig) -> WorldMTKView {
        let view = WorldMTKView(frame: CGRect(x: 0, y: 0, width: 900, height: 700), device: nil)
        view.model = rig.model
        return view
    }

    /// AppKit's point (origin at the bottom left) of a point of the view (origin at the top left).
    private func appKit(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x, y: 700 - p.y) }

    private func mouse(_ type: NSEvent.EventType, at p: CGPoint, clicks: Int = 1, flags: NSEvent.ModifierFlags = []) throws -> NSEvent {
        try #require(NSEvent.mouseEvent(with: type, location: appKit(p), modifierFlags: flags, timestamp: 0, windowNumber: 0,
                                        context: nil, eventNumber: 0, clickCount: clicks, pressure: 1))
    }

    private func key(_ characters: String, flags: NSEvent.ModifierFlags = []) throws -> NSEvent {
        try #require(NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: flags, timestamp: 0, windowNumber: 0, context: nil,
                                      characters: characters, charactersIgnoringModifiers: characters, isARepeat: false, keyCode: 0))
    }

    private func loner(_ rig: Rig) throws -> CGPoint {
        let frame = try #require(rig.play.store.frame)
        let index = try #require(rig.play.store.listedIndex(ofUID: uid0 + 147))
        return try #require(rig.model.projector(size: CGSize(width: 900, height: 700)).screen(frame.position(index)))
    }

    @Test func aPressAndReleaseInPlaceIsAClickAtTheRightPoint() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        let view = view(for: rig)
        let at = try loner(rig)
        view.mouseDown(with: try mouse(.leftMouseDown, at: at))
        view.mouseUp(with: try mouse(.leftMouseUp, at: at))
        #expect(rig.model.selection == .organism(uid: uid0 + 147), "the y axis is flipped from AppKit's")
    }

    @Test func aReleaseAFewPointsAwayIsStillAClickButALongDragIsNot() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        let view = view(for: rig)
        let at = try loner(rig)
        view.mouseDown(with: try mouse(.leftMouseDown, at: at))
        view.mouseDragged(with: try mouse(.leftMouseDragged, at: CGPoint(x: at.x + 2, y: at.y)))  // under the slop
        view.mouseUp(with: try mouse(.leftMouseUp, at: CGPoint(x: at.x + 2, y: at.y)))
        #expect(rig.model.selection == .organism(uid: uid0 + 147))
        rig.model.clearSelection()
        view.mouseDown(with: try mouse(.leftMouseDown, at: at))
        view.mouseDragged(with: try mouse(.leftMouseDragged, at: CGPoint(x: at.x + 30, y: at.y)))  // a drag
        view.mouseUp(with: try mouse(.leftMouseUp, at: CGPoint(x: at.x + 30, y: at.y)))
        #expect(rig.model.selection.isNone, "a pan is not a click")
    }

    @Test func shiftAndOptionAndTheClickCountReachTheModel() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        let view = view(for: rig)
        let at = try loner(rig)
        view.mouseDown(with: try mouse(.leftMouseDown, at: at, flags: .shift))
        view.mouseUp(with: try mouse(.leftMouseUp, at: at, flags: .shift))
        #expect(rig.model.selection == .organism(uid: uid0 + 147) && rig.model.followTarget == .none, "shift: select only")
        rig.model.clearSelection()
        view.mouseDown(with: try mouse(.leftMouseDown, at: at, flags: .option))
        view.mouseUp(with: try mouse(.leftMouseUp, at: at, flags: .option))
        #expect(rig.model.selection.isNone, "option is the orbit modifier, not a pick")
        view.mouseDown(with: try mouse(.leftMouseDown, at: at, clicks: 2))
        view.mouseUp(with: try mouse(.leftMouseUp, at: at, clicks: 2))
        #expect(rig.model.isFlying, "a double click flies in")
    }

    @Test func aReleaseWithoutAPressIsNothing() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        let view = view(for: rig)
        view.mouseUp(with: try mouse(.leftMouseUp, at: try loner(rig)))  // e.g. the press went to the dock's divider
        #expect(rig.model.selection.isNone)
    }

    @Test func thePointerMovingNamesWhatIsUnderItAndLeavingClearsIt() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        let view = view(for: rig)
        view.mouseMoved(with: try mouse(.mouseMoved, at: try loner(rig)))
        #expect(rig.model.overlay.hover?.text.contains("organism \(uid0 + 147)") == true)
        let exit = try #require(NSEvent.enterExitEvent(with: .mouseExited, location: .zero, modifierFlags: [], timestamp: 0, windowNumber: 0,
                                                       context: nil, eventNumber: 0, trackingNumber: 0, userData: nil))
        view.mouseExited(with: exit)
        #expect(rig.model.overlay.hover == nil)
        rig.clock.t += 1
        view.mouseMoved(with: try mouse(.mouseMoved, at: try loner(rig)))
        #expect(rig.model.overlay.hover != nil)
        view.mouseDown(with: try mouse(.leftMouseDown, at: try loner(rig)))
        #expect(rig.model.overlay.hover == nil, "no tooltip over a press")
    }

    @Test func theKeysAreTheWorldKeysAndAChordIsNot() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        let view = view(for: rig)
        view.keyDown(with: try key(".", flags: .command))  // a menu chord is the menu's
        view.keyDown(with: try key("q"))  // no action
        view.keyDown(with: try key("."))
        view.keyDown(with: try key(" "))  // paused in the fixture: space runs (and ends the sequence)
        #expect(await eventually { rig.sent.last == "run" })
        #expect(rig.sent == ["step", "run"], "the chord and the unknown key sent nothing: \(rig.sent)")
        view.keyDown(with: try key("\u{1B}"))
        #expect(rig.model.selection.isNone)
        view.keyDown(with: try key("3"))
        #expect(!rig.model.display.territory)
    }

    @Test func theWorldKeysWorkWhateverHasTheFocusButATextField() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        let view = view(for: rig)
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 900, height: 700), styleMask: [.titled], backing: .buffered, defer: true)
        let content = NSView(frame: window.contentRect(forFrameRect: window.frame))
        let list = NSTableView(frame: CGRect(x: 0, y: 0, width: 100, height: 100))  // the Cast or Chronicle list
        content.addSubview(view)
        content.addSubview(list)
        window.contentView = content
        #expect(window.makeFirstResponder(list), "the focus went to the list")
        // The monitor of a view in a window hands the key to the model whoever the first responder is...
        let space = try key(" ")
        #expect(view.handleKey(space))
        #expect(await eventually { rig.sent.last == "run" })
        // ...and a chord or an unknown key is not the world's.
        let chord = try key(".", flags: .command)
        let unknown = try key("q")
        #expect(!view.handleKey(chord) && !view.handleKey(unknown))
        // Text being edited keeps its keys.
        let field = NSTextView(frame: CGRect(x: 0, y: 0, width: 100, height: 20))
        content.addSubview(field)
        #expect(window.makeFirstResponder(field))
        let typed = try key(" ")
        #expect(!view.handleKey(typed))
        #expect(rig.sent == ["run"], "\(rig.sent)")
    }

    @Test func theViewAcceptsKeysAndTheFirstClick() {
        let view = WorldMTKView(frame: .zero, device: nil)
        #expect(view.acceptsFirstResponder && view.acceptsFirstMouse(for: nil))
        view.updateTrackingAreas()
        #expect(view.trackingAreas.contains { $0.options.contains(.mouseMoved) }, "hover needs a tracking area")
        view.updateTrackingAreas()
        #expect(view.trackingAreas.filter { $0.options.contains(.mouseMoved) }.count == 1, "replaced, not piled up")
    }
}

/// The overlay and the panels, drawn offscreen with `ImageRenderer`: they build and draw from the fixture's records without
/// trapping, and the overlay draws more where something is selected or hovered. (A `List` does not render offscreen, so the
/// tabs are checked for building and for their model; their rows are pure types tested in `AlifeCoreTests`.)
@MainActor
@Suite struct WorldPanelsTests {
    private let uid0: Int64 = 5_000_000_000

    /// Pixels with any alpha in an image of the view.
    private func inked<V: View>(_ view: V, width: CGFloat = 900, height: CGFloat = 700) throws -> Int {
        let renderer = ImageRenderer(content: view.frame(width: width, height: height))
        renderer.scale = 1
        guard let image = renderer.cgImage, image.width > 0, image.height > 0 else { return 0 }  // (nothing in it: no image)
        var data = [UInt8](repeating: 0, count: image.width * image.height * 4)
        let context = try #require(CGContext(data: &data, width: image.width, height: image.height, bitsPerComponent: 8,
                                             bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                             bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return stride(from: 3, to: data.count, by: 4).reduce(0) { $0 + (data[$1] > 0 ? 1 : 0) }
    }

    private func overlay(_ rig: Rig) -> some View {
        WorldOverlayView(model: rig.model, overlay: rig.model.overlay, play: rig.play)
    }

    @Test func theOverlayDrawsTheStormsAndMoreWhenSomethingIsSelected() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        let base = try inked(overlay(rig))
        #expect(base > 200, "the rings, names and pools of the fixture: \(base) pixels")
        rig.model.edit { $0.target = SIMD3(40, 5, 9) }  // the selected organism in the middle of the view
        let moved = try inked(overlay(rig))
        rig.model.setSelection(.organism(uid: uid0 + 147))
        let ringed = try inked(overlay(rig))
        #expect(ringed > moved - 5, "(the ring is in the view)")
        #expect(ringed > moved + 100, "the selection ring")
        rig.model.edit { $0.target = SIMD3(24, 24, 0) }
        rig.model.setSelection(.storm(id: 7))
        let storm = try inked(overlay(rig))
        #expect(storm > base, "the selected storm's outer ring")
        rig.model.setSelection(.kin(id: 17))
        let kin = try inked(overlay(rig))
        #expect(kin > base + 1000, "the group's territory cells are filled: \(kin) against \(base)")
    }

    @Test func hoveringAndTheCastHoverAddTheirMarks() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        let base = try inked(overlay(rig))
        let frame = try #require(rig.play.store.frame)
        let at = try #require(rig.model.projector(size: CGSize(width: 900, height: 700)).screen(frame.position(6)))
        rig.model.hover(at: at)
        #expect(try inked(overlay(rig)) > base + 300, "the tooltip and the outline")
        rig.model.hover(at: nil)
        #expect(try inked(overlay(rig)) == base)
        rig.model.overlay.setCastHover(.kin(id: 21))
        #expect(try inked(overlay(rig)) > base + 500, "a cast row hovered shows its territory")
        rig.model.overlay.setCastHover(.storm(id: 9))
        #expect(try inked(overlay(rig)) > base)
    }

    @Test func theOverlayHandlesAViewWithNothingInIt() throws {
        let play = PlayController()
        let model = WorldModel(play: play)
        #expect(try inked(WorldOverlayView(model: model, overlay: model.overlay, play: play), width: 300, height: 200) == 0)
        model.setSelection(.organism(uid: 5))  // selected before any frame: no ring, no crash
        model.setSelection(.kin(id: 1))
        model.setSelection(.pool(index: 3))
        #expect(try inked(WorldOverlayView(model: model, overlay: model.overlay, play: play), width: 300, height: 200) == 0)
    }

    @Test func thePanelsBuildAndTheInspectorAndToastsDraw() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let rig = try await makeRig(scratch)
        defer { rig.play.shutdown() }
        #expect(try inked(CastView(play: rig.play, model: rig.model), width: 360, height: 500) >= 0)
        #expect(try inked(ChronicleView(play: rig.play, model: rig.model), width: 360, height: 500) >= 0)
        rig.model.setSelection(.organism(uid: uid0 + 100))
        let card = try #require(rig.model.inspectorCard)
        #expect(try inked(InspectorCardView(card: card, model: rig.model), width: 360, height: 300) > 1000)
        let toast = WorldEvent(id: 1, step: 1, kind: EventKind.stormDissolved, fam: "storm", a: 7, b: 0, x: nil, y: nil, z: nil, imp: 2, text: "Storm Merrow dissolved")
        #expect(try inked(ToastStackView(model: rig.model), width: 600, height: 200) == 0, "no toasts: nothing")
        rig.model.handle(events: [toast], backlog: false)
        #expect(try inked(ToastStackView(model: rig.model), width: 600, height: 200) > 500)
        rig.model.followSelection()
        #expect(try inked(FollowBadge(model: rig.model, play: rig.play), width: 400, height: 60) > 100)
        rig.model.cancelCameraMotion()
        #expect(FollowBadge.label(for: rig.model.followTarget, store: rig.play.store) == nil)
    }

    @Test func theFollowBadgeSaysWhatIsFollowed() throws {
        let store = try PlayFixtureStore.store("3d")
        #expect(FollowBadge.label(for: .none, store: store) == nil)
        #expect(FollowBadge.label(for: .organism(uid: 42), store: store) == "Following organism 42")
        #expect(FollowBadge.label(for: .storm(id: 7), store: store) == "Following Merrow")
        #expect(FollowBadge.label(for: .storm(id: 9), store: store) == "Following disturbance 9")
        #expect(FollowBadge.label(for: .storm(id: 99), store: store) == "Following storm 99")
        #expect(FollowBadge.label(for: .lineage(kinID: 17), store: store) == "Following Aldric kin")
        #expect(FollowBadge.label(for: .lineage(kinID: 99), store: store) == "Following kin group 99")
    }

    @Test func theRowsOfTheCastDraw() throws {
        let store = try PlayFixtureStore.store("3d")
        let storms = CastPolicy.storms(store.storms, baseRate: 14.4)
        #expect(try inked(StormRow(storm: storms[0], selected: true, following: true), width: 340, height: 60) > 300)
        let lineage = try #require(CastPolicy.lineages(store.identities, territory: store.territory).first)
        #expect(try inked(LineageRow(lineage: lineage, names: store.play?.labels.arch ?? [], selected: false), width: 340, height: 90) > 300)
        #expect(try inked(Bar(fraction: 0.5).frame(height: 6), width: 100, height: 6) > 100)
        #expect(try inked(ArchetypeBar(mix: [0.5, 0.25, 0, 0, 0.25, 0, 0]).frame(height: 6), width: 100, height: 6) > 100)
    }
}
