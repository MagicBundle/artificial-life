import AppKit
import Foundation
import SwiftUI
import Testing

@testable import AlifeControl
@testable import AlifeCore

/// The size monitor and the save switch of the World window (DESIGN §34.10): what is measured and when, the growth, the switch
/// sending its line and showing the state the world echoes, and the strip's room in the dock.
@MainActor
@Suite struct StorageMonitorTests {
    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var n = 0
        func next() -> Int { lock.lock(); defer { lock.unlock() }; n += 1; return n }
        var count: Int { lock.lock(); defer { lock.unlock() }; return n }
    }

    private func clock(saving: Bool = false, chapter: UInt32 = 1, flags: UInt16 = 0) -> PlayClock {
        PlayClock(head: RecordHead(step: 1, seq: 0, version: 1, flags: 0), rate: 1, achieved: 14, stepMs: 1, frameMs: 1,
                  lateSteps: 0, saveStallSteps: 0, reason: .running, saving: saving, flags: flags, fe: 1, chapter: chapter,
                  inChapter: 1, chapterLen: 6000)
    }

    /// A monitor whose measurements are 1 GB, 2 GB, ... of saves, counted.
    private func monitor(counter: Counter = Counter()) -> StorageMonitor {
        let monitor = StorageMonitor()
        monitor.period = .seconds(3600)  // only the first measurement and the ones a save asks for happen
        monitor.measure = { _ in
            StorageSample(savesBytes: Int64(counter.next()) * 1_000_000_000, logsBytes: 1_000_000, saveCount: 2,
                          newestSaveBytes: 686_000_000, freeBytes: 400_000_000_000)
        }
        return monitor
    }

    // MARK: When it measures

    @Test func itMeasuresAtOnceAndAgainWhenASaveEnds() async throws {
        let counter = Counter()
        let monitor = monitor(counter: counter)
        monitor.start(world: URL(fileURLWithPath: "/tmp/alife-no-such-world"))
        defer { monitor.stop() }
        #expect(await eventually { monitor.sample != nil })
        #expect(monitor.sample?.savesBytes == 1_000_000_000)
        monitor.note(clock: clock(saving: false))
        monitor.note(clock: clock(saving: true))  // a save starts: nothing yet
        try await Task.sleep(for: .milliseconds(100))
        #expect(counter.count == 1)
        monitor.note(clock: clock(saving: false))  // and ends: the new file is in place and the old ones are pruned
        #expect(await eventually { monitor.sample?.savesBytes == 2_000_000_000 })
        monitor.note(clock: clock(saving: false))  // an ordinary clock measures nothing
        try await Task.sleep(for: .milliseconds(100))
        #expect(counter.count == 2)
    }

    @Test func itMeasuresEveryPeriodOnItsOwn() async throws {
        let counter = Counter()
        let monitor = monitor(counter: counter)
        monitor.period = .milliseconds(20)
        monitor.start(world: URL(fileURLWithPath: "/tmp/alife-no-such-world"))
        defer { monitor.stop() }
        #expect(await eventually { counter.count >= 4 })
    }

    @Test func theGrowthCountsFromTheFirstSampleAndFromTheChapter() async throws {
        let monitor = monitor()
        monitor.start(world: URL(fileURLWithPath: "/tmp/alife-no-such-world"))
        defer { monitor.stop() }
        #expect(await eventually { monitor.sample != nil })
        monitor.note(clock: clock(chapter: 4))
        await monitor.refresh()  // 2 GB + 1 MB
        await monitor.refresh()  // 3 GB + 1 MB
        let readout = monitor.readout(mode: .all)
        #expect(readout.sinceStart == "+2.0 GB")
        #expect(readout.sinceChapter == "+1.0 GB", "the chapter began at the first sample that knew it")
    }

    @Test func aNewWorldForgetsTheOldOnesSampleAndGrowth() async throws {
        let monitor = monitor()
        monitor.start(world: URL(fileURLWithPath: "/tmp/alife-no-such-world"))
        defer { monitor.stop() }
        #expect(await eventually { monitor.sample != nil })
        monitor.stop()
        #expect(monitor.sample == nil && monitor.world == nil && monitor.trend == StorageTrend())
        await monitor.refresh()  // nothing to measure
        #expect(monitor.sample == nil)
    }

    @Test func aMeasurementThatFinishesAfterTheWorldWasReplacedIsDropped() async throws {
        let monitor = StorageMonitor()
        monitor.period = .seconds(3600)
        let gate = DispatchSemaphore(value: 0)
        monitor.measure = { url in
            if url.lastPathComponent == "slow" { gate.wait() }
            return StorageSample(savesBytes: url.lastPathComponent == "slow" ? 1 : 2)
        }
        monitor.start(world: URL(fileURLWithPath: "/tmp/slow"))
        try await Task.sleep(for: .milliseconds(50))
        monitor.start(world: URL(fileURLWithPath: "/tmp/fast"))
        #expect(await eventually { monitor.sample?.savesBytes == 2 })
        gate.signal()
        try await Task.sleep(for: .milliseconds(100))
        #expect(monitor.sample?.savesBytes == 2, "the first world's late answer does not replace the second's")
        monitor.stop()
    }

    @Test func itMeasuresARealDirectory() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let saves = scratch.url.appendingPathComponent("checkpoints")
        try FileManager.default.createDirectory(at: saves, withIntermediateDirectories: true)
        try Data(count: 200_000).write(to: saves.appendingPathComponent("step_0000000010.safetensors"))
        try Data(count: 50_000).write(to: scratch.url.appendingPathComponent("metrics.jsonl"))
        let monitor = StorageMonitor()
        monitor.period = .seconds(3600)
        monitor.start(world: scratch.url)
        defer { monitor.stop() }
        #expect(await eventually { monitor.sample != nil })
        let sample = try #require(monitor.sample)
        #expect(sample.saveCount == 1 && sample.savesBytes >= 200_000 && sample.logsBytes >= 50_000)
    }

    // MARK: The switch

    private func rig(_ scratch: Scratch) async throws -> (play: PlayController, model: WorldModel, commands: URL) {
        let commands = scratch.url.appendingPathComponent("commands.txt")
        let script = try scratch.script("alife-live", "cat \"$1\"\nwhile IFS= read -r line; do echo \"$line\" >> \"$2\"; done")
        let play = PlayController()
        let model = WorldModel(play: play, defaults: try #require(UserDefaults(suiteName: "alife-storage-\(UUID().uuidString)")))
        play.start(.resume(PlayResume(run: "r")), executable: script, directory: scratch.url,
                   arguments: [Repo.playFixtureURL("3d_g3b").path, commands.path])
        #expect(await eventually { play.store.header != nil && play.store.clock?.reason == .alert })
        return (play, model, commands)
    }

    @Test func theSwitchSendsItsLineAndShowsWhatTheWorldEchoes() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (play, _, commands) = try await rig(scratch)
        defer { play.shutdown() }
        var now = 100.0
        play.now = { now }
        // the fixture is a world that keeps the newest save only: its header and its clock say so
        #expect(play.store.play?.saves == PlayHeader.Saves(mode: "newest", keep: 3))
        #expect(play.savesMode == .newest && play.requestedSaves == nil)
        #expect(play.send(.saves(.all)))
        #expect(play.requestedSaves == .all && play.savesMode == .all, "the switch moves at once")
        let sent = await eventually { ((try? String(contentsOf: commands, encoding: .utf8)) ?? "").contains("saves all\n") }
        #expect(sent)
        var echoed = try #require(play.store.clock)
        play.settleRequest(after: echoed)  // a clock sent before the world read the line still shows the old mode
        #expect(play.requestedSaves == .all && play.savesMode == .all)
        echoed.flags &= ~4  // the world's answer
        play.settleRequest(after: echoed)
        #expect(play.requestedSaves == nil)
        // a world that never confirms (an older server): the request is dropped after a second and the switch goes back
        #expect(play.send(.saves(.newest)))
        #expect(play.savesMode == .newest)
        echoed.flags &= ~4
        now += 1.5
        play.settleRequest(after: echoed)
        #expect(play.requestedSaves == nil && play.savesMode == .newest, "the clock in the store still says newest here")
        play.stop()
        #expect(play.requestedSaves == nil)
    }

    @Test func theWorldIsWhereTheMonitorLooks() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (play, model, _) = try await rig(scratch)
        defer { play.shutdown() }
        model.storage.period = .seconds(3600)
        #expect(play.worldURL == scratch.url.appendingPathComponent("runs/r", isDirectory: true))
        #expect(model.storage.world == play.worldURL, "the model began to watch the world when its header arrived")
        model.storage.measure = { _ in StorageSample(savesBytes: 3, saveCount: 3) }
        let started = model.storage.world
        model.startStorage()  // a view that attaches to a world in play does not restart the count
        #expect(model.storage.world == started)
        play.stop()  // the old world is forgotten at once
        #expect(model.storage.world == nil && model.storage.sample == nil)
    }

    @Test func theStripDrawsAndShowsTheSecondLineOnlyWhileOlderSavesAreWaiting() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (play, model, _) = try await rig(scratch)
        defer { play.shutdown() }
        model.storage.measure = { _ in StorageSample(savesBytes: 2_000_000_000, saveCount: 4, newestSaveBytes: 686_000_000, freeBytes: 500_000_000_000) }
        // (the model began to watch the world when its header arrived)
        #expect(model.storage.world == play.worldURL)
        await model.storage.refresh()
        #expect(play.savesMode == .newest, "mode \(play.savesMode) requested \(String(describing: play.requestedSaves)) flags \(String(describing: play.store.clock?.flags))")
        #expect(model.storage.sample?.saveCount == 4, "sample \(String(describing: model.storage.sample))")
        #expect(model.storageNoteShown, "four saves are on disk and only the newest is kept")
        let tall = height(StorageStripView(play: play, model: model).frame(width: 320), width: 320)
        play.send(.saves(.all))
        #expect(!model.storageNoteShown)
        let plain = height(StorageStripView(play: play, model: model).frame(width: 320), width: 320)
        print("measured: storage strip \(plain), with its second line \(tall)")
        #expect(plain <= DockBudget.storage && tall <= DockBudget.storage + DockBudget.storageNote)
        #expect(tall > plain)
        #expect(StorageWords.label(.all) == "All saves" && StorageWords.label(.newest) == "Newest only")
        #expect(StorageWords.help == "Recording off keeps only the newest save; your world stays resumable; logs are still written.")
        #expect(StorageWords.tooltip(.newest, keep: 3).contains(StorageWords.pending))
        model.storage.stop()
    }

    private func height<V: View>(_ view: V, width: CGFloat) -> CGFloat {
        let host = NSHostingView(rootView: view.frame(width: width))
        host.layoutSubtreeIfNeeded()
        return host.fittingSize.height
    }
}
