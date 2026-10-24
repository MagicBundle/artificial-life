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

    private func clock(saving: Bool = false, chapter: UInt32 = 1, flags: UInt16 = 0, reason: PlayClock.Reason = .running,
                       rate: Float = 1) -> PlayClock {
        PlayClock(head: RecordHead(step: 1, seq: 0, version: 1, flags: 0), rate: rate, achieved: 14, stepMs: 1, frameMs: 1,
                  lateSteps: 0, saveStallSteps: 0, reason: reason, saving: saving, flags: flags, fe: 1, chapter: chapter,
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
        // the third choice: the world echoes it as bit 3 with bit 2 (DESIGN 34.11)
        #expect(play.send(.saves(.single)))
        #expect(play.requestedSaves == .single && play.savesMode == .single)
        echoed.flags |= 4 | 8
        play.settleRequest(after: echoed)
        #expect(play.requestedSaves == nil && SaveModeView.current(clock: echoed, header: nil, requested: nil) == .single)
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

    @Test func theStripHasNoSecondLineAndTheSwitchIsNamedForRecording() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (play, model, _) = try await rig(scratch)
        defer { play.shutdown() }
        model.storage.measure = { _ in StorageSample(savesBytes: 2_000_000_000, saveCount: 4, newestSaveBytes: 686_000_000, freeBytes: 500_000_000_000) }
        #expect(model.storage.world == play.worldURL)
        await model.storage.refresh()
        #expect(play.savesMode == .newest)
        #expect(model.storage.readout(mode: .newest).pruningPending, "four saves are on disk and only the newest is kept")
        let waiting = height(StorageStripView(play: play, model: model).frame(width: 320), width: 320)
        play.send(.saves(.all))
        let plain = height(StorageStripView(play: play, model: model).frame(width: 320), width: 320)
        print("measured: storage strip \(plain) (\(waiting) while older saves wait)")
        #expect(plain <= DockBudget.storage && waiting == plain, "the dock's budget is the same whichever mode and saves")
        #expect(StorageWords.label == "Saves kept")
        #expect(SaveMode.allCases.map(StorageWords.segment) == ["All", "Start", "Newest"])
        #expect(SaveMode.allCases.map(StorageWords.name) == ["All saves", "Newest + start", "Newest only"])
        #expect(StorageWords.resumable == "Your world stays resumable; logs are still written.")
        #expect(StorageWords.tooltip(.newest, keep: 3).contains(StorageWords.pending))
        #expect(StorageWords.tooltip(.single, keep: 3).contains(StorageWords.pending))
        #expect(StorageWords.tooltip(.all, keep: 3).hasPrefix("All saves: the last 3 saves and the origin"))
        #expect(StorageWords.keeps(.newest, keep: 3).hasPrefix("Newest + start: the newest save and the origin."))
        #expect(StorageWords.tooltip(.all, keep: 3).contains("deletes the older saves at the next save"))
        #expect(StorageWords.tooltip(.single, keep: 3).contains(StorageWords.resumable), "the world stays resumable in every mode")
        model.storage.stop()
    }

    @Test func theStripFitsTheNarrowestDock() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (play, model, _) = try await rig(scratch)
        defer { play.shutdown() }
        let room = DockLayout.minExtent - 20  // the dock's least width less its 10-point gutters
        let samples = [StorageSample(savesBytes: 2_100_000_000, saveCount: 3, newestSaveBytes: 686_000_000, freeBytes: 400_000_000_000),
                       StorageSample(savesBytes: 2_100_000_000, saveCount: 3, newestSaveBytes: 686_000_000, freeBytes: 1_500_000_000),
                       StorageSample(savesBytes: 999_000_000_000, saveCount: 3, newestSaveBytes: 686_000_000, freeBytes: 400_000_000)]
        var widths: [CGFloat] = []
        for sample in samples {
            model.storage.measure = { _ in sample }
            await model.storage.refresh()
            widths.append(idealWidth(StorageStripView(play: play, model: model)))
        }
        model.storage.stop()  // no sample: "measuring…"
        widths.append(idealWidth(StorageStripView(play: play, model: model)))
        print("measured: storage strip ideal widths \(widths) of \(room)")
        #expect(widths.allSatisfy { $0 <= room }, "the strip wants \(widths) of the \(room) points of the narrowest dock")
    }

    @Test func theBottomDocksClockColumnFitsItsLeastHeight() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (play, model, _) = try await rig(scratch)
        defer { play.shutdown() }
        model.storage.measure = { _ in StorageSample(savesBytes: 2_100_000_000, saveCount: 3, newestSaveBytes: 686_000_000, freeBytes: 1_500_000_000) }
        await model.storage.refresh()
        // the column of the bottom dock, as `ControlCenterView.bottomDock` builds it, at its width, whatever the clock says
        var most: CGFloat = 0
        for reason: PlayClock.Reason in [.running, .userPause, .endOfSteps, .chapter, .alert, .error] {
            for saving in [false, true] {
                play.store.apply(.clck(clock(saving: saving, chapter: 2, flags: 1, reason: reason, rate: reason == .running ? 1 : 0)))
                let column = height(VStack(spacing: 0) {
                    ClockStripView(play: play, model: model).padding(.horizontal, 10).padding(.top, 10).padding(.bottom, 4)
                    StorageStripView(play: play, model: model).padding(.horizontal, 10).padding(.bottom, 6)
                }, width: ControlCenterView.clockColumn)
                most = max(most, column)
            }
        }
        let needed = DockBudget.header + 1 + most  // (the dock's header and its divider are above the columns)
        print("measured: bottom dock clock column at most \(most), with the header \(needed), of the least \(DockLayout.minBottomExtent)")
        #expect(needed <= DockLayout.minBottomExtent, "the recording switch would be clipped in a dock of \(DockLayout.minBottomExtent): it needs \(needed)")
        #expect(DockBudget.header + 1 + DockBudget.tabs + ControlCenterView.listMinHeight <= DockLayout.minBottomExtent, "the tabs column fits too")
        model.storage.stop()
    }

    @Test func aStaleMeasurementDoesNotReplaceANewerOne() async throws {
        let monitor = StorageMonitor()
        monitor.period = .seconds(3600)
        let gate = DispatchSemaphore(value: 0)
        let counter = Counter()
        monitor.measure = { _ in
            let n = counter.next()
            if n == 2 { gate.wait() }  // the second request is slow: it was asked for before the third
            return StorageSample(savesBytes: Int64(n))
        }
        monitor.start(world: URL(fileURLWithPath: "/tmp/alife-no-such-world"))
        defer { monitor.stop() }
        #expect(await eventually { monitor.sample?.savesBytes == 1 })
        let slow = Task { await monitor.refresh() }  // asks 2, waits
        try await Task.sleep(for: .milliseconds(50))
        await monitor.refresh()  // asks 3, answers at once
        #expect(monitor.sample?.savesBytes == 3)
        gate.signal()
        await slow.value
        #expect(monitor.sample?.savesBytes == 3, "the older answer is dropped")
    }

    @Test func aScanAskedDuringASaveIsNotTheChaptersBaseline() async throws {
        let monitor = StorageMonitor()
        monitor.period = .seconds(3600)
        let midSave = DispatchSemaphore(value: 0)
        let saveEnd = DispatchSemaphore(value: 0)
        let counter = Counter()
        monitor.measure = { _ in
            switch counter.next() {
            case 2: midSave.wait(); return StorageSample(savesBytes: 2_350)  // asked during the save, answered after it
            case 3: saveEnd.wait(); return StorageSample(savesBytes: 2_686)  // the scan the end of the save asks for
            default: return StorageSample(savesBytes: 2_000)
            }
        }
        monitor.start(world: URL(fileURLWithPath: "/tmp/alife-no-such-world"))
        defer { monitor.stop() }
        #expect(await eventually { monitor.sample?.savesBytes == 2_000 })
        monitor.note(clock: clock(saving: false, chapter: 3))
        monitor.note(clock: clock(saving: true, chapter: 4))  // a chapter end: the new chapter's number while its save is written
        let slow = Task { await monitor.refresh() }
        try await Task.sleep(for: .milliseconds(50))
        monitor.note(clock: clock(saving: false, chapter: 4))  // the save ends: a scan is asked for
        try await Task.sleep(for: .milliseconds(50))
        midSave.signal()
        await slow.value
        #expect(monitor.sample?.savesBytes == 2_350)
        #expect(monitor.trend.chapter == 4 && monitor.trend.chapterStart == nil, "a mid-save total is no baseline: \(monitor.trend)")
        saveEnd.signal()
        #expect(await eventually { monitor.sample?.savesBytes == 2_686 })
        #expect(monitor.trend.chapterStart == 2_686, "the chapter counts from its first sample after the save that closed the last")
    }

    @Test func aChoiceSendsItsLineOnlyWhenItIsNewAndOffered() {
        let world = PlayHeader.Saves(mode: "all", keep: 3)
        let adopted = PlayHeader.Saves(mode: "all", keep: 8, adopted: true)
        #expect(StorageStripView.command(choosing: .single, current: .all, header: world) == .saves(.single))
        #expect(StorageStripView.command(choosing: .newest, current: .single, header: world) == .saves(.newest))
        #expect(StorageStripView.command(choosing: .all, current: .all, header: world) == nil, "the mode in force sends nothing")
        #expect(StorageStripView.command(choosing: .single, current: .all, header: adopted) == nil, "a research run keeps all saves")
        #expect(StorageStripView.command(choosing: .all, current: .all, header: adopted) == nil)
        #expect(StorageStripView.command(choosing: .newest, current: .all, header: nil) == nil, "an older server")
    }

    @Test func aResearchRunPlayedInPlaceOffersOnlyAllSaves() {
        let adopted = PlayHeader.Saves(mode: "all", keep: 8, adopted: true)
        let world = PlayHeader.Saves(mode: "all", keep: 3)
        #expect(StorageWords.choiceHelp(.all, header: adopted) == StorageWords.adopted)
        #expect(StorageWords.choiceHelp(.all, header: world) == StorageWords.tooltip(.all, keep: 3))
        #expect(StorageWords.choiceHelp(.single, header: world) == StorageWords.tooltip(.single, keep: 3), "the smaller choices are offered")
        #expect(StorageWords.choiceHelp(.all, header: nil) == StorageWords.unknown)
        #expect(StorageWords.adopted.contains("alife play RUN --saves newest"))
    }

    @Test func theLoopStopsWhenTheWorldIsNoLongerLive() async throws {
        let counter = Counter()
        let monitor = monitor(counter: counter)
        monitor.period = .milliseconds(20)
        final class Flag { var on: Bool; init(_ on: Bool) { self.on = on } }
        let live = Flag(true)
        monitor.isLive = { live.on }
        monitor.start(world: URL(fileURLWithPath: "/tmp/alife-no-such-world"))
        defer { monitor.stop() }
        #expect(await eventually { counter.count >= 3 })
        let loop = try #require(monitor.loop)
        let before = counter.count
        live.on = false  // the world ended
        // The loop is awaited, not timed: it awaits each of its scans, so one that a loaded scheduler delays lands before the end.
        let ended = Flag(false)
        Task { await loop.value; ended.on = true }
        try #require(await eventually { ended.on }, "the loop ends of itself (`stop` is only in the defer)")
        let settled = counter.count
        #expect(settled - before <= 1, "at most the scan in flight or the one after the sleep: \(before) then \(settled)")
        try await Task.sleep(for: monitor.period * 5)  // nothing is left to scan; this only catches a scan the loop did not await
        #expect(counter.count == settled, "no scan after the end: \(settled) then \(counter.count)")
        #expect(monitor.sample != nil, "the last size stays on show")
    }

    private func idealWidth<V: View>(_ view: V) -> CGFloat {
        let host = NSHostingView(rootView: view)
        host.layoutSubtreeIfNeeded()
        return host.fittingSize.width
    }

    private func height<V: View>(_ view: V, width: CGFloat) -> CGFloat {
        let host = NSHostingView(rootView: view.frame(width: width))
        host.layoutSubtreeIfNeeded()
        return host.fittingSize.height
    }
}

@Suite struct StorageChoiceTests {
    @Test func onlyTheChoicesThatDeleteOlderSavesAreConfirmed() {
        #expect(!StorageWords.deletesOlder(.all))
        #expect(StorageWords.deletesOlder(.newest))
        #expect(StorageWords.deletesOlder(.single))
        #expect(StorageWords.confirmMessage(.single, keep: 3).contains("cannot be restored"))
        #expect(StorageWords.confirmTitle(.newest) == "Keep newest + start?")
    }
}
