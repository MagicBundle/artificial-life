import Foundation
import Testing

@testable import AlifeCore

private final class StopFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return flag }
    func set() { lock.lock(); flag = true; lock.unlock() }
}

/// The size monitor's pure parts (DESIGN §34.10): the measurement of a world directory, the formatting, the warning, the growth and
/// the save mode the window shows. The measurement runs on a temporary directory with checkpoints and logs.
@Suite struct StorageTests {
    // MARK: The measurement

    private func world(_ files: [String: Int]) throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("alife-storage-\(UUID().uuidString)")
        for (path, size) in files {
            let url = root.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data((0..<size).map { UInt8(truncatingIfNeeded: $0 &* 31 &+ $0 >> 8) }).write(to: url)
        }
        return root
    }

    /// Allocated blocks are a little more than the bytes written: a file of `n` bytes takes between `n` and `n` + 2 blocks.
    private func near(_ measured: Int64, _ written: Int, files: Int = 1) -> Bool {
        measured >= Int64(written) && measured <= Int64(written) + Int64(files) * 2 * 65_536
    }

    @Test func savesAreTheCheckpointsDirectoryAndLogsAreEverythingElse() throws {
        let root = try world([
            "checkpoints/step_0000000000.safetensors": 300_000,
            "checkpoints/step_0000000030.safetensors": 500_000,
            "checkpoints/step_0000000031.safetensors.tmp": 70_000,  // a save being written: space on disk, not a save
            "checkpoints/LATEST": 30,
            "events/births_0000000000_0000000030.npz": 90_000,
            "observers/step_0000000030.npz": 40_000,
            "metrics.jsonl": 20_000,
            "director.jsonl": 5_000,
            "commands.jsonl": 100,
        ])
        defer { try? FileManager.default.removeItem(at: root) }
        let sample = StorageScan.measure(world: root)
        #expect(near(sample.savesBytes, 870_030, files: 4), "saves \(sample.savesBytes)")
        #expect(near(sample.logsBytes, 155_100, files: 5), "logs \(sample.logsBytes)")
        #expect(sample.totalBytes == sample.savesBytes + sample.logsBytes)
        #expect(sample.saveCount == 2, "the temporary file is no save")
        #expect(near(sample.newestSaveBytes, 500_000), "the newest save is the one of step 30: \(sample.newestSaveBytes)")
        let free = try #require(sample.freeBytes)
        #expect(free > 0)
    }

    @Test func aWorldReachedThroughALinkIsMeasuredAndAnEmptyOrMissingOneIsZero() throws {
        let root = try world(["checkpoints/step_0000000007.safetensors": 100_000, "metrics.jsonl": 1_000])
        let link = FileManager.default.temporaryDirectory.appendingPathComponent("alife-storage-link-\(UUID().uuidString)")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: root)
        defer {
            try? FileManager.default.removeItem(at: link)
            try? FileManager.default.removeItem(at: root)
        }
        var viaLink = StorageScan.measure(world: link), direct = StorageScan.measure(world: root)
        #expect(viaLink.freeBytes != nil)
        viaLink.freeBytes = nil  // (what the volume has free moves with every other process)
        direct.freeBytes = nil
        #expect(viaLink == direct)
        #expect(StorageScan.measure(world: link).saveCount == 1)
        let missing = root.appendingPathComponent("no-such-world")
        let none = StorageScan.measure(world: missing)
        #expect(none.totalBytes == 0 && none.saveCount == 0 && none.newestSaveBytes == 0)
        let bare = try world(["metrics.jsonl": 10])  // a world with no checkpoints directory
        defer { try? FileManager.default.removeItem(at: bare) }
        let sample = StorageScan.measure(world: bare)
        #expect(sample.savesBytes == 0 && sample.logsBytes > 0 && sample.saveCount == 0)
    }

    @Test func aSaveThatVanishesWhileTheDirectoryIsWalkedIsSkippedNotAnError() throws {
        let root = try world(["checkpoints/step_0000000001.safetensors": 10_000, "checkpoints/step_0000000002.safetensors": 10_000])
        defer { try? FileManager.default.removeItem(at: root) }
        // The walk is repeated while a thread creates and removes files, as a save does (a new file, then the pruning of an old one).
        let done = DispatchSemaphore(value: 0)
        let stop = StopFlag()
        let churn = Thread {
            var n = 0
            while !stop.isSet {
                let url = root.appendingPathComponent("checkpoints/step_\(String(format: "%010d", 100 + n % 7)).safetensors")
                try? Data(count: 4_096).write(to: url)
                try? FileManager.default.removeItem(at: url)
                n += 1
            }
            done.signal()
        }
        churn.start()
        for _ in 0..<200 {
            let sample = StorageScan.measure(world: root)
            #expect(sample.saveCount >= 2 && sample.savesBytes >= 20_000)
        }
        stop.set()
        done.wait()
    }

    @Test func aSaveIsRecognisedByItsName() {
        #expect(StorageScan.saveStep(of: "step_0000504001.safetensors") == 504_001)
        #expect(StorageScan.saveStep(of: "step_0000504001.safetensors.tmp") == nil)
        #expect(StorageScan.saveStep(of: "step_0000504001.safetensors.zst") == 504_001)
        #expect(StorageScan.saveStep(of: "step_0000504001.safetensors.zst.tmp") == nil)
        #expect(StorageScan.saveStep(of: "LATEST") == nil && StorageScan.saveStep(of: "step_x.safetensors") == nil)
    }

    // MARK: Text

    @Test(arguments: [
        (Int64(0), "0 B"), (999, "999 B"), (1_000, "1.0 kB"), (1_536, "1.5 kB"), (12_345, "12 kB"), (999_400, "999 kB"),
        (999_600, "1.0 MB"), (686_000_000, "686 MB"), (9_949_000, "9.9 MB"), (9_960_000, "10 MB"),
        (2_600_000_000, "2.6 GB"), (3_000_000_000, "3.0 GB"), (412_000_000_000, "412 GB"), (1_500_000_000_000, "1.5 TB"),
        (-5, "0 B"),
    ])
    func sizesAreInDecimalUnitsWithOneDecimalUnderTen(count: Int64, text: String) {
        #expect(StorageFormat.bytes(count) == text)
    }

    @Test func aChangeCarriesItsSign() {
        #expect(StorageFormat.signed(180_000_000) == "+180 MB")
        #expect(StorageFormat.signed(-1_400_000_000) == "-1.4 GB")
        #expect(StorageFormat.signed(0) == "+0 B")
    }

    // MARK: The warning

    @Test func theWarningIsInSavesOfRoom() {
        let save: Int64 = 686_000_000
        #expect(StorageHealth.level(free: nil, saveBytes: save) == .ok, "an unknown figure is no warning")
        #expect(StorageHealth.level(free: 3 * save, saveBytes: save) == .ok)
        #expect(StorageHealth.level(free: 3 * save - 1, saveBytes: save) == .low)
        #expect(StorageHealth.level(free: save * 5 / 4, saveBytes: save) == .low)
        #expect(StorageHealth.level(free: save * 5 / 4 - 1, saveBytes: save) == .critical)
        #expect(StorageHealth.level(free: 0, saveBytes: save) == .critical)
        // no save yet: a typical one stands for it
        #expect(StorageHealth.level(free: 2_100_000_000, saveBytes: 0) == .ok)
        #expect(StorageHealth.level(free: 2_100_000_000 - 1, saveBytes: 0) == .low)
        #expect(StorageHealth.ok < .low && StorageHealth.low < .critical)
    }

    // MARK: The growth

    @Test func growthIsSinceTheWorldWentLiveAndSinceTheChapterBegan() {
        var trend = StorageTrend()
        #expect(trend.sinceStart(5) == nil && trend.sinceChapter(5) == nil)
        trend.note(total: 2_000, chapter: nil)  // (before the first clock)
        trend.note(total: 2_100, chapter: 7)
        #expect(trend.sinceStart(2_100) == 100 && trend.sinceChapter(2_100) == 0)
        trend.note(total: 2_600, chapter: 7)
        #expect(trend.sinceStart(2_600) == 600 && trend.sinceChapter(2_600) == 500)
        trend.note(total: 2_700, chapter: 8)  // a new chapter starts its own count
        #expect(trend.sinceStart(2_700) == 700 && trend.sinceChapter(2_700) == 0)
        #expect(trend.sinceStart(1_500) == -500, "a prune is a negative growth")
    }

    // MARK: The save mode

    private func clock(_ flags: UInt16) -> PlayClock {
        PlayClock(head: RecordHead(step: 1, seq: 0, version: 1, flags: 0), rate: 1, achieved: 14, stepMs: 1, frameMs: 1,
                  lateSteps: 0, saveStallSteps: 0, reason: .running, saving: false, flags: flags, fe: 1, chapter: 1,
                  inChapter: 1, chapterLen: 6000)
    }

    @Test func theModeIsWhatWasAskedThenTheClockThenTheHeader() {
        let header = PlayHeader.Saves(mode: "newest", keep: 3)
        #expect(SaveModeView.current(clock: nil, header: nil, requested: nil) == .all)
        #expect(SaveModeView.current(clock: nil, header: header, requested: nil) == .newest)
        #expect(SaveModeView.current(clock: clock(0), header: header, requested: nil) == .all, "the clock is the truth after the header")
        #expect(SaveModeView.current(clock: clock(5), header: nil, requested: nil) == .newest)
        #expect(SaveModeView.current(clock: clock(5), header: header, requested: .all) == .all, "what was just asked shows at once")
        #expect(SaveModeView.current(clock: nil, header: PlayHeader.Saves(mode: "bogus", keep: 3), requested: nil) == .all)
        #expect(clock(4).savesNewestOnly && !clock(3).savesNewestOnly && clock(4).chapterPauseArmed == false)
    }

    @Test func recordingOffIsNotOfferedInAResearchRunPlayedInPlace() throws {
        let decode = { (json: String) in try JSONDecoder().decode(PlayHeader.Saves.self, from: Data(json.utf8)) }
        let adopted = try decode(#"{"mode": "all", "keep": 8, "adopted": true}"#)
        let world = try decode(#"{"mode": "all", "keep": 3}"#)
        #expect(adopted.adopted == true && world.adopted == nil, "the key is written only for such a run")
        #expect(!SaveModeView.offers(.newest, header: adopted) && SaveModeView.offers(.all, header: adopted))
        #expect(SaveModeView.offers(.newest, header: world) && SaveModeView.offers(.all, header: world))
        #expect(!SaveModeView.offers(.newest, header: nil) && !SaveModeView.offers(.all, header: nil), "an older server")
    }

    // MARK: What the strip says

    @Test func theReadoutNamesTheSplitTheGrowthAndTheRoom() {
        var trend = StorageTrend()
        trend.note(total: 2_000_000_000, chapter: 3)
        let sample = StorageSample(savesBytes: 2_500_000_000, logsBytes: 52_000_000, saveCount: 4, newestSaveBytes: 686_000_000,
                                   freeBytes: 412_000_000_000)
        let readout = StorageReadout(sample: sample, trend: trend, mode: .all)
        #expect(readout.total == "2.6 GB" && readout.saves == "2.5 GB" && readout.logs == "52 MB" && readout.saveCount == 4)
        #expect(readout.free == "412 GB free" && readout.sinceStart == "+552 MB" && readout.sinceChapter == "+552 MB")
        #expect(readout.health == .ok && readout.warning == nil && readout.mode == .all)
        #expect(readout.pruningPending, "four saves on disk: a world that keeps the newest only would delete two")
        #expect(!StorageReadout(sample: StorageSample(saveCount: 2), mode: .newest).pruningPending, "the origin and the newest")
    }

    @Test func aFullDiskIsSaidInWords() throws {
        let low = StorageReadout(sample: StorageSample(savesBytes: 1, saveCount: 1, newestSaveBytes: 686_000_000, freeBytes: 1_500_000_000),
                                 mode: .all)
        #expect(low.health == .low && low.warning == "Low disk space: 1.5 GB free (a save is about 686 MB).")
        let critical = StorageReadout(sample: StorageSample(newestSaveBytes: 686_000_000, freeBytes: 400_000_000), mode: .all)
        #expect(critical.health == .critical)
        #expect(critical.warning == "Almost no disk space: the next save (about 686 MB) may not fit.")
        let waiting = StorageReadout(sample: nil, mode: .all)
        #expect(waiting.total == "measuring…" && waiting.warning == nil && waiting.free == nil)
    }

    // MARK: Review fixes

    @Test func aChapterCountsFromTheFirstSampleAfterItsSave() {
        var trend = StorageTrend()
        trend.note(total: 2_000, chapter: 3)
        // the chapter end: the clock carries the new chapter while its save is being written; the save belongs to the old chapter
        trend.note(total: 2_350, chapter: 4, saving: true)
        #expect(trend.sinceChapter(2_350) == nil, "no baseline mid-save")
        trend.note(total: 2_686, chapter: 4, saving: false)
        #expect(trend.sinceChapter(2_686) == 0, "the 686-byte save landed in chapter 3")
        trend.note(total: 2_700, chapter: 4, saving: false)
        #expect(trend.sinceChapter(2_700) == 14 && trend.sinceStart(2_700) == 700)
        trend.note(total: 2_900, chapter: 4, saving: true)  // a quit save inside the chapter does not move the baseline
        #expect(trend.sinceChapter(2_900) == 214)
    }

    @Test func theSavesOfALinkedCheckpointsDirectoryAreCounted() throws {
        let root = try world(["metrics.jsonl": 1_000])
        let elsewhere = try world(["step_0000000007.safetensors": 100_000, "step_0000000009.safetensors": 120_000])
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: elsewhere)
        }
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("checkpoints"), withDestinationURL: elsewhere)
        let sample = StorageScan.measure(world: root)
        #expect(sample.saveCount == 2 && near(sample.savesBytes, 220_000, files: 2), "saves \(sample.savesBytes) in \(sample.saveCount)")
        #expect(near(sample.newestSaveBytes, 120_000))
        #expect(near(sample.logsBytes, 1_000), "the link is not counted twice, as logs: \(sample.logsBytes)")
    }

    @Test func theStripShowsTheRoomLeftOnlyWhenTheDiskIsGettingFull() {
        let ok = StorageReadout(sample: StorageSample(savesBytes: 2_100_000_000, newestSaveBytes: 686_000_000, freeBytes: 400_000_000_000),
                                mode: .all)
        #expect(ok.stripText == "2.1 GB" && ok.room == nil)
        let low = StorageReadout(sample: StorageSample(savesBytes: 2_100_000_000, newestSaveBytes: 686_000_000, freeBytes: 1_500_000_000),
                                 mode: .all)
        #expect(low.stripText == "2.1 GB, 1.5 GB free" && low.room == "1.5 GB free")
        #expect(StorageReadout(sample: nil, mode: .all).stripText == "measuring…")
    }
}
