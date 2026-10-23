import AlifeCore
import Foundation

/// Watches the size of the world in play on disk (DESIGN §34.10). It measures the world's directory from file attributes (no file
/// is opened and the engine is not asked), on a background task every `period` and right after each save, and keeps the growth
/// since the world went live and since the chapter began. The measurement is an injected function so that the tests feed theirs.
@MainActor
final class StorageMonitor: ObservableObject {
    @Published private(set) var sample: StorageSample?
    @Published private(set) var trend = StorageTrend()

    /// Measures a world directory off the main actor. Tests inject theirs.
    var measure: @Sendable (URL) -> StorageSample = { StorageScan.measure(world: $0) }
    /// The time between two measurements.
    var period: Duration = .seconds(5)
    /// Whether the world is still being played. The periodic loop measures once more after it turns false and then ends: a world
    /// that ended or failed no longer grows, and a new world sends a new header, which starts a new loop.
    var isLive: @MainActor () -> Bool = { true }
    private(set) var world: URL?
    private(set) var measurements = 0
    private var chapter: UInt32?
    private var wasSaving = false
    private var loop: Task<Void, Never>?
    private var generation = 0
    /// The numbers of the measurements asked for and of the newest one whose answer was published: a slow answer that was asked
    /// for before a newer one was published is stale and dropped.
    private var asked = 0
    private var published = 0

    init() {}

    /// Starts watching `world`: forgets the old sample and trend, measures now and then every `period`.
    func start(world: URL) {
        stop()
        self.world = world
        generation += 1
        loop = Task { [weak self] in
            while !Task.isCancelled {
                await self?.refresh()
                guard let self, self.isLive() else { return }
                try? await Task.sleep(for: self.period)
            }
        }
    }

    func stop() {
        loop?.cancel()
        loop = nil
        world = nil
        generation += 1
        sample = nil
        trend = StorageTrend()
        chapter = nil
        wasSaving = false
    }

    /// Measures once, off the main actor, and publishes the result unless the world changed meanwhile or a newer measurement was
    /// published first. The growth is told the chapter and the saving state of when the measurement was asked for, which is what
    /// the directory was in: a scan begun during a save and answered after it is a mid-save total, not a chapter's baseline.
    func refresh() async {
        guard let url = world else { return }
        let world = generation
        asked += 1
        let number = asked
        let (chapter, saving) = (self.chapter, wasSaving)
        let measure = self.measure
        let fresh = await Task.detached(priority: .utility) { measure(url) }.value
        guard world == generation, number > published else { return }
        published = number
        measurements += 1
        trend.note(total: fresh.totalBytes, chapter: chapter, saving: saving)
        if fresh != sample { sample = fresh }
    }

    /// A `CLCK` arrived: the end of a save (`saving` falls) is when the new file is in place and the old ones are pruned, so it
    /// is measured at once; a new chapter starts a new growth.
    func note(clock: PlayClock) {
        chapter = clock.chapter
        let finished = wasSaving && !clock.saving
        wasSaving = clock.saving
        guard finished, world != nil else { return }
        Task { await refresh() }
    }

    /// The readout for the mode in force.
    func readout(mode: SaveMode) -> StorageReadout {
        StorageReadout(sample: sample, trend: trend, mode: mode)
    }
}
