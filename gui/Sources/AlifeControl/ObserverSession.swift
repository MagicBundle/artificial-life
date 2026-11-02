import AlifeCore
import Combine
import Foundation

/// The observer's stream for the research form of Watch (docs/APP.md §3.4, §7.5): which research world it looks at, and what to do next about
/// it (`WatchGuide.observerStep`). It is the app's, like `WorldSession`, so that the World menu can reach the viewer's actions and the viewer
/// keeps its camera while the player goes to another page and back. The stream starts when the Watch page shows a research world and ends
/// when the page is left (`onDisappear { stream.stop() }`, as the 3D Viewer window did): the world itself goes on.
@MainActor
final class ObserverSession: ObservableObject {
    let stream: StreamController
    let viewer: ViewerModel
    private let controller: RunController
    /// Whether a world has a save point to look at (a short listing of its `checkpoints/`; the tests give theirs).
    var hasSavePoint: (String) -> Bool
    /// The research world this app grows now (the run job's); the tests give theirs.
    var runningWorld: () -> String?
    /// Starts a stream (the tests record the request instead of launching the engine).
    var startStream: (StreamSource, _ keepingView: Bool) -> Void

    /// The world the page shows.
    @Published private(set) var world: String?
    /// What the last tick decided, for the page to say.
    @Published private(set) var step: WatchGuide.ObserverStep = .none
    /// Starts that failed in a row for this world. Reset by a live picture, a new world or *Try again*.
    @Published private(set) var failures = 0
    /// The page is on screen (the stream is wanted).
    private var shown = false
    private var cancellables: Set<AnyCancellable> = []

    init(controller: RunController, stream: StreamController = .shared) {
        self.controller = controller
        self.stream = stream
        hasSavePoint = { [weak controller] world in
            guard let controller else { return false }
            return !PlayCatalogScanner.checkpointSteps(in: controller.repoURL.appendingPathComponent("runs/\(world)", isDirectory: true)).isEmpty
        }
        runningWorld = { [weak controller] in controller?.runningRunName }
        startStream = { [weak controller, weak stream] source, keeping in
            guard let controller, let stream else { return }
            stream.start(source, executable: controller.alifeURL, directory: controller.repoURL, keepingView: keeping)
        }
        viewer = ViewerModel(stream: stream)
        stream.$phase.removeDuplicates().sink { [weak self] phase in
            MainActor.assumeIsolated {
                guard let self else { return }
                switch phase {
                case .failed: self.failures += 1
                case .live: self.failures = 0
                default: break
                }
            }
        }.store(in: &cancellables)
    }

    /// Whether this app's run grows `world`.
    func isGrowing(_ world: String) -> Bool { runningWorld().map { NameFolding.same($0, world) } ?? false }

    func facts(for world: String) -> WatchGuide.ObserverFacts {
        let phase: WatchGuide.ObserverFacts.Stream
        switch stream.phase {
        case .idle: phase = .idle
        case .starting: phase = .starting
        case .live: phase = .live
        case .ended: phase = .ended
        case .failed: phase = .failed
        }
        return WatchGuide.ObserverFacts(
            world: world, hasSavePoint: hasSavePoint(world), growing: isGrowing(world),
            streamWorld: stream.source?.run, follows: stream.source?.follow ?? false, stream: phase, failures: failures)
    }

    /// The page shows `world`: the stream is wanted for it.
    func show(_ world: String) {
        if self.world != world {
            self.world = world
            failures = 0
        }
        shown = true
        tick()
    }

    /// The page was left: the stream ends (a stream has nothing to save).
    func leave() {
        shown = false
        step = .none
        if stream.phase != .idle { stream.stop() }
    }

    /// One look, every second or so while the page is shown: starts, restarts or waits for the stream.
    func tick() {
        guard shown, let world, controller.repoReady else { return }
        let next = WatchGuide.observerStep(facts(for: world))
        if next != step { step = next }
        switch next {
        case .none, .wait: break
        case .start(let follow), .restart(let follow):
            startStream(StreamSource(run: world, follow: follow), stream.source?.run == world)
        }
    }

    /// *Try again* after the starts that failed: the count starts over.
    func tryAgain() {
        failures = 0
        if stream.phase != .live, stream.phase != .starting { stream.stop() }
        tick()
    }

    /// The Source picker: follow the run, or re-simulate it.
    func choose(follow: Bool) {
        guard let world, stream.source?.follow != follow else { return }
        startStream(StreamSource(run: world, follow: follow), true)
    }

    /// The stream's failure, in words, for the picture's place.
    var failureText: String? {
        if case .failed(let message) = stream.phase { return message }
        return nil
    }
}
