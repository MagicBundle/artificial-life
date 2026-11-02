import AlifeCore
import Foundation
import Testing

@testable import AlifeControl

/// The observer of the research Watch (docs/APP.md §3.4): which world its stream looks at, and what it does next, with a recorded start
/// instead of the engine.
@MainActor
@Suite("The observer session", .serialized)
struct ObserverSessionTests {
    @MainActor struct Rig {
        let scratch: Scratch
        let controller: RunController
        let stream = StreamController()
        let session: ObserverSession
        var started: Recorder

        final class Recorder { var starts: [(source: StreamSource, keeping: Bool)] = [] }

        init(engine: Bool = true, savePoints: Set<String> = ["w"], running: String? = nil) throws {
            scratch = try Scratch()
            let root = scratch.url.appendingPathComponent("folder")
            if engine {
                try FileManager.default.createDirectory(at: root.appendingPathComponent(".venv/bin"), withIntermediateDirectories: true)
                let alife = try scratch.script("alife", "exit 0")
                try FileManager.default.moveItem(at: alife, to: root.appendingPathComponent(".venv/bin/alife"))
            }
            controller = RunController(storedPath: root.path, savePath: { _ in }, candidates: [], isApp: false)
            session = ObserverSession(controller: controller, stream: stream)
            started = Recorder()
            let recorder = started
            session.hasSavePoint = { savePoints.contains($0) }
            session.runningWorld = { running }
            session.startStream = { source, keeping in recorder.starts.append((source, keeping)) }
        }
    }

    @Test func showingAWorldWithASavePointStartsItsStream() throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        rig.session.show("w")
        #expect(rig.started.starts.count == 1 && rig.started.starts[0].source == StreamSource(run: "w", follow: false))
        #expect(rig.session.world == "w")
    }

    @Test func aRunThisAppGrowsIsFollowed() throws {
        let rig = try Rig(running: "w")
        defer { rig.scratch.cleanup() }
        rig.session.show("w")
        #expect(rig.started.starts.map(\.source) == [StreamSource(run: "w", follow: true)])
        #expect(rig.session.isGrowing("w") && rig.session.isGrowing("W") && !rig.session.isGrowing("x"))
    }

    @Test func aWorldWithNoSavePointWaitsAndStartsNothing() throws {
        let rig = try Rig(savePoints: [])
        defer { rig.scratch.cleanup() }
        rig.session.show("w")
        #expect(rig.started.starts.isEmpty)
        #expect(rig.session.step == .wait(WatchGuide.waitingText))
    }

    @Test func nothingStartsBeforeThePageShowsAWorldOrWithoutAnEngine() throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        rig.session.tick()
        #expect(rig.started.starts.isEmpty, "no world shown")
        let none = try Rig(engine: false)
        defer { none.scratch.cleanup() }
        none.session.show("w")
        #expect(none.started.starts.isEmpty, "no engine")
    }

    @Test func leavingThePageStopsTheStreamAndTheTickDoesNothingMore() throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        rig.session.show("w")
        rig.session.leave()
        rig.session.tick()
        #expect(rig.started.starts.count == 1 && rig.session.step == .none)
        #expect(rig.stream.phase == .idle)
    }

    @Test func comingBackStartsItAgain() throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        rig.session.show("w")
        rig.session.leave()
        rig.session.show("w")
        #expect(rig.started.starts.count == 2)
    }

    @Test func theSourcePickerRestartsTheSameWorldKeepingTheView() throws {
        let rig = try Rig(running: "w")
        defer { rig.scratch.cleanup() }
        rig.session.show("w")
        rig.session.choose(follow: false)
        #expect(rig.started.starts.count == 2 && rig.started.starts[1].source == StreamSource(run: "w", follow: false) && rig.started.starts[1].keeping)
    }

    @Test func aFailedStartIsCountedAndResetByTryAgain() async throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        rig.session.show("w")
        // a stand-in engine that exits at once: the real stream controller fails
        let exe = try rig.scratch.script("fail", "exit 4")
        rig.stream.start(StreamSource(run: "w", follow: false), executable: exe, directory: rig.scratch.url)
        #expect(await eventually { if case .failed = rig.stream.phase { true } else { false } })
        #expect(rig.session.failures == 1)
        rig.session.tick()
        #expect(rig.started.starts.count == 2, "tried again by itself")
        rig.session.tryAgain()
        #expect(rig.session.failures == 0)
    }

    /// A stream that goes live and stays (a stand-in engine that plays the committed fixture and then waits for `quit`).
    private func goLive(_ rig: Rig, _ source: StreamSource) async throws {
        let commands = rig.scratch.url.appendingPathComponent("commands.txt")
        let script = try rig.scratch.script("live", "cat \"$1\"\nwhile IFS= read -r line; do echo \"$line\" >> \"$2\"; if [ \"$line\" = quit ]; then exit 0; fi; done\nexit 0")
        rig.stream.start(source, executable: script, directory: rig.scratch.url, arguments: [Repo.streamFixtureURL("3d").path, commands.path])
        #expect(await eventually { rig.stream.phase == .live })
    }

    @Test func leavingThePageEndsALiveStream() async throws {
        let rig = try Rig()
        defer { rig.stream.stop(); rig.scratch.cleanup() }
        rig.session.show("w")
        try await goLive(rig, StreamSource(run: "w", follow: false))
        rig.session.leave()
        #expect(rig.stream.phase == .idle && rig.stream.source == nil, "a stream has nothing to save: it ends with the page")
    }

    @Test func theSourcePickerDoesNothingWhenTheSourceIsAlreadyTheOneAsked() async throws {
        let rig = try Rig(running: "w")
        defer { rig.stream.stop(); rig.scratch.cleanup() }
        rig.session.show("w")
        let before = rig.started.starts.count
        try await goLive(rig, StreamSource(run: "w", follow: true))
        rig.session.choose(follow: true)
        #expect(rig.started.starts.count == before, "following a followed run starts nothing")
        rig.session.choose(follow: false)
        #expect(rig.started.starts.count == before + 1)
    }

    @Test func aLivePictureClearsTheCountOfFailedStarts() async throws {
        let rig = try Rig()
        defer { rig.stream.stop(); rig.scratch.cleanup() }
        rig.session.show("w")
        let exe = try rig.scratch.script("fail", "exit 4")
        rig.stream.start(StreamSource(run: "w", follow: false), executable: exe, directory: rig.scratch.url)
        #expect(await eventually { if case .failed = rig.stream.phase { true } else { false } })
        #expect(rig.session.failures == 1)
        try await goLive(rig, StreamSource(run: "w", follow: false))
        #expect(rig.session.failures == 0)
    }

    @Test func aDifferentWorldStartsTheCountOverAndTheSameWorldKeepsIt() async throws {
        let rig = try Rig(savePoints: ["w", "v"])
        defer { rig.stream.stop(); rig.scratch.cleanup() }
        rig.session.show("w")
        let exe = try rig.scratch.script("fail", "exit 4")
        rig.stream.start(StreamSource(run: "w", follow: false), executable: exe, directory: rig.scratch.url)
        #expect(await eventually { rig.session.failures == 1 })
        rig.session.show("w")
        #expect(rig.session.failures == 1, "the page appearing again for the same world does not forgive the failures")
        rig.session.show("v")
        #expect(rig.session.failures == 0 && rig.session.world == "v")
    }

    @Test func aRunStartedFromATerminalIsFollowedAndTheSourcePickerCanStillChoose() throws {
        let rig = try Rig()
        defer { rig.scratch.cleanup() }
        rig.session.inUseElsewhere = { $0 == "w" }
        rig.session.show("w")
        #expect(rig.started.starts.map(\.source) == [StreamSource(run: "w", follow: true)])
        // a world nobody runs can be followed too, when the player asks (the old 3D Viewer could): the choice is kept for the world
        let other = try Rig(savePoints: ["w"])
        defer { other.scratch.cleanup() }
        other.session.show("w")
        other.session.choose(follow: true)
        #expect(other.started.starts.last?.source == StreamSource(run: "w", follow: true) && other.session.chosenFollow == true)
        other.session.show("v")
        #expect(other.session.chosenFollow == nil, "a choice is for the world it was made on")
    }

    @Test func aFailedStartIsRetriedAsTheChoiceTheWorldHad() async throws {
        let rig = try Rig(running: "w")
        defer { rig.stream.stop(); rig.scratch.cleanup() }
        rig.session.show("w")
        rig.session.choose(follow: false)
        let exe = try rig.scratch.script("fail", "exit 4")
        rig.stream.start(StreamSource(run: "w", follow: false), executable: exe, directory: rig.scratch.url)
        #expect(await eventually { rig.session.failures == 1 })
        let before = rig.started.starts.count
        rig.session.tick()
        #expect(rig.started.starts.count == before + 1 && rig.started.starts.last?.source == StreamSource(run: "w", follow: false),
                "a deliberate Re-simulate is not turned into Follow by the retry of a world that grows")
    }
}
