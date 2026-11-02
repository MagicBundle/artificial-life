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
}
