import AlifeCore
import Foundation
import Testing

@testable import AlifeControl

/// The process side of the World window: a stand-in script plays `alife play` (a committed play stream on stdout, control
/// lines read from stdin and appended to a file), so that launching, control, the stop ladder and the end of a world are
/// exercised with real processes and pipes, without the engine or the GPU.
@MainActor
@Suite struct PlayControllerTests {
    private let request = PlayRequest.resume(PlayResume(run: "r"))

    private final class Recorder {
        var lines: [(text: String, isError: Bool)] = []
        var deliveries: [WorldStore.Changes] = []
    }

    private func make(ladder: EngineSession.Ladder = .init(beforeInterrupt: .milliseconds(300), beforeTerminate: .milliseconds(300)))
        -> (PlayController, Recorder) {
        let controller = PlayController()
        let recorder = Recorder()
        controller.log = { lines, isError in for text in lines { recorder.lines.append((text, isError)) } }
        controller.onDelivery = { recorder.deliveries.append($0) }
        controller.ladder = ladder
        controller.interruptGrace = .milliseconds(300)
        return (controller, recorder)
    }

    /// Plays a stream: the fixture, then reads control lines into `commands` until `quit` or EOF.
    private func standIn(in scratch: Scratch, fixture: String = "3d", serve: Bool = true) throws -> (URL, commands: URL) {
        let commands = scratch.url.appendingPathComponent("commands.txt")
        let body = """
        cat "$1"
        \(serve ? """
        while IFS= read -r line; do
          echo "$line" >> "$2"
          if [ "$line" = "quit" ]; then exit 0; fi
        done
        """ : "")
        exit 0
        """
        return (try scratch.script("alife-play-stand-in", body), commands)
    }

    private func commandLines(_ url: URL) -> [String] {
        ((try? String(contentsOf: url, encoding: .utf8)) ?? "").split(separator: "\n").map(String.init)
    }

    private func args(_ scratch: Scratch, commands: URL? = nil, kind: String = "3d") -> [String] {
        [Repo.playFixtureURL(kind).path, (commands ?? scratch.url.appendingPathComponent("commands.txt")).path]
    }

    // MARK: The stream

    @Test func aWorldThatStartsIsLiveWithItsRecords() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (script, commands) = try standIn(in: scratch)
        let (controller, recorder) = make()
        defer { controller.shutdown() }
        controller.start(request, executable: script, directory: scratch.url, arguments: args(scratch, commands: commands))
        #expect(controller.phase == .starting)
        #expect(controller.world == "r")
        #expect(await eventually { controller.phase == .live && controller.store.clock?.head.step == 104 })
        let store = controller.store
        #expect(store.header?.play?.records.contains("FRMA") == true)
        #expect(store.frame?.step == 104, "the newest frame")
        #expect(store.map != nil && store.storms != nil && store.identities != nil && store.focus != nil)
        #expect(store.chronicle.events.count >= 2)
        #expect(controller.revision > 1)
        #expect(recorder.deliveries.contains { $0.contains(.header) } && recorder.deliveries.contains { $0.contains(.frame) })
        #expect(recorder.lines.first?.text.hasPrefix("$ ") == true)
        controller.shutdown()
        #expect(await eventually { !controller.hasProcess })
    }

    @Test func aWorldThatEndsDeliversItsLastRecordsBeforeItFinishes() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (script, _) = try standIn(in: scratch, serve: false)
        let (controller, _) = make()
        defer { controller.shutdown() }
        controller.start(request, executable: script, directory: scratch.url, arguments: args(scratch))
        #expect(await eventually { !controller.hasProcess })
        guard case .ended = controller.phase else {
            Issue.record("expected the world to have ended, got \(controller.phase)")
            return
        }
        #expect(controller.store.frame?.step == 104 && controller.store.clock?.head.step == 104)
    }

    @Test func recordsAndEventsKeepTheirOrderWhateverTheChunking() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        // 7-byte writes cut every record and every number.
        let chopped = try scratch.script("alife-chopped", "dd if=\"$1\" bs=7 2>/dev/null\nexit 0")
        let (controller, _) = make()
        defer { controller.shutdown() }
        controller.start(request, executable: chopped, directory: scratch.url, arguments: [Repo.playFixtureURL("3d").path])
        #expect(await eventually(timeout: .seconds(30)) { !controller.hasProcess })
        let whole = try PlayFixtureStore.store("3d")
        #expect(controller.store.chronicle.events == whole.chronicle.events)
        #expect(controller.store.frame?.step == whole.frame?.step && controller.store.clock == whole.clock)
    }

    @Test func aStreamWithoutThePlayKeyStillCarriesAWorld() async throws {
        // The Lab's own stream has no `play` key and no records: the window shows a world with nothing in it, not a crash.
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let script = try scratch.script("alife-lab", "cat \"$1\"\nexit 0")
        let (controller, _) = make()
        defer { controller.shutdown() }
        controller.start(request, executable: script, directory: scratch.url, arguments: [Repo.streamFixtureURL("3d").path])
        #expect(await eventually { !controller.hasProcess })
        #expect(controller.store.header?.play == nil && controller.store.frame == nil)
    }

    // MARK: Commands

    @Test func commandsReachTheWorldInOrderAndStopQuitsOnce() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (script, commands) = try standIn(in: scratch)
        let (controller, _) = make()
        defer { controller.shutdown() }
        controller.start(request, executable: script, directory: scratch.url, arguments: args(scratch, commands: commands))
        #expect(await eventually { controller.phase == .live })

        #expect(controller.send(.pause))
        #expect(controller.requestedRate == .multiplier(0))
        controller.setRate(.multiplier(2))
        #expect(controller.requestedRate == .multiplier(2))
        controller.step(25)
        controller.setColor("kin")
        controller.setColor("nonsense")  // not a mode of this stream: nothing is sent
        controller.send(.labels(false))
        controller.send(.focus(7))
        controller.resume()
        #expect(!controller.send(.step(count: 0)), "a command the grammar cannot express is not sent")

        controller.stop()
        controller.stop()
        #expect(controller.phase == .idle)
        #expect(controller.store.header == nil)
        #expect(await eventually { !controller.hasProcess })
        #expect(commandLines(commands) == ["pause", "rate 2", "step 25", "color kin", "labels off", "focus 7", "run", "quit"])
        #expect(!controller.send(.pause), "a stopped world takes no more commands")
    }

    @Test func nothingIsSentBeforeTheHeader() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let slow = try scratch.script("alife-slow", "sleep 0.5\ncat \"$1\"\nwhile IFS= read -r line; do echo \"$line\" >> \"$2\"; done")
        let (controller, _) = make()
        defer { controller.shutdown() }
        controller.start(request, executable: slow, directory: scratch.url, arguments: args(scratch))
        #expect(controller.phase == .starting)
        #expect(!controller.send(.pause))
        #expect(controller.requestedRate == nil)
        controller.shutdown()
        #expect(await eventually { !controller.hasProcess })
    }

    @Test func theNextClockConfirmsTheRateTheWindowAsked() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        // After the fixture, the stand-in answers `pause` with the fixture's last CLCK again.
        let clock = try scratch.script("alife-clock", """
            cat "$1"
            while IFS= read -r line; do
              if [ "$line" = "pause" ]; then
                dd if="$1" bs=1 skip=115784 count=68 2>/dev/null
              fi
              if [ "$line" = "quit" ]; then exit 0; fi
            done
            """)
        let (controller, _) = make()
        defer { controller.shutdown() }
        controller.start(request, executable: clock, directory: scratch.url, arguments: args(scratch))
        #expect(await eventually { controller.phase == .live && controller.store.clock != nil })
        controller.pause()
        #expect(controller.requestedRate == .multiplier(0))
        #expect(await eventually { controller.requestedRate == nil }, "a CLCK arrived: the world speaks for itself")
        controller.shutdown()
        #expect(await eventually { !controller.hasProcess })
    }

    // MARK: Stopping

    @Test func aWorldThatIgnoresQuitIsInterruptedAfterTheFirstWait() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let marker = scratch.url.appendingPathComponent("signals.txt")
        // Reads no stdin and ignores quit; SIGINT is what ends it (the runner's own stop path, which works while paused).
        let deaf = try scratch.script("alife-deaf", """
            trap 'echo INT >> "$2"; exit 0' INT
            cat "$1"
            while true; do sleep 0.05; done
            """)
        let (controller, _) = make()
        defer { controller.shutdown() }
        controller.start(request, executable: deaf, directory: scratch.url, arguments: [Repo.playFixtureURL("3d").path, marker.path])
        #expect(await eventually { controller.phase == .live })
        controller.stop()
        #expect(controller.phase == .idle)
        #expect(controller.retiredWorlds == ["r"], "asked to stop and still saving")
        #expect(await eventually(timeout: .seconds(10)) { !controller.hasProcess })
        #expect(commandLines(marker) == ["INT"])
        #expect(controller.retiredWorlds.isEmpty)
        #expect(controller.phase == .idle, "an asked-for stop is not reported as a failure")
    }

    @Test func aWorldThatIgnoresSIGINTToGetsSIGTERMAtTheEnd() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let stubborn = try scratch.script("alife-stubborn", "trap '' INT\ncat \"$1\"\nexec sleep 60")
        let (controller, _) = make()
        defer { controller.shutdown() }
        controller.start(request, executable: stubborn, directory: scratch.url, arguments: [Repo.playFixtureURL("3d").path])
        #expect(await eventually { controller.phase == .live })
        let began = ContinuousClock.now
        controller.stop()
        #expect(await eventually(timeout: .seconds(10)) { !controller.hasProcess })
        #expect(ContinuousClock.now - began >= .milliseconds(500), "the ladder waited for each rung")
        #expect(ContinuousClock.now - began < .seconds(8))
    }

    @Test func aWorldThatQuitsOnItsOwnNeverGetsASignal() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (script, commands) = try standIn(in: scratch)
        let (controller, _) = make(ladder: .init(beforeInterrupt: .milliseconds(100), beforeTerminate: .milliseconds(100)))
        defer { controller.shutdown() }
        controller.start(request, executable: script, directory: scratch.url, arguments: args(scratch, commands: commands))
        #expect(await eventually { controller.phase == .live })
        controller.stop()
        #expect(await eventually { !controller.hasProcess })
        try await Task.sleep(for: .milliseconds(400))
        #expect(commandLines(commands) == ["quit"])
        #expect(controller.phase == .idle)
    }

    @Test func aWorldBeingBuiltIsInterruptedAtOnceAndItsProgressIsShown() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let marker = scratch.url.appendingPathComponent("signals.txt")
        // The fork and the spin-up read no stdin: only SIGINT (exit 130, as `_spin_weather` does) ends them.
        let building = try scratch.script("alife-building", """
            trap 'echo INT >> "$1"; exit 130' INT
            echo "weather spin-up: 1000 of 20000 steps" >&2
            echo "weather spin-up: 2000 of 20000 steps" >&2
            while true; do sleep 0.05; done
            """)
        let (controller, recorder) = make(ladder: .init(beforeInterrupt: .seconds(60), beforeTerminate: .seconds(60)))
        defer { controller.shutdown() }
        controller.start(.planet(PlanetSpec(source: "s", name: "p")), executable: building, directory: scratch.url,
                         arguments: [marker.path])
        #expect(controller.world == "p")
        #expect(await eventually { controller.progress.count == 2 })
        #expect(controller.progress.last == "weather spin-up: 2000 of 20000 steps")
        #expect(controller.phase == .starting)
        #expect(recorder.lines.contains { $0.text == "[play] weather spin-up: 1000 of 20000 steps" && $0.isError })
        controller.stop()
        #expect(controller.phase == .idle)
        #expect(await eventually(timeout: .seconds(5)) { !controller.hasProcess })
        #expect(commandLines(marker) == ["INT"], "SIGINT at once, not after the 60 s of the live ladder")
    }

    @Test func aBuildThatIgnoresSIGINTIsTerminatedAfterTheGrace() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let stubborn = try scratch.script("alife-stubborn-build", "trap '' INT\necho ready >&2\nexec sleep 60")
        let (controller, _) = make()
        defer { controller.shutdown() }
        controller.start(.planet(PlanetSpec(source: "s", name: "p")), executable: stubborn, directory: scratch.url, arguments: [])
        #expect(await eventually { controller.progress == ["ready"] })
        let began = ContinuousClock.now
        controller.stop()
        try await Task.sleep(for: .milliseconds(100))
        #expect(controller.hasProcess, "SIGINT was sent and ignored; the grace has not run out")
        #expect(await eventually(timeout: .seconds(8)) { !controller.hasProcess })
        #expect(ContinuousClock.now - began >= .milliseconds(250), "SIGTERM came after the grace")
    }

    @Test func aWorldThatIsStillSavingCannotBeStartedAgain() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let deaf = try scratch.script("alife-saving", "trap '' INT\ncat \"$1\"\nwhile true; do sleep 0.05; done")
        let (controller, recorder) = make(ladder: .init(beforeInterrupt: .seconds(30), beforeTerminate: .seconds(30)))
        defer { controller.shutdown() }
        controller.start(request, executable: deaf, directory: scratch.url, arguments: [Repo.playFixtureURL("3d").path])
        #expect(await eventually { controller.phase == .live })
        controller.stop()
        #expect(controller.retiredWorlds == ["r"])
        controller.start(request, executable: deaf, directory: scratch.url, arguments: [Repo.playFixtureURL("3d").path])
        guard case .failed(let message) = controller.phase else {
            Issue.record("expected a refusal, got \(controller.phase)")
            return
        }
        #expect(message.contains("still saving"))
        #expect(recorder.lines.filter { $0.text.hasPrefix("$ ") }.count == 1, "no second process was started")
        // Another world is not held up by it.
        controller.start(.resume(PlayResume(run: "other")), executable: deaf, directory: scratch.url,
                         arguments: [Repo.playFixtureURL("3d").path])
        #expect(controller.phase == .starting)
        controller.shutdown()  // the app is quitting: no ladder
        #expect(await eventually(timeout: .seconds(10)) { !controller.hasProcess })
    }

    @Test func quittingEndsEveryProcessAtOnce() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let deaf = try scratch.script("alife-deaf-long", "cat \"$1\"\nexec sleep 60")
        let (controller, _) = make(ladder: .init(beforeInterrupt: .seconds(120), beforeTerminate: .seconds(120)))
        defer { controller.shutdown() }
        controller.start(request, executable: deaf, directory: scratch.url, arguments: [Repo.playFixtureURL("3d").path])
        #expect(await eventually { controller.phase == .live })
        controller.start(.resume(PlayResume(run: "other")), executable: deaf, directory: scratch.url,
                         arguments: [Repo.playFixtureURL("3d").path])
        #expect(await eventually { controller.phase == .live })
        #expect(controller.hasProcess && controller.retiredWorlds == ["r"])
        controller.shutdown()
        #expect(await eventually(timeout: .seconds(10)) { !controller.hasProcess }, "a replaced process was left running")
    }

    // MARK: Failures

    @Test func aFailingProcessReportsItsExitStatusAndItsLastErrorLine() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        for (code, text) in [(2, "exit code 2"), (4, "exit code 4"), (1, "exit code 1"), (3, "invariant"), (130, "interrupted")] {
            let script = try scratch.script("alife-fails-\(code)",
                                            "echo 'note: starting' >&2\necho 'error: no play.json in runs/r' >&2\nexit \(code)")
            let (controller, recorder) = make()
            controller.start(request, executable: script, directory: scratch.url, arguments: [])
            #expect(await eventually { !controller.hasProcess })
            guard case .failed(let message) = controller.phase else {
                Issue.record("expected a failure, got \(controller.phase)")
                return
            }
            #expect(message.contains(text) && message.contains("error: no play.json in runs/r"), "\(message)")
            #expect(recorder.lines.contains { $0.text == "[play] note: starting" && $0.isError })
        }
    }

    @Test func aSignalIsReported() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let script = try scratch.script("alife-killed", "kill -9 $$")
        let (controller, _) = make()
        defer { controller.shutdown() }
        controller.start(request, executable: script, directory: scratch.url, arguments: [])
        #expect(await eventually { !controller.hasProcess })
        guard case .failed(let message) = controller.phase else {
            Issue.record("expected a failure, got \(controller.phase)")
            return
        }
        #expect(message.contains("signal 9"))
    }

    @Test func aCorruptStreamIsRefusedAndItsProcessStopped() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let script = try scratch.script("alife-garbage", "echo 'this is not a stream'\nexec sleep 30")
        let (controller, _) = make()
        defer { controller.shutdown() }
        controller.start(request, executable: script, directory: scratch.url, arguments: [])
        #expect(await eventually(timeout: .seconds(10)) { !controller.hasProcess })
        #expect(controller.phase == .failed(StreamError.badMagic.description))
    }

    @Test func aRecordThatContradictsTheHeaderIsRefused() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        // The header, then a record whose length is not a multiple of 4: the decoder refuses it and the process is stopped.
        let bad = try scratch.script("alife-bad-record", """
            fixture="$1"
            head -c 1504 "$fixture"
            printf 'FRMA\\x07\\x00\\x00\\x00abcdefg'
            exec sleep 30
            """)
        let (controller, _) = make()
        defer { controller.shutdown() }
        controller.start(request, executable: bad, directory: scratch.url, arguments: [Repo.playFixtureURL("3d").path])
        #expect(await eventually(timeout: .seconds(10)) { !controller.hasProcess })
        guard case .failed(let message) = controller.phase else {
            Issue.record("expected a failure, got \(controller.phase)")
            return
        }
        #expect(message.contains("FRMA"))
    }

    @Test func anExecutableThatCannotLaunchIsAFailure() throws {
        let (controller, recorder) = make()
        defer { controller.shutdown() }
        controller.start(request, executable: URL(fileURLWithPath: "/nonexistent/alife"), directory: URL(fileURLWithPath: "/tmp"))
        guard case .failed(let message) = controller.phase else {
            Issue.record("expected a failure, got \(controller.phase)")
            return
        }
        #expect(message.contains("Failed to launch"))
        #expect(!controller.hasProcess)
        #expect(recorder.lines.contains { $0.isError })
    }

    @Test func aNewWorldReplacesTheOldOneAndTheOldOnesEndIsIgnored() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (script, commands) = try standIn(in: scratch)
        let (controller, _) = make()
        defer { controller.shutdown() }
        controller.start(request, executable: script, directory: scratch.url, arguments: args(scratch, commands: commands))
        #expect(await eventually { controller.phase == .live })
        controller.start(.resume(PlayResume(run: "other")), executable: script, directory: scratch.url,
                         arguments: args(scratch, commands: commands))
        #expect(controller.phase == .starting && controller.world == "other" && controller.store.header == nil)
        #expect(await eventually { controller.phase == .live })
        #expect(await eventually { commandLines(commands).contains("quit") })
        try await Task.sleep(for: .milliseconds(300))
        #expect(controller.phase == .live, "the first process leaving did not touch the second world")
        controller.shutdown()
        #expect(await eventually { !controller.hasProcess })
    }
}

// MARK: - The mailbox

@Suite struct PlayMailboxTests {
    private func records(_ kind: String = "3d") throws -> [StreamRecord] {
        try PlayFixtureStore.decode(kind).records
    }

    @Test func aLatestWinsTagKeepsItsNewestRecordAndCountsTheOthers() throws {
        let all = try records()
        let mailbox = StreamMailbox()
        for record in all { mailbox.publish(record: record) }
        let delivery = mailbox.take()
        // FRMA and CLCK each arrive twice in the fixture; the others once.
        #expect(delivery.records["FRMA"]?.head.step == 104 && delivery.records["CLCK"]?.head.step == 104)
        #expect(delivery.droppedRecords == ["FRMA": 1, "CLCK": 1])
        #expect(delivery.records.keys.sorted() == ["CLCK", "ENVS", "FOCS", "FRMA", "IDNT", "INTN", "MAP1", "STRM", "TERR"])
        #expect(delivery.received == 2, "the frames that arrived")
        #expect(mailbox.take().isEmpty)
    }

    @Test func eventsAreQueuedInOrderAndNeverReplaced() throws {
        let all = try records()
        let batches = all.compactMap { if case .evnt(let b) = $0 { b } else { nil } }
        #expect(batches.count == 2)
        let mailbox = StreamMailbox()
        for batch in batches { mailbox.publish(record: .evnt(batch)) }
        let delivery = mailbox.take()
        #expect(delivery.events == batches)
        #expect(delivery.records.isEmpty && !delivery.isEmpty)
        #expect(delivery.droppedRecords.isEmpty)
    }

    @Test func aRecordThroughTheReaderThreadReachesTheMailbox() async throws {
        let bytes = [UInt8](try Data(contentsOf: Repo.playFixtureURL("3d")))
        let pipe = Pipe()
        let mailbox = StreamMailbox()
        let (wakeups, wake) = AsyncStream.makeStream(of: Void.self, bufferingPolicy: .bufferingNewest(1))
        StreamReader.start(reading: pipe.fileHandleForReading, into: mailbox, wake: wake, chunkSize: 1 << 12)
        let writer = pipe.fileHandleForWriting
        let writing = Task.detached {
            try? writer.write(contentsOf: Data(bytes))
            try? writer.close()
        }
        var wakes = 0
        for await _ in wakeups { wakes += 1 }
        await writing.value
        let delivery = mailbox.take()
        #expect(wakes >= 1 && delivery.header?.play != nil)
        #expect(delivery.end == .complete)
        #expect(delivery.records["FRMA"]?.head.step == 104)
        #expect(delivery.events.count == 2)
    }
}

/// The records of a play fixture, decoded in one piece.
enum PlayFixtureStore {
    /// The store a client would hold after the whole fixture.
    static func store(_ kind: String) throws -> WorldStore {
        let (header, records) = try decode(kind)
        var store = WorldStore()
        store.reset(header: header)
        for record in records { store.apply(record) }
        return store
    }

    static func decode(_ kind: String) throws -> (header: StreamHeader, records: [StreamRecord]) {
        var decoder = FrameDecoder()
        let events = try decoder.feed(try Data(contentsOf: Repo.playFixtureURL(kind)))
        try decoder.finish()
        guard case .header(let header)? = events.first else { throw StreamError.badMagic }
        return (header, events.compactMap { if case .record(let r) = $0 { r } else { nil } })
    }
}

