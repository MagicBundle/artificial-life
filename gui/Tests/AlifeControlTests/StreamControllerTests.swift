import AlifeCore
import Foundation
import Testing

@testable import AlifeControl

/// The process side of the viewer: a stand-in script plays `alife stream` (a fixture on stdout, control lines read from
/// stdin and appended to a file), so that launching, control, stopping and the end of a stream are exercised with real
/// processes and pipes, without the engine or the GPU.
@MainActor
@Suite struct StreamControllerTests {
    private let source = StreamSource(run: "r", follow: false)

    /// A controller that records what it logs.
    private final class Recorder {
        var lines: [(text: String, isError: Bool)] = []
    }

    private func make(grace: Duration = .seconds(2)) -> (StreamController, Recorder) {
        let controller = StreamController()
        let recorder = Recorder()
        controller.log = { lines, isError in
            for text in lines { recorder.lines.append((text, isError)) }
        }
        controller.terminationGrace = grace
        return (controller, recorder)
    }

    /// Plays a stream: the fixture, then (when `serve`) reads control lines into `commands` until `quit` or EOF.
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
        return (try scratch.script("alife-stand-in", body), commands)
    }

    private func commandLines(_ url: URL) -> [String] {
        ((try? String(contentsOf: url, encoding: .utf8)) ?? "").split(separator: "\n").map(String.init)
    }

    // MARK: Frames and the end of a stream

    @Test func aStreamThatEndsDeliversItsLastFrameBeforeItFinishes() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (script, _) = try standIn(in: scratch, serve: false)
        let fixture = try StreamFixtureData(kind: "3d")
        let (controller, recorder) = make()
        controller.start(source, executable: script, directory: scratch.url,
                         arguments: [Repo.streamFixtureURL("3d").path])
        #expect(controller.phase == .starting)
        #expect(await eventually { !controller.hasProcess })
        // The process has exited and both pipes are at end of file: the last frame is in, whatever the order of events.
        #expect(controller.phase == .ended("The stream ended."))
        #expect(controller.frame.map { Int($0.step) } == fixture.lastStep)
        #expect(controller.header == fixture.header)
        #expect(recorder.lines.allSatisfy { !$0.isError })
    }

    @Test func theViewStartsAsTheHeaderSays() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (script, _) = try standIn(in: scratch)
        let (controller, _) = make()
        controller.start(source, executable: script, directory: scratch.url,
                         arguments: [Repo.streamFixtureURL("3d").path, scratch.url.appendingPathComponent("c").path])
        #expect(await eventually { controller.phase == .live && controller.frameSerial > 1 })
        let fixture = try StreamFixtureData(kind: "3d")
        #expect(controller.control.color == fixture.header.color)
        #expect(controller.control.every == fixture.header.every)
        #expect(controller.control.fields == fixture.header.fields)
        controller.shutdown()
        #expect(await eventually { !controller.hasProcess })
    }

    @Test func aFailingProcessReportsItsExitStatusAndItsLastErrorLine() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let script = try scratch.script("alife-fails", "echo 'note: starting' >&2\necho 'error: no checkpoint in runs/r' >&2\nexit 4")
        let (controller, recorder) = make()
        controller.start(source, executable: script, directory: scratch.url, arguments: [])
        #expect(await eventually { !controller.hasProcess })
        guard case .failed(let message) = controller.phase else {
            Issue.record("expected a failure, got \(controller.phase)")
            return
        }
        #expect(message.contains("exit code 4"))
        #expect(message.contains("error: no checkpoint in runs/r"))
        #expect(recorder.lines.contains { $0.text == "[3d] note: starting" && $0.isError })
    }

    @Test func aSignalIsReported() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let script = try scratch.script("alife-killed", "kill -9 $$")
        let (controller, _) = make()
        controller.start(source, executable: script, directory: scratch.url, arguments: [])
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
        // Not a stream, and a process that would live for half a minute: the viewer ends it.
        let script = try scratch.script("alife-garbage", "echo 'this is not a stream'\nexec sleep 30")
        let (controller, _) = make()
        controller.start(source, executable: script, directory: scratch.url, arguments: [])
        #expect(await eventually(timeout: .seconds(10)) { !controller.hasProcess })
        #expect(controller.phase == .failed(StreamError.badMagic.description))
    }

    @Test func anExecutableThatCannotLaunchIsAFailure() throws {
        let (controller, recorder) = make()
        controller.start(source, executable: URL(fileURLWithPath: "/nonexistent/alife"),
                         directory: URL(fileURLWithPath: "/tmp"))
        guard case .failed(let message) = controller.phase else {
            Issue.record("expected a failure, got \(controller.phase)")
            return
        }
        #expect(message.contains("Failed to launch"))
        #expect(!controller.hasProcess)
        #expect(recorder.lines.contains { $0.isError })
    }

    @Test func theStartCommandIsLogged() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (script, _) = try standIn(in: scratch, serve: false)
        let (controller, recorder) = make()
        controller.start(source, executable: script, directory: scratch.url,
                         arguments: [Repo.streamFixtureURL("3d").path])
        #expect(recorder.lines.first?.text.hasPrefix("$ ") == true)
        #expect(recorder.lines.first?.isError == false)
        #expect(await eventually { !controller.hasProcess })
    }

    // MARK: Control and stopping

    @Test func commandsReachTheProcessInOrderAndStopQuitsOnce() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (script, commands) = try standIn(in: scratch)
        let (controller, _) = make()
        controller.start(source, executable: script, directory: scratch.url,
                         arguments: [Repo.streamFixtureURL("3d").path, commands.path])
        #expect(await eventually { controller.phase == .live })

        controller.pause()
        #expect(controller.control.paused)
        controller.faster()  // the fixture's period is 4
        #expect(controller.control.every == 8)
        controller.slower()
        controller.slower()
        #expect(controller.control.every == 2)
        controller.setColor("energy")
        controller.setColor("kin")  // not a mode of this stream: nothing is sent
        controller.cycleColor()  // energy -> clade
        #expect(controller.control.color == "clade")
        controller.setFields(true)
        controller.setFieldLayer(99)  // clamped to the last layer, FZ - 1 = 5
        #expect(controller.control.fieldLayer == 5)
        controller.step(3)
        controller.run()
        #expect(!controller.control.paused)

        // Exactly once, however many times it is asked.
        controller.stop()
        controller.stop()
        #expect(controller.phase == .idle)
        #expect(await eventually { !controller.hasProcess })

        #expect(commandLines(commands) == [
            "pause", "every 8", "every 4", "every 2", "color energy", "color clade", "fields on", "fields z 5",
            "step 3", "run", "quit",
        ])
        // A stopped stream takes no more commands.
        controller.pause()
        #expect(commandLines(commands).last == "quit")
    }

    @Test func aFollowedRunHasNoSpeed() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (script, commands) = try standIn(in: scratch)
        let (controller, _) = make()
        controller.start(StreamSource(run: "r", follow: true), executable: script, directory: scratch.url,
                         arguments: [Repo.streamFixtureURL("3d").path, commands.path])
        #expect(await eventually { controller.phase == .live })
        controller.faster()
        controller.setEvery(7)
        controller.pause()
        controller.stop()
        #expect(await eventually { !controller.hasProcess })
        #expect(commandLines(commands) == ["pause", "quit"])
    }

    @Test func aProcessThatIgnoresQuitIsTerminatedAfterTheGrace() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        // Sends a header and then sleeps, ignoring stdin: only the SIGTERM after the grace period ends it.
        let script = try scratch.script("alife-deaf", "cat \"$1\"\nexec sleep 30")
        let (controller, _) = make(grace: .milliseconds(200))
        controller.start(source, executable: script, directory: scratch.url,
                         arguments: [Repo.streamFixtureURL("3d").path])
        #expect(await eventually { controller.phase == .live })
        controller.stop()
        #expect(controller.phase == .idle)
        #expect(controller.header == nil)
        #expect(controller.frame == nil)
        #expect(await eventually(timeout: .seconds(10)) { !controller.hasProcess })
        #expect(controller.phase == .idle)  // an asked-for stop is not reported as a failure
    }

    @Test func aNewStreamReplacesTheOldOneAndTheOldOnesEndIsIgnored() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (script, commands) = try standIn(in: scratch)
        let (controller, _) = make()
        let arguments = [Repo.streamFixtureURL("3d").path, commands.path]
        controller.start(source, executable: script, directory: scratch.url, arguments: arguments)
        #expect(await eventually { controller.phase == .live })
        let first = controller.frameSerial
        controller.start(StreamSource(run: "other", follow: true), executable: script, directory: scratch.url,
                         arguments: arguments)
        #expect(controller.phase == .starting)
        #expect(controller.source?.run == "other")
        #expect(await eventually { controller.phase == .live && controller.frameSerial > first })
        // The first process gets its quit and leaves; that must not touch the second stream.
        #expect(await eventually { commandLines(commands).contains("quit") })
        try await Task.sleep(for: .milliseconds(300))
        #expect(controller.phase == .live)
        #expect(controller.header != nil)
        controller.shutdown()
        #expect(await eventually { !controller.hasProcess })
    }

    @Test func whatAReplacedStreamSendsLaterIsIgnored() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        // The old process only speaks after it has been replaced; the new one says nothing.
        let late = try scratch.script("alife-late", "sleep 0.4\ncat \"$1\"\nwhile IFS= read -r line; do :; done")
        let silent = try scratch.script("alife-silent", "while IFS= read -r line; do :; done")
        let (controller, _) = make()
        controller.start(source, executable: late, directory: scratch.url,
                         arguments: [Repo.streamFixtureURL("3d").path])
        controller.start(StreamSource(run: "other", follow: false), executable: silent, directory: scratch.url,
                         arguments: [])
        try await Task.sleep(for: .milliseconds(1200))
        #expect(controller.phase == .starting)
        #expect(controller.header == nil && controller.frame == nil)
        controller.stop()
        #expect(await eventually { !controller.hasProcess })
    }

    @Test func switchingTheSourceRestartsTheSameRun() async throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let (script, commands) = try standIn(in: scratch)
        let (controller, _) = make()
        controller.start(source, executable: script, directory: scratch.url,
                         arguments: [Repo.streamFixtureURL("3d").path, commands.path])
        #expect(await eventually { controller.phase == .live })
        controller.restart(follow: false)  // already a re-simulation: nothing happens
        #expect(controller.phase == .live)
        // The restart builds the real arguments, which the stand-in ignores or fails on; the source is what counts.
        controller.restart(follow: true)
        #expect(controller.source == StreamSource(run: "r", follow: true))
        #expect(controller.phase == .starting)
        controller.shutdown()
        #expect(await eventually { !controller.hasProcess })
    }

    @Test func aWriteToAPipeWhoseReaderHasGoneFailsInsteadOfKillingTheApp() throws {
        let pipe = Pipe()
        pipe.fileHandleForWriting.suppressSIGPIPE()
        try pipe.fileHandleForReading.close()
        #expect(throws: (any Error).self) { try pipe.fileHandleForWriting.write(contentsOf: Data("pause\n".utf8)) }
    }

    // MARK: Command line

    @Test func theArgumentsCarryTheViewAcrossARestart() {
        let sim = StreamSource(run: "r", follow: false)
        let follow = StreamSource(run: "r", follow: true)
        #expect(StreamLaunch.arguments(for: sim, every: 1, color: "tag", fieldLayer: nil)
                == ["stream", "runs/r", "--color", "tag"])
        #expect(StreamLaunch.arguments(for: sim, every: 8, color: "clade", fieldLayer: 3)
                == ["stream", "runs/r", "--every", "8", "--color", "clade", "--field-z", "3"])
        // --follow refuses --every.
        #expect(StreamLaunch.arguments(for: follow, every: 8, color: "energy", fieldLayer: nil)
                == ["stream", "runs/r", "--follow", "--color", "energy"])
    }

    @Test func theSnapshotLandsNextToTheTwoDViewersImages() {
        let url = ViewerSnapshot.url(directory: URL(fileURLWithPath: "/repo"), run: "r", step: 1200)
        #expect(url.path == "/repo/runs/r/media/view3d_1200.png")
    }
}
