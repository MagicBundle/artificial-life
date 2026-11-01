import AlifeCore
import AppKit
import Foundation
import Testing

@testable import AlifeControl

/// A scratch alife folder with a stand-in engine, and the objects of the set-up page wired to it as the app wires them. The stand-in is a bash
/// script that makes the files the real engine makes (config, meta, save points, metrics) and behaves as `.mode` says, `run:play`:
/// - run: `ok` (finishes at once), `extinct` (ends at step 4,200 with status extinct and exit 0, as the engine does when nothing is alive), `silent` (makes its directory and nothing else, and ends on SIGINT as a signal: the engine in its first seconds), `long` (runs until it gets SIGINT, then saves at step 12,000 and exits 130), `refuse` (prints a traceback, exit 2)
/// - play: `ok` (progress lines, then the committed play stream until `quit`), `slow` (progress lines, then waits for SIGINT: exit 130, leaves an
///   unfinished build), `fail` (prints its reason, exit 4), `extinct` (says the world is extinct and exits 0 before any stream), `quits` (ends by itself before any stream)
@MainActor
final class PipelineRig {
    let scratch: Scratch
    let repo: URL
    let runs: URL
    let controller: RunController
    let play = PlayController()
    let library: WorldLibrary
    let setup: SetupModel
    var coordinator: PipelineCoordinator { setup.coordinator }

    var looking = true
    var bounces = 0
    var playerStarts = 0
    var activityShown = 0
    var dismissedFailures = 0
    var opened: [(SetupIntent, String?)] = []
    private var ownsScratch: Bool

    init(mode: String = "ok:ok", repo existing: URL? = nil, scratch existingScratch: Scratch? = nil) throws {
        if let existingScratch, let existing {
            scratch = existingScratch
            repo = existing
            ownsScratch = false
        } else {
            scratch = try Scratch()
            repo = scratch.url.appendingPathComponent("alife-folder")
            ownsScratch = true
            try Self.makeFolder(repo, in: scratch)
        }
        runs = repo.appendingPathComponent("runs")
        try mode.write(to: repo.appendingPathComponent(".mode"), atomically: true, encoding: .utf8)
        controller = RunController(storedPath: repo.path, savePath: { _ in }, candidates: [], isApp: false)
        library = WorldLibrary(
            runsDirectory: { [repo] in repo.appendingPathComponent("runs") }, scanner: LibraryScanner(settle: 2, timeout: 60, inline: true), observesApp: false,
            watchesFolder: false)
        let defaults = UserDefaults(suiteName: "alife-rig-\(UUID().uuidString)")!
        setup = SetupModel.forApp(controller: controller, play: play, library: library, defaults: defaults)
        play.ladder = .init(beforeInterrupt: .seconds(2), beforeTerminate: .seconds(2))
        wire()
    }

    var mode: String {
        get { (try? String(contentsOf: repo.appendingPathComponent(".mode"), encoding: .utf8)) ?? "ok:ok" }
        set { try? newValue.write(to: repo.appendingPathComponent(".mode"), atomically: true, encoding: .utf8) }
    }

    private func wire() {
        coordinator.hooks.isLooking = { [unowned self] in looking }
        coordinator.hooks.bounce = { [unowned self] in bounces += 1 }
        coordinator.hooks.playerStartsWorld = { [unowned self] in playerStarts += 1 }
        coordinator.hooks.showActivity = { [unowned self] in activityShown += 1 }
        coordinator.hooks.dismissStartFailure = { [unowned self] in dismissedFailures += 1 }
        coordinator.hooks.openSetup = { [unowned self] intent, name in opened.append((intent, name)) }
        coordinator.isQuitting = { false }
    }

    /// Tells the library what the app does with its worlds, as the window does.
    func tellLibrary() {
        var facts = LiveFacts()
        if play.isActive, let world = play.world { facts.openWorld = world; facts.openPhase = play.phase == .starting ? .starting : .live }
        facts.growing = controller.runningRunName
        facts.saving = Set(play.retiredWorlds)
        facts.pending = coordinator.pendingPlanets
        library.update(live: facts)
        setup.launcher.apply(catalog: library.state.catalog, saving: Set(play.retiredWorlds))
    }

    func scan() async {
        let before = library.scans
        library.refresh()
        _ = await patiently { library.scans > before && library.hasScanned }
        tellLibrary()
    }

    /// Ends what the test started. `keepFolder`: the folder is wanted by a second rig that stands for the app launched again.
    func finish(keepFolder: Bool = false) {
        play.shutdown()
        controller.forceStop()
        if ownsScratch, !keepFolder { scratch.cleanup() }
    }

    /// The numbers a 2 s tick would read.
    func tickMetrics() { controller.tick(); controller.tick() }

    // MARK: Pressing the page's button

    /// Opens Create on a recipe, names it and presses the button. Returns the reasons the button was grey for (empty when it started).
    @discardableResult
    func create(_ name: String = "planet-1", recipe: Recipe = Recipes.smallPlanet, goal: Goal = .play, steps: String? = nil) async -> [String] {
        await scan()
        setup.openCreate(.recipe(recipe.id))
        setup.goal = goal
        setup.name = name
        if let steps { setup.setGrowSteps(steps) }
        let reasons = setup.reasons
        if reasons.isEmpty { setup.start() }
        tellLibrary()
        return reasons
    }

    func waitForState(timeout: Duration = .seconds(120), _ test: @MainActor (PipelinePlan.State) -> Bool) async -> Bool {
        await patiently(timeout: timeout) { coordinator.plan.map { test($0.state) } ?? false }
    }

    /// The run of the pipeline has started and the stand-in has made its directory (it listens for Stop from then on).
    func waitForGrowing(_ name: String = "planet-1-grow") async -> Bool {
        await patiently { controller.job == .run && exists(name) }
    }

    /// A research world with a whole config (every 3D value of the planet preset), so that a Branch from it is a valid form.
    @discardableResult
    func makeSource(_ name: String, steps: [Int] = [30_000], status: String = "done") throws -> URL {
        let dir = runs.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("checkpoints"), withIntermediateDirectories: true)
        let config = """
        {"world": {"dims": 3, "geometry": "sphere", "radius": 36, "height": 16, "nside": 8, "layers": 8, "size": 64},
         "physics": {"gravity": 0.03, "buoyancy_max": 0.1, "ceiling_band": 0.0}, "energy": {"buoyancy_cost": 1, "buoyancy_cap_cost": 1},
         "fields": {"benthic_lambda": 4.0, "photic_amp": 0.0, "photic_lambda": 1.0}, "mutation": {"buoy_init": 0.5, "buoy_sigma": 0.02},
         "weather": {"enabled": false}, "run": {"steps": 30000, "metrics_every": 10}}
        """
        try config.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "{\"status\": \"\(status)\", \"start_step\": 0}".write(to: dir.appendingPathComponent("meta.json"), atomically: true, encoding: .utf8)
        for step in steps { try "x".write(to: dir.appendingPathComponent("checkpoints/step_\(String(format: "%010d", step)).safetensors"), atomically: true, encoding: .utf8) }
        return dir
    }

    func exists(_ name: String) -> Bool { FileManager.default.fileExists(atPath: runs.appendingPathComponent(name).path) }

    // MARK: The folder

    static func makeFolder(_ repo: URL, in scratch: Scratch) throws {
        let fm = FileManager.default
        let bin = repo.appendingPathComponent(".venv/bin")
        try fm.createDirectory(at: bin, withIntermediateDirectories: true)
        try fm.createDirectory(at: repo.appendingPathComponent("runs"), withIntermediateDirectories: true)
        // the real presets, read-only, so that the recipes' planet shapes are the real ones
        try fm.createDirectory(at: repo.appendingPathComponent("src/alife"), withIntermediateDirectories: true)
        try fm.createSymbolicLink(at: repo.appendingPathComponent("src/alife/presets"), withDestinationURL: Repo.root.appendingPathComponent("src/alife/presets"))
        let script = bin.appendingPathComponent("alife")
        try (Self.engine(fixture: Repo.playFixtureURL("3d_g3b").path)).write(to: script, atomically: true, encoding: .utf8)
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
    }

    static func engine(fixture: String) -> String {
        """
        #!/bin/bash
        stopped=0; trap 'stopped=1' INT
        cd "$(dirname "$0")/../.."
        mode=$(cat .mode 2>/dev/null || echo ok:ok)
        runmode=${mode%%:*}
        playmode=${mode##*:}
        cmd=$1; shift
        first=$1
        out=""; init=""
        while [ $# -gt 0 ]; do
          case "$1" in
            --out) out=$2; shift 2;;
            --init-from) init=$2; shift 2;;
            *) shift;;
          esac
        done
        mkworld() {
          mkdir -p "$1/checkpoints"
          echo '{"world": {"dims": 3, "geometry": "sphere", "radius": 36, "height": 16, "nside": 8, "layers": 8, "size": 64}, "weather": {"enabled": false}, "run": {"steps": 30000, "metrics_every": 10}}' > "$1/config.json"
          echo "{\\"status\\": \\"$2\\", \\"start_step\\": 0}" > "$1/meta.json"
          : > "$1/checkpoints/step_$(printf %010d $3).safetensors"
          echo "{\\"step\\": $3, \\"n_alive\\": 100, \\"steps_per_s\\": 300}" >> "$1/metrics.jsonl"
        }
        case "$cmd" in
        run|resume)
          [ "$cmd" = resume ] && out=$first
          case "$runmode" in
            refuse)
              mkworld "$out" running 0
              echo "Traceback (most recent call last):" >&2
              echo "ValueError: refused by the stand-in" >&2
              exit 2;;
            extinct)
              mkworld "$out" extinct 4200
              echo "extinct at step 4200" >&2
              exit 0;;
            silent)
              mkdir -p "$out"
              exec sleep 30;;
            long)
              mkworld "$out" running 12000
              i=0
              while [ $stopped = 0 ] && [ $i -lt 900 ]; do sleep 0.1; i=$((i+1)); done
              if [ $stopped = 1 ]; then mkworld "$out" interrupted 12600; exit 130; fi
              mkworld "$out" done 30000; exit 0;;
            *)
              mkworld "$out" running 100
              sleep 0.3
              mkworld "$out" done 30000
              exit 0;;
          esac;;
        play)
          case "$playmode" in
            fail) echo "alife play found no usable checkpoint (exit code 4)" >&2; exit 4;;
            extinct) echo "play: the world is extinct at step 4200" >&2; exit 0;;
            quits) mkdir -p "$out"; echo "weather spin-up: 5000/10000 steps" >&2; exit 0;;
          esac
          mkdir -p "$out"
          echo '{"status": "running", "play_build": "planet", "parent": "'"$init"'"}' > "$out/meta.json"
          echo "houses: rebuilding the exact houses of $init at step 30000" >&2
          echo "weather spin-up: 5000/10000 steps" >&2
          if [ "$playmode" = slow ]; then
            i=0
            while [ $stopped = 0 ] && [ $i -lt 900 ]; do sleep 0.1; i=$((i+1)); done
            echo '{"status": "interrupted", "play_build": "planet", "parent": "'"$init"'"}' > "$out/meta.json"
            exit 130
          fi
          mkworld "$out" running 30000
          echo '{"play_version": 1, "base_rate": 14.4, "cadence": {"metrics_every": 250, "heavy_metrics_every": 250, "checkpoint_every": 6000, "keep_checkpoints": 3}, "tick": 25, "weather": {"dc_ref": 0.5}, "planet": {"source": "'"$init"':30000", "step": 30000, "weather_spinup": 10000, "houses": "auto", "made_at_step": 30000}}' > "$out/play.json"
          cat "\(fixture)"
          while IFS= read -r line; do
            if [ "$line" = "quit" ]; then exit 0; fi
          done
          exit 0;;
        esac
        exit 0
        """
    }
}

/// Polls `condition` on the main actor every 50 ms until it holds or `timeout` passes. The pipeline's tests wait on real processes, often for
/// seconds; a poll every 10 ms (`eventually`) from each of them adds up when hundreds of main-actor tests run at once in one process, and
/// the timing tests of other suites then lag.
@MainActor
func patiently(timeout: Duration = .seconds(120), _ condition: @MainActor () -> Bool) async -> Bool {
    let deadline = ContinuousClock.now + timeout
    while !condition() {
        if ContinuousClock.now > deadline { return false }
        try? await Task.sleep(for: .milliseconds(50))
    }
    return true
}
