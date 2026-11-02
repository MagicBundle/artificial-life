import AlifeCore
import AppKit
import Combine
import Foundation

/// Runs the pipeline of docs/APP.md §3.2: it starts the processes, tells the pure state machine (`PipelinePlan`) what they do, and does what
/// the machine asks. One pipeline at a time. It owns no process itself: stage 1 is the `RunController`'s run job, stage 2 the
/// `PlayController`'s world. Nothing it starts is invisible: stage 2 starts by itself only when `hooks.isLooking()` says the person can see
/// it (the window is shown and the app is in front), and a Stop never continues by itself.
@MainActor
final class PipelineCoordinator: ObservableObject {
    /// What the window around the pipeline provides. The tests give their own.
    @MainActor struct Hooks {
        /// The main window or the set-up window is shown and the app is in front, and the app is not quitting.
        var isLooking: () -> Bool = { PipelineCoordinator.windowIsLooking() }
        /// The Dock icon bounces once (only when the app is not in front).
        var bounce: () -> Void = { if !NSApp.isActive { NSApp.requestUserAttention(.informationalRequest) } }
        /// The player started a world: when it is live, the window shows it.
        var playerStartsWorld: () -> Void = {}
        var showActivity: () -> Void = {}
        var showWatch: () -> Void = {}
        /// Opens the set-up page on an intent, with a name to suggest.
        var openSetup: (SetupIntent, String?) -> Void = { _, _ in }
        /// The planet's failure is told by the pipeline, not also by the Worlds page.
        var dismissStartFailure: () -> Void = {}
        var now: () -> Date = { Date() }
    }

    @Published private(set) var plan: PipelinePlan?
    /// Seconds since the current stage started, for the banner's clock.
    @Published private(set) var elapsed: Double = 0
    /// The stop-after timer's seconds left, while the run that has one grows (the bar shows it wherever the run's progress is shown).
    @Published private(set) var timerRemaining: TimeInterval?
    /// The plan and the second stage's spec of the pipeline that is under way or just over (kept so that "Make again" can use them).
    private(set) var setup: SetupPlan?
    private(set) var spec: PlanetSpec?

    let controller: RunController
    let play: PlayController
    let launcher: WorldLauncherModel
    var hooks: Hooks
    /// `<alife folder>/runs`, nil without a usable folder.
    var runsDirectory: () -> URL?
    /// The app is quitting (the quit waits for what runs): nothing is started then, whatever the windows say.
    var isQuitting: @MainActor () -> Bool = { AppDelegate.current?.terminationGate.isWaiting == true }
    /// What other processes hold or seem to write, from the library (names), for the guard.
    var elsewhere: () -> Set<String> = { [] }

    private var cancellables: Set<AnyCancellable> = []
    private var stageStartedAt = Date()
    private var seenExit = 0
    private var ticker: Task<Void, Never>?

    init(controller: RunController, play: PlayController, launcher: WorldLauncherModel, hooks: Hooks = Hooks(),
         runsDirectory: @escaping () -> URL?) {
        self.controller = controller
        self.play = play
        self.launcher = launcher
        self.hooks = hooks
        self.runsDirectory = runsDirectory
        seenExit = controller.lastRunExit?.serial ?? 0
        controller.$lastRunExit.dropFirst().sink { [weak self] exit in
            MainActor.assumeIsolated { if let exit { self?.runEnded(exit) } }
        }.store(in: &cancellables)
        controller.$metrics.sink { [weak self] metrics in
            MainActor.assumeIsolated { self?.metricsChanged(metrics) }
        }.store(in: &cancellables)
        controller.$stopRequested.sink { [weak self] stopping in
            MainActor.assumeIsolated { if stopping { self?.stopWasRequested() } }
        }.store(in: &cancellables)
        play.$phase.sink { [weak self] phase in
            MainActor.assumeIsolated { self?.playPhaseChanged(phase) }
        }.store(in: &cancellables)
        play.$progress.sink { [weak self] lines in
            MainActor.assumeIsolated { self?.playProgressChanged(lines) }
        }.store(in: &cancellables)
    }

    deinit { ticker?.cancel() }

    // MARK: What the window shows

    var banner: PipelinePlan.Banner? { plan?.banner(elapsed: elapsed) }

    /// A pipeline is under way (a process runs or it waits for a button): the set-up page shows its progress instead of the form.
    var isActive: Bool { plan?.isActive ?? false }

    /// The planets the library lists before they are worlds.
    var pendingPlanets: [PendingPlanet] {
        guard let plan, plan.mode == .growThenPlay, let seed = plan.growName else { return [] }
        switch plan.state {
        case .growing: return [PendingPlanet(name: plan.name, seed: seed, stage: .growing, since: startedAt)]
        case .making: return [PendingPlanet(name: plan.name, seed: seed, stage: .making, since: startedAt)]
        default: return []
        }
    }
    private var startedAt = Date()

    /// Names this pipeline holds, for the taken names of the form (a second Create cannot take them).
    var reservedNames: Set<String> {
        guard let plan, plan.isActive else { return [] }
        return Set(plan.directories)
    }

    /// The guard as it is now: what this app writes (the pipeline's directories, the open world, the worlds still saving) and what other
    /// processes hold.
    func currentGuard() -> DirectoryGuard {
        var g = DirectoryGuard()
        g.heldElsewhere = elsewhere()
        if play.isActive, let world = play.world { g.record(world, as: .playing) }
        for world in play.retiredWorlds { g.record(world, as: .saving) }
        if let plan, plan.isActive {
            switch plan.state {
            case .growing:
                if let grow = plan.growName { g.record(grow, as: .growing) }
                if plan.mode == .growThenPlay { g.record(plan.name, as: .makingPlanet) }
            case .parked:
                // (waiting for a button: nothing is written, but the names stay promised so that nothing else takes them)
                if plan.mode == .growThenPlay { g.record(plan.name, as: .makingPlanet) }
            case .making:
                g.record(plan.name, as: .makingPlanet)
            default: break
            }
        }
        return g
    }

    // MARK: Starting

    struct Timing {
        var targetSteps: Int?
        var startStep = 0
        var expectedPlanetSeconds: Double?
        var offersPlanet = true
    }

    /// Starts the plan. Returns the sentence that says why it did not start (nothing is then changed). The form is `controller.settings`, as
    /// the Lab's was; `planet` is the spec of the second stage (or the only one, for Make planet).
    func begin(_ setup: SetupPlan, planet: PlanetSpec?, timing: Timing) -> String? {
        if let plan, plan.isActive { return "Another world is being made. Wait for it, or stop it." }
        guard controller.repoReady else { return "Set up the engine first." }
        let g = currentGuard()
        for stage in setup.stages {
            if let why = g.writeRefusal(stage.world) { return why }
        }
        guard let first = setup.stages.first else { return "Nothing to start." }
        if first.kind != .makePlanet, controller.job != nil { return "A run is going. Wait for it to finish, or stop it." }
        if first.kind == .makePlanet, let open = play.isActive ? play.world : nil { return "Close \(open) first: a new planet replaces the open world." }
        // (what the first stage reads must not be written by a world that is open or still saving: its newest save point is not final)
        if let reads = first.reads, first.kind != .keepGrowing, let (_, writer) = g.writer(of: reads), writer == .playing || writer == .saving {
            return writer == .playing ? "Close \(reads) first: it is being written." : "\(reads) is still saving."
        }

        var p: PipelinePlan
        switch first.kind {
        case .grow, .branch:
            p = PipelinePlan(mode: setup.pipelineMode, name: setup.name, growName: first.world, targetSteps: timing.targetSteps,
                             startStep: timing.startStep, expectedPlanetSeconds: timing.expectedPlanetSeconds)
        case .keepGrowing:
            p = PipelinePlan(mode: .keepGrowing, name: setup.name, growName: first.world, targetSteps: timing.targetSteps,
                             startStep: timing.startStep)
        case .makePlanet:
            p = PipelinePlan(mode: .makePlanet, name: setup.name, growName: nil, source: first.reads, targetSteps: nil,
                             expectedPlanetSeconds: timing.expectedPlanetSeconds)
        }
        p.offersPlanet = timing.offersPlanet
        self.setup = setup
        self.spec = planet
        startedAt = hooks.now()

        if first.kind == .makePlanet {
            guard let planet else { return "Nothing to start." }
            plan = p
            stageStartedAt = hooks.now()
            elapsed = 0
            hooks.playerStartsWorld()
            launcher.start(spec: planet)
            startTicker()
            return nil
        }
        // Stage 1: a research run, from the form the page filled in. The command that runs is the command the page showed.
        controller.refreshRuns()
        if controller.settings.mode != .resume { controller.settings.name = first.world }
        let command = controller.settings.arguments(existingRuns: Set(controller.runs))
        guard command == first.arguments else {
            return "The command changed while it was being started. Press the button again."
        }
        controller.start()
        guard controller.job == .run else {
            return controller.startBlockedReason ?? "The engine could not be started. Activity has what it said."
        }
        seenExit = controller.lastRunExit?.serial ?? seenExit
        plan = p
        stageStartedAt = hooks.now()
        elapsed = 0
        startTicker()
        return nil
    }

    // MARK: Buttons

    func press(_ button: PipelinePlan.Button) {
        guard let p = plan else { return }
        switch button {
        case .stop: apply(.stopPressed)
        case .cancel: apply(.cancelPressed)
        case .makePlanet, .makePlanetFromHere:
            if case .parked = p.state {
                apply(.makePlanetPressed)
            } else if let world = p.growName ?? p.source {
                hooks.openSetup(.makePlanet(from: world), p.name)
            }
        case .keepGrowing:
            if case .parked = p.state {
                apply(.keepGrowingPressed)
            } else if let world = p.growName {
                hooks.openSetup(.keepGrowing(world: world), nil)
            }
        case .done: dismiss()
        case .showLog, .showActivity: hooks.showActivity()
        case .watch: hooks.showWatch()
        case .tryAgainNewName: hooks.openSetup(setup?.intent ?? .create, nil)
        case .makeAgainNewName:
            guard let world = p.planetSource else { return }
            // A plan that waits for a button is still active, and the page then shows its bar and no form: "make again" ends it first.
            if p.isActive { dismiss() }
            hooks.openSetup(.makePlanet(from: world), p.name)
        }
    }

    /// Done: the bar goes, whatever state the plan is in (a parked plan is left as it is on disk).
    private func dismiss() {
        _ = plan?.apply(.donePressed, in: surroundings())
        plan = nil
        stopTicker()
    }

    // MARK: Events from the processes

    private func surroundings() -> PipelinePlan.Surroundings {
        let name = plan?.name ?? ""
        return PipelinePlan.Surroundings(
            visible: hooks.isLooking() && !isQuitting(), openWorld: play.isActive ? play.world : nil, nameTaken: plan?.mode == .keepGrowing ? false : nameTakenNow(name),
            engineReady: controller.repoReady)
    }

    /// Whether a folder with this name (or one that differs only in capitals) is under `runs/` right now, looked at on disk: the library may
    /// not have scanned yet, and a link to a drive that is not connected still takes its name.
    func nameTakenNow(_ name: String) -> Bool {
        guard let runs = runsDirectory(), !name.isEmpty else { return false }
        let names = (try? FileManager.default.contentsOfDirectory(atPath: runs.path)) ?? []
        return NameFolding.clash(of: name, in: names) != nil
    }

    /// The one way an event reaches the machine: the new state is stored first, then the effects are done (an effect can make a process
    /// report at once, and that report must find the new state).
    private func apply(_ event: PipelinePlan.Event, around: PipelinePlan.Surroundings? = nil) {
        guard var p = plan else { return }
        let effects = p.apply(event, in: around ?? surroundings())
        if p != plan { plan = p }
        for effect in effects { perform(effect) }
        if plan?.isOver == true { stopTicker() }
    }

    /// A run that was started by this app has ended (exposed to the tests, which feed it exits that the real process would not).
    func runEnded(_ exit: RunExit) {
        guard exit.serial > seenExit else { return }
        seenExit = exit.serial
        guard let p = plan, case .growing = p.state else { return }
        // A run that ends under another name than the one the page showed must never leave the pipeline waiting for an exit that already came.
        guard NameFolding.same(exit.run, p.growName ?? "") else {
            plan?.state = .failed(.growing("The engine's run ended under another name (\(exit.run)) than the one that was asked for (\(p.growName ?? "")). Nothing was lost: look in Worlds."))
            stopTicker()
            return
        }
        var savePoint = true
        var savedStep: Int?
        var worldStatus: String?
        if let grow = p.growName, let runs = runsDirectory() {
            let directory = runs.appendingPathComponent(grow)
            let saved = PlayCatalogScanner.checkpointSteps(in: directory, fileManager: .default)
            savePoint = !saved.isEmpty
            savedStep = saved.max()
            worldStatus = Self.worldStatus(in: directory)
        }
        apply(.growExited(.init(
            status: exit.status, signalled: exit.signalled, stopRequested: exit.stopRequested && !exit.forced, problem: exit.problem,
            savePoint: savePoint, savedStep: savedStep, worldStatus: worldStatus)))
    }

    /// The `status` the engine wrote in a run's `meta.json` (`done`, `extinct`, `interrupted`, `violation`); nil when it cannot be read.
    static func worldStatus(in directory: URL) -> String? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("meta.json")),
              let meta = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return meta["status"] as? String
    }

    private func metricsChanged(_ metrics: MetricsSnapshot?) {
        guard let p = plan, case .growing = p.state, controller.runningRunName == p.growName, let m = metrics else { return }
        apply(.progress(step: m.step, stepsPerSecond: m.stepsPerSecond), around: PipelinePlan.Surroundings())
    }

    private func stopWasRequested() {
        guard let p = plan, case .growing(false) = p.state, controller.runningRunName == p.growName else { return }
        // (the run has been asked already: the machine's `.stopGrowing` effect finds `stop()` guarded)
        apply(.stopPressed, around: PipelinePlan.Surroundings())
    }

    private func playPhaseChanged(_ phase: PlayPhase) {
        guard let p = plan, case .making = p.state, NameFolding.same(play.world ?? "", p.name) else { return }
        switch phase {
        case .live: apply(.makeLive, around: PipelinePlan.Surroundings())
        case .failed(let message):
            // A world that died out ends `alife play` with exit 0 before its header and a line on stderr: that line is the reason.
            if let line = play.progress.last(where: { PipelineWords.diedOutStep(in: $0) != nil }) {
                apply(.makeEnded(line), around: PipelinePlan.Surroundings())
            } else {
                apply(.makeFailed(PipelineWords.planetFailure(message, said: PipelineWords.lastSaid(in: play.progress))), around: PipelinePlan.Surroundings())
            }
            // The Worlds page and the Watch row are told of a world that did not open by the shell's own observer of the same phase, which may
            // run after this one: the pipeline says it once, in its bar, so the shell's note is taken back on the next turn.
            Task { @MainActor [weak self] in self?.hooks.dismissStartFailure() }
        case .idle: apply(.makeEnded(nil), around: PipelinePlan.Surroundings())
        case .ended(let message):
            let line = play.progress.last(where: { PipelineWords.diedOutStep(in: $0) != nil })
            apply(.makeEnded(line ?? PipelineWords.planetEnded(message)), around: PipelinePlan.Surroundings())
        case .starting: return
        }
    }

    private func playProgressChanged(_ lines: [String]) {
        guard let p = plan, case .making = p.state, NameFolding.same(play.world ?? "", p.name) else { return }
        apply(.makeProgress(lines), around: PipelinePlan.Surroundings())
    }

    // MARK: Effects

    private func perform(_ effect: PipelinePlan.Effect) {
        switch effect {
        case .stopGrowing:
            controller.stop()
        case .stopMaking:
            play.stop()
        case .bounce:
            hooks.bounce()
        case .startMaking:
            guard let spec else {
                plan?.state = .failed(.making("The planet's settings were lost. Make it again from the grown world."))
                return
            }
            stageStartedAt = hooks.now()
            elapsed = 0
            hooks.playerStartsWorld()
            launcher.start(spec: spec)
        case .resumeGrowing:
            guard let grow = plan?.growName else { return }
            controller.refreshRuns()
            controller.settings.mode = .resume
            controller.settings.sourceRun = grow
            controller.settings.steps = ""
            controller.loadDirector()
            controller.start()
            if controller.job == .run {
                stageStartedAt = hooks.now()
                elapsed = 0
                seenExit = controller.lastRunExit?.serial ?? seenExit
            } else {
                plan?.state = .failed(.growing(controller.startBlockedReason ?? "The engine could not be started. Activity has what it said."))
            }
        }
    }

    // MARK: The clock

    private func startTicker() {
        ticker?.cancel()
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                guard let self, !Task.isCancelled else { return }
                self.elapsed = self.hooks.now().timeIntervalSince(self.stageStartedAt)
                if case .growing? = self.plan?.state { self.timerRemaining = self.controller.remaining } else if self.timerRemaining != nil { self.timerRemaining = nil }
            }
        }
    }

    private func stopTicker() {
        ticker?.cancel()
        ticker = nil
    }

    // MARK: The window

    /// The main window or the set-up window is on screen, not hidden behind other windows or minimised, and the app is in front.
    static func windowIsLooking() -> Bool {
        guard NSApp.isActive else { return false }
        return NSApp.windows.contains { window in
            let id = window.identifier?.rawValue
            return (id == AppShell.mainWindowID || id == AppShell.setupWindowID) && window.isVisible && !window.isMiniaturized
                && window.occlusionState.contains(.visible)
        }
    }
}
