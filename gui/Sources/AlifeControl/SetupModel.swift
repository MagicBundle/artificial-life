import AlifeCore
import Combine
import Foundation

/// The set-up page's state (docs/APP.md §3.2): what the person is doing (Create, Branch, Keep growing, Make planet), the starting point, the
/// goal, the name, and everything behind Advanced. The form is `RunController.settings`, as the Lab's was, so the command lines are the
/// existing builders'; the planet's form is the launcher's `PlanetSpec`. The page asks this model why the button is grey, what changed from
/// the starting point, and what command each stage would run.
@MainActor
final class SetupModel: ObservableObject {
    let controller: RunController
    let play: PlayController
    let launcher: WorldLauncherModel
    let coordinator: PipelineCoordinator
    let library: WorldLibrary
    private let defaults: UserDefaults

    @Published private(set) var intent: SetupIntent = .create
    @Published private(set) var startingPoint: StartingPoint = .initial
    @Published var goal: Goal = .play
    @Published var name = ""
    /// The steps to grow (`--steps`), as typed. Always a number for Create (§3.2: there is no "until I stop").
    @Published var growSteps = ""
    /// Advanced: make a world that takes no powers (`director.enabled` off); how the family lines of a planet are rebuilt.
    @Published var withoutPowers = false
    @Published var houses = PlayHouses.auto
    /// The preset guide of the alife folder (cards, speeds), loaded by the page.
    @Published var guide = PresetGuideContent.empty
    /// The keys set from memory when the form was filled, and the ones that did not apply to this world.
    @Published private(set) var rememberedKeys: Set<String> = []
    @Published private(set) var rememberedSkipped: [String] = []
    /// What the last press of the button was told (a sentence), until the form changes.
    @Published private(set) var refusal: String?
    /// Asks the shell to show the set-up page (a card's button, File > New World).
    var show: () -> Void = {}
    /// The sheet that stops the open world before a planet is made (R5); without one a planet is refused while a world is open.
    var switcher: WorldSwitcher?

    private var remembered = RememberedLevers()
    private var forwarding: AnyCancellable?

    /// The set-up model of the alife folder a `RunController` is set to, with the pipeline that runs what it starts.
    static func forApp(controller: RunController, play: PlayController, library: WorldLibrary, defaults: UserDefaults = .standard) -> SetupModel {
        let launcher = WorldLauncherModel.forApp(run: controller, play: play)
        let coordinator = PipelineCoordinator(
            controller: controller, play: play, launcher: launcher,
            runsDirectory: { controller.repoReady ? controller.repoURL.appendingPathComponent("runs") : nil })
        return SetupModel(controller: controller, play: play, launcher: launcher, coordinator: coordinator, library: library, defaults: defaults)
    }

    init(controller: RunController, play: PlayController, launcher: WorldLauncherModel, coordinator: PipelineCoordinator, library: WorldLibrary,
         defaults: UserDefaults = .standard) {
        self.controller = controller
        self.play = play
        self.launcher = launcher
        self.coordinator = coordinator
        self.library = library
        self.defaults = defaults
        remembered = RememberedLevers(data: defaults.data(forKey: RememberedLevers.defaultsKey))
        // The page and the window read the pipeline through this model too.
        forwarding = coordinator.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
        applySuggestion()
    }

    // MARK: Opening

    /// While a world is being made the page shows its bar: a new form is not opened under it (the buttons that lead here are grey then).
    private var pipelineBusy: Bool { coordinator.isActive }

    func openCreate(_ point: StartingPoint? = nil) {
        guard !pipelineBusy else { return show() }
        prepareCreate(point)
        requested = true
        show()
    }

    private func prepareCreate(_ point: StartingPoint? = nil) {
        prepared = true
        intent = .create
        refusal = nil
        let chosen = point ?? startingPoint
        setMode(.new)
        choose(chosen, applyMemory: true)
        applySuggestion()
    }

    /// The page appeared (the sidebar's row, "Open in window", a return from another page, or the page restored at launch). A form that is
    /// half filled in is never thrown away by that: Branch, Keep growing and Make planet stay as they are until the person asks for a new world
    /// (File > New World, the Worlds page's button, or the form's own "New world instead"); only a first appearance prepares a Create.
    func pageAppeared() {
        requested = false
        if !prepared { prepareCreate() }
    }
    private var prepared = false
    private var requested = false

    func openBranch(from world: String) {
        guard !pipelineBusy else { return show() }
        prepared = true
        requested = true
        intent = .branch(from: world)
        refusal = nil
        goal = .research
        let controller = controller
        controller.refreshRuns()
        controller.settings.sourceRun = world
        setMode(.fork)
        controller.settings.seed = ""
        controller.settings.steps = ""
        growSteps = ""
        controller.loadForkLevers()
        controller.loadDirector()
        // (a branch starts from its source's own values: what was remembered from a Create is not carried into it)
        rememberedKeys = []
        rememberedSkipped = []
        applySuggestion()
        show()
    }

    func openKeepGrowing(world: String) {
        guard !pipelineBusy else { return show() }
        prepared = true
        requested = true
        intent = .keepGrowing(world: world)
        refusal = nil
        goal = .research
        controller.refreshRuns()
        controller.settings.sourceRun = world
        setMode(.resume)
        controller.settings.steps = ""
        growSteps = ""
        controller.loadDirector()
        rememberedKeys = []
        name = world
        show()
    }

    /// Make planet from a world (and a save point, when the card or an unfinished build says which).
    func openMakePlanet(from world: String?, step: Int? = nil, name suggested: String? = nil) {
        guard !pipelineBusy else { return show() }
        prepared = true
        requested = true
        intent = .makePlanet(from: world ?? launcher.planet.source)
        refusal = nil
        goal = .play
        launcher.prepareForm(from: world)
        if let step, launcher.selectedSource?.checkpoints.contains(step) == true { launcher.selectCheckpoint(step) }
        withoutPowers = !launcher.planet.directorEnabled
        if let suggested {
            name = launcher.freeName(after: suggested)
            autoName = name
        } else {
            applySuggestion()
        }
        rememberedKeys = []
        show()
    }

    /// An unfinished build made again under a new name: the form of this session's own request, else the source and save point `meta.json` names.
    func openMakeAgain(_ build: UnfinishedRow) {
        guard !pipelineBusy else { return show() }
        prepared = true
        requested = true
        launcher.prepareAgain(build)
        let source = launcher.planet.source
        intent = .makePlanet(from: source)
        refusal = nil
        goal = .play
        withoutPowers = !launcher.planet.directorEnabled
        houses = launcher.planet.houses
        name = launcher.planet.name
        rememberedKeys = []
        show()
    }

    // MARK: The form

    private func setMode(_ new: RunMode) {
        let old = controller.settings.mode
        guard old != new else { return }
        controller.settings.mode = new
        if new == .fork {
            controller.settings.seed = ""
            controller.loadForkLevers()
        }
        if old == .fork { controller.resetLevers() }
        if new == .new { controller.refreshPresetWorld3D() }
        if new == .new, controller.settings.seed.isEmpty { controller.settings.seed = "1" }
        controller.loadDirector()
        controller.refreshTargetRunDims()
    }

    /// Chooses what the world is made from: its preset, the steps to grow, the goal it allows.
    func choose(_ point: StartingPoint, applyMemory memory: Bool = false) {
        startingPoint = point
        refusal = nil
        let c = controller
        if c.settings.mode == .new {
            c.settings.preset = point.preset
            c.refreshDirectorCadence()
            c.refreshPresetWorld3D()
            c.settings.levers = Levers(world3D: c.settings.presetWorld3D, features: c.settings.featureBaseline)
        }
        growSteps = defaultSteps(for: point)
        c.settings.steps = growSteps
        if !goalsAllowed.contains(goal) { goal = goalsAllowed.first ?? .research }
        if memory { applyMemory() } else { rememberedKeys = [] }
        // A name nobody typed follows the starting point (`flat-1` for a flat world, `planet-1` for a planet).
        if intent == .create, autoName == nil || name == autoName { applySuggestion() }
    }

    /// The steps the starting point grows by default: the recipe's, else the preset file's own `run.steps`.
    func defaultSteps(for point: StartingPoint) -> String {
        if let recipe = point.recipe { return String(recipe.growSteps) }
        let file = PresetFile.preset(point.preset.rawValue, repo: controller.repoURL)
        return file?.int("run", "steps").map(String.init) ?? String(Self.engineDefaultSteps)
    }

    /// The length the engine grows a world for when neither the preset nor the command says (`Config.run.steps`, `config.py`).
    static let engineDefaultSteps = 100_000

    /// Whose number "Grow for" starts at, in a few words: a recipe's own, the preset file's own, or the engine's default.
    var stepsOrigin: String {
        if startingPoint.recipe != nil { return "This is the recipe's own length." }
        let file = PresetFile.preset(startingPoint.preset.rawValue, repo: controller.repoURL)
        return file?.int("run", "steps") != nil ? "This is the preset's own length." : "This is the engine's usual length, not the preset's own."
    }

    func setGrowSteps(_ text: String) {
        growSteps = text
        controller.settings.steps = text
    }

    /// What the world can be for. A flat world has no weather (§3.2, M16), and a planet of a shape the weather is not calibrated for has none.
    var goalsAllowed: [Goal] {
        if let recipe = startingPoint.recipe { return recipe.goals }
        let w = controller.settings.presetWorld3D
        guard w.dims == 3 else { return [.research] }
        if w.sphere.isSphere, PlanetSpec.overlay(forShape: planetShape(of: w)) == nil { return [.research] }
        return [.play, .research]
    }

    var goalReason: String? {
        guard goalsAllowed == [.research] else { return nil }
        if controller.settings.presetWorld3D.dims != 3 { return "A flat world has no weather, so it is for research only." }
        return "The weather is not calibrated for this planet's size, so it is for research only."
    }

    private func planetShape(of w: World3DLevers) -> PlanetShape? {
        guard w.sphere.isSphere, let r = Parse.decimal(w.sphere.radius), let h = Parse.decimal(w.height),
              let n = Parse.integer(w.sphere.nside), let l = Parse.integer(w.sphere.layers) else { return nil }
        return PlanetShape(radius: r, height: h, nside: n, layers: l)
    }

    // MARK: Remembered values

    private func applyMemory() {
        let applied = remembered.apply(to: &controller.settings)
        rememberedKeys = Set(applied.keys)
        rememberedSkipped = applied.skipped
    }

    /// The remembered values that are still in the form as they were remembered (a chip on each row).
    var rememberedShown: [SettingChange] {
        ChangeList.levers(controller.settings).filter { change in
            rememberedKeys.contains(change.key)
                && LeverKeys.same(change.key, remembered.values[change.key] ?? "", LeverKeys.text(change.key, in: controller.settings.effectiveLevers) ?? "")
        }
    }

    func isRemembered(_ key: String) -> Bool { rememberedShown.contains { $0.key == key } }

    /// Puts every remembered value back to what it was before it was remembered, and forgets them.
    func resetRemembered() {
        let base = LeverKeys.baseline(of: controller.settings)
        for key in rememberedKeys {
            if let text = LeverKeys.text(key, in: base) { LeverKeys.set(key, to: text, in: &controller.settings.levers) }
        }
        rememberedKeys = []
        rememberedSkipped = []
        remembered = RememberedLevers()
        defaults.removeObject(forKey: RememberedLevers.defaultsKey)
    }

    private func remember() {
        guard intent == .create else { return }
        remembered = RememberedLevers.capture(controller.settings)
        if remembered.isEmpty { defaults.removeObject(forKey: RememberedLevers.defaultsKey) } else { defaults.set(remembered.data, forKey: RememberedLevers.defaultsKey) }
    }

    func clearRefusal() { if refusal != nil { refusal = nil } }

    /// Make planet: another world to start from (its newest save point; the weather and the name follow it).
    func chooseSource(_ world: String) {
        launcher.selectSource(world)
        intent = .makePlanet(from: world)
        withoutPowers = !launcher.planet.directorEnabled
        clearRefusal()
    }

    /// What a planet source means for the new planet: it keeps its shape, and its weather has the overlay its shape calls for.
    static func shapeNote(_ source: PlaySource) -> String {
        let keeps = "The new planet keeps this world's shape, a sphere"
        guard !source.weatherOn, PlanetSpec.overlay(forShape: source.planetShape) != nil else { return keeps + "." }
        return keeps + ", with the weather ready for that size."
    }

    /// The plain message under a lever's row, by engine key.
    func fieldError(_ key: String) -> String? {
        var form = controller.settings
        form.name = name.trimmingCharacters(in: .whitespaces)
        for message in form.validate(existingRuns: library.state.catalog.existingNames) where SetupMessages.key(of: message) == key {
            return SetupMessages.plain(message)
        }
        return nil
    }

    // MARK: Names

    /// Every name that is not free: worlds, builds, links to drives that are not connected, entries that cannot be read, and what the
    /// pipeline in progress has reserved.
    var taken: Set<String> { library.state.catalog.existingNames.union(coordinator.reservedNames) }

    func suggestedName() -> String {
        let base: String
        switch intent {
        case .create: base = startingPoint.recipe?.nameBase ?? "world"
        case .branch(let from): base = from + "-branch"
        case .keepGrowing(let w): return w
        case .makePlanet: base = "planet"
        }
        return SetupPlan.suggestName(base: base, goal: seedMatters ? .play : .research, taken: taken)
    }

    private var seedMatters: Bool { intent == .create && goal == .play }

    func suggestNewName() {
        applySuggestion()
        refusal = nil
    }

    /// The name this model made up, while nobody has typed over it: a library that arrives or changes after the form was made (a launch
    /// opens on this page before the first scan has finished) may take it, and then the suggestion is made again.
    private var autoName: String?

    private func applySuggestion() {
        let made = suggestedName()
        name = made
        autoName = made
    }

    /// Called when the library's worlds change: an untouched suggestion that is no longer free is replaced; a typed name is never touched.
    func libraryChanged() {
        guard let auto = autoName, name == auto, !nameProblems.isEmpty else { return }
        applySuggestion()
    }

    /// The first free name after the typed one, for the "Use ..." button beside a taken name.
    var freeAlternative: String? {
        guard !nameProblems.isEmpty else { return nil }
        let typed = name.trimmingCharacters(in: .whitespaces)
        let base = typed.isEmpty ? "world" : typed
        let free = SetupPlan.suggestName(base: base.range(of: #"-\d+$"#, options: .regularExpression).map { String(base[..<$0.lowerBound]) } ?? base,
                                         goal: seedMatters ? .play : .research, taken: taken)
        return SetupPlan.nameProblems(free, goal: goal, intent: intent, taken: taken).isEmpty ? free : nil
    }

    var nameProblems: [String] {
        SetupPlan.nameProblems(name, goal: goal, intent: intent, taken: taken)
    }

    // MARK: Why the button is grey

    /// The row of the library for the world the intent starts from.
    var sourceRow: LibraryRow? { intent.world.flatMap { library.state.row(named: $0) } }

    var ruleFacts: SetupRules.Facts {
        SetupRules.Facts(
            runJob: controller.hasRunJob, openWorld: play.isActive ? play.world : nil, guardState: coordinator.currentGuard(),
            folderExists: controller.repoFolderExists, engineReady: controller.repoReady,
            folderProblem: controller.repoFolderExists ? nil : "The alife folder was not found.", pipeline: pipelineActivity)
    }

    /// What the pipeline is doing, for the rules: running, waiting for a button, or nothing.
    var pipelineActivity: SetupRules.Facts.Pipeline {
        guard let plan = coordinator.plan, plan.isActive else { return .none }
        if case .parked = plan.state { return .waiting }
        return .running
    }

    /// Everything that keeps the button grey, in the page's words. Empty when the button works.
    /// Why a planet cannot have lighter brains: `--overlay fast` sets `sensing.cell_capacity` 12, and a planet's hash cells need more
    /// (`config.validate`: 30 at `mini3s`), so the engine refuses the run. Nil where the option works.
    var lighterBrainsObstacle: String? {
        controller.settings.presetWorld3D.sphere.isSphere ? "Not available on a planet: a planet needs the full neighbour search." : nil
    }

    /// A remembered or earlier choice must not ride into a planet, where the engine would refuse it.
    func dropLighterBrainsIfRefused() {
        if lighterBrainsObstacle != nil, controller.settings.fast { controller.settings.fast = false }
    }

    var reasons: [String] {
        var out: [String] = []
        func add(_ text: String?) { if let text, !out.contains(text) { out.append(text) } }
        if !controller.repoFolderExists { add("The alife folder was not found. Choose it in Settings.") }
        else if !controller.repoReady { add("Set up the engine first.") }
        if let busy = SetupRules.pipelineReason(pipelineActivity) { add(busy) }
        else if controller.job == .setup { add("The engine is being set up. Wait for it to finish.") }
        else if controller.job == .inspect { add("A check is running. Wait for it to finish.") }
        else if controller.job == .run { add("A run is going. Wait for it to finish, or stop it in Activity.") }
        guard controller.repoReady else { return out }
        switch intent {
        case .create:
            nameProblems.forEach { add($0) }
            runSettingMessages.forEach { add($0) }
            if Parse.integer(growSteps).map({ $0 >= 1 }) != true { add("Grow for must be a whole number of steps, 1 or more.") }
            if goal == .play, let obstacle = weatherObstacle { add(obstacle) }
        case .branch:
            if let row = sourceRow { add(SetupRules.reason(.branch, on: row, in: ruleFacts)) } else { add("That world is no longer here.") }
            nameProblems.forEach { add($0) }
            runSettingMessages.forEach { add($0) }
        case .keepGrowing:
            if let row = sourceRow { add(SetupRules.reason(.keepGrowing, on: row, in: ruleFacts)) } else { add("That world is no longer here.") }
            runSettingMessages.forEach { add($0) }
        case .makePlanet:
            if let row = sourceRow { add(SetupRules.reason(.makePlanet, on: row, in: ruleFacts)) }
            else if launcher.catalog.sources.isEmpty { add("A planet is made from a world with a save point, and there is none yet. Create one first.") }
            planetMessages.forEach { add($0) }
        }
        return out
    }

    /// The planet form's own checks (the launcher's spec as the page edits it), in the page's words, and the pipeline's reserved names.
    private var planetMessages: [String] {
        let spec = makePlanetSpec
        let taken = self.taken
        var messages = spec.validate(existing: taken).map(WorldsGuide.plain)
        let asked = spec.name.trimmingCharacters(in: .whitespaces)
        if !asked.isEmpty, !taken.contains(asked), let clash = NameFolding.clash(of: asked, in: taken) {
            messages.append("A world called \u{201C}\(clash)\u{201D} is already here, and names that differ only in capitals count as the same.")
        }
        return messages
    }

    /// The launcher's planet with the page's name, powers and family lines.
    var makePlanetSpec: PlanetSpec {
        var spec = launcher.planet
        spec.name = name.trimmingCharacters(in: .whitespaces)
        spec.directorEnabled = !withoutPowers
        spec.houses = houses
        return spec
    }

    /// The engine's checks of the run form in the page's words (the name is told separately).
    private var runSettingMessages: [String] {
        var form = controller.settings
        form.name = name.trimmingCharacters(in: .whitespaces)
        let messages = form.validate(existingRuns: library.state.catalog.existingNames)
        return messages.filter { !$0.hasPrefix("Enter a run name") && !$0.hasPrefix("Run name may use") }.map(SetupMessages.plain)
    }

    /// The reason the weather cannot be turned on for the world being set up, in the page's words (it is told before the first stage, not
    /// after two minutes of growing).
    private var weatherObstacle: String? {
        let spec = stageTwoSpec(seed: "seed")
        return PlanetSpec.weatherObstacle(dims: spec.sourceDims, geometry: spec.sourceGeometry, shape: spec.sourceShape).map(WorldsGuide.plain)
    }

    /// A note beside an allowed button (a source that is still growing, another process that may be writing).
    var note: String? {
        guard let row = sourceRow else { return nil }
        switch intent {
        case .branch: return SetupRules.note(.branch, on: row, in: ruleFacts)
        case .makePlanet: return SetupRules.note(.makePlanet, on: row, in: ruleFacts)
        default: return nil
        }
    }

    // MARK: The plan

    /// The second stage's spec for a planet that Create grows (`source` is the seed it will have).
    func stageTwoSpec(seed: String) -> PlanetSpec {
        let w = controller.settings.effectiveLevers.world3D
        let shape = planetShape(of: w)
        let size = PresetFile.preset(controller.settings.preset.rawValue, repo: controller.repoURL)?.double("world", "size")
        var spec = PlanetSpec(
            source: seed, name: name.trimmingCharacters(in: .whitespaces), storm: true,
            weatherSpinup: PlanetSpec.defaultSpinup(forGeometry: w.sphere.geometry, shape: shape, worldSize: size),
            directorEnabled: !withoutPowers, houses: houses, sourceWorldSize: size, sourceDims: w.dims, sourceGeometry: w.sphere.geometry,
            sourceShape: shape)
        spec.rate = .multiplier(1)
        return spec
    }

    var plan: SetupPlan {
        let asked = name.trimmingCharacters(in: .whitespaces)
        var form = controller.settings
        form.name = asked
        var planet: PlanetSpec?
        switch intent {
        case .create where goal == .play: planet = stageTwoSpec(seed: SetupPlan.seedName(asked))
        case .makePlanet: planet = makePlanetSpec
        default: break
        }
        return SetupPlan.make(intent: intent, goal: goal, name: asked, settings: form, planet: planet, existing: library.state.catalog.existingNames)
    }

    var commandBlocks: [CommandBlock] { plan.commandBlocks(folder: controller.repoPath) }

    var changes: [SettingChange] { ChangeList.levers(controller.settings) }
    var otherChanges: [String] {
        var out = ChangeList.others(controller.settings)
        if withoutPowers, intent != .keepGrowing(world: "") { out.append("No powers in the planet") }
        if houses != .auto, goal == .play || intent.isMakePlanet { out.append("Family lines rebuilt \(houses.plainTitle.lowercased())") }
        return out
    }

    // MARK: Time

    /// The steps per second to expect: measured for the recipe, else the preset guide's snapshot.
    var stepsPerSecond: Double? {
        if let recipe = startingPoint.recipe, let m = recipe.measured { return m.stepsPerSecond }
        return guide.entry(for: startingPoint.preset)?.snapshot?.stepsPerSecond
    }

    /// "about 1 min 50 s", with a note when the world is very short; nil when there is nothing measured to say.
    var growLine: String? {
        let start = startingPoint.recipe?.measured?.startSeconds ?? 0
        return GrowEstimate.line(steps: Parse.integer(growSteps), stepsPerSecond: stepsPerSecond, startSeconds: start)
    }

    /// The line under "Grow for" of a Create: "About 1 min 50 s on a recent Mac (measured on an M5 Pro), then about 10 s to make the planet."
    /// Nothing is said for a preset nobody measured (the page does not make a number up).
    var timeSentence: String? {
        guard intent == .create, let grow = growLine else { return nil }
        let measured = startingPoint.recipe?.measured != nil
        var line = grow.prefix(1).uppercased() + grow.dropFirst()
        // (a very short world carries its own sentence after the time)
        if let dot = line.range(of: ". ") {
            let tail = String(line[dot.upperBound...])
            line = String(line[..<dot.lowerBound]) + (measured ? " on a recent Mac (measured on an M5 Pro)" : " at the speed the preset guide measured")
            if goal == .play { line += ", then " + (planetSeconds.map { TimeWords.about($0) } ?? "a short while") + " to make the planet" }
            return line + ". " + tail + "."
        }
        line += measured ? " on a recent Mac (measured on an M5 Pro)" : " at the speed the preset guide measured"
        if goal == .play { line += ", then " + (planetSeconds.map { TimeWords.about($0) } ?? "a short while") + " to make the planet" }
        return line + "."
    }

    /// Under the time: whose length it is, and that the first world after an install is slower.
    var timeNote: String? {
        guard intent == .create else { return nil }
        return stepsOrigin + " The first world after installing alife takes longer: the engine prepares itself once."
    }

    /// Seconds from "Make planet" to the first picture, when measured.
    var planetSeconds: Double? { startingPoint.recipe?.measured?.planetSeconds }

    // MARK: Starting

    func start() {
        guard reasons.isEmpty else { return }
        dropLighterBrainsIfRefused()
        let setup = plan
        var timing = PipelineCoordinator.Timing()
        timing.expectedPlanetSeconds = planetSeconds
        timing.offersPlanet = goalsAllowed.contains(.play) || intent != .create
        let spec: PlanetSpec?
        switch intent {
        case .create:
            timing.targetSteps = Parse.integer(growSteps)
            spec = goal == .play ? stageTwoSpec(seed: SetupPlan.seedName(setup.name)) : nil
        case .branch(let from):
            let source = library.state.row(named: from)
            timing.startStep = source?.step ?? 0
            // (blank: the engine grows as many steps as the source's own run was set to, beyond the save point it starts from)
            timing.targetSteps = (Parse.integer(growSteps) ?? source?.targetSteps).map { $0.saturatingAdding(timing.startStep) }
            spec = nil
        case .keepGrowing(let world):
            let row = library.state.row(named: world)
            timing.startStep = row?.step ?? 0
            timing.targetSteps = Parse.integer(growSteps).map { $0.saturatingAdding(timing.startStep) } ?? row?.targetSteps
            timing.offersPlanet = row?.shape == .planet
            spec = nil
        case .makePlanet:
            spec = makePlanetSpec
        }
        if let switcher, switcher.isBusy {  // (the open world is being saved for another switch: its sheet comes back, the click is answered)
            switcher.showAgain()
            return
        }
        let go: @MainActor () -> Void = { [weak self] in
            guard let self else { return }
            refusal = coordinator.begin(setup, planet: spec, timing: timing)
            // What was edited is remembered only once the start went through (a refused press forgets nothing and remembers nothing).
            if refusal == nil { remember() }
        }
        // A planet replaces the open world: the sheet asks first, and the planet starts once the world is saved (R5).
        if case .makePlanet = intent, let switcher, switcher.ask(.makePlanet(setup.name), then: go) { return }
        go()
    }

    /// The same press with the name made free (the "Try again with a new name" of a failed build).
    func retryWithNewName() {
        applySuggestion()
    }
}
