import AlifeCore
import AppKit
import SwiftUI

/// The set-up page (docs/APP.md §3.2): Create, Branch, Keep growing and Make planet in one place. The first screen is a name, a starting
/// point, a goal and a length, and one button; everything else is behind Advanced, which is remembered. While a pipeline is under way the
/// page shows its progress in place of the form.
struct SetupPage: View {
    @ObservedObject var model: SetupModel
    @ObservedObject var controller: RunController
    @ObservedObject var coordinator: PipelineCoordinator
    @ObservedObject var library: WorldLibrary
    @ObservedObject var launcher: WorldLauncherModel
    /// True in the window of its own (no "Open in window" there).
    var inWindow = false
    @AppStorage("setup.advanced") private var advancedOpen = false
    @AppStorage("setup.showCommand") private var commandOpen = false
    @State private var compare = false
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                header
                if coordinator.plan != nil {
                    PipelineCard(coordinator: coordinator)
                }
                if coordinator.isActive {
                    activeNotes
                } else {
                    form
                }
            }
            .padding(24)
            .frame(maxWidth: 780, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .navigationTitle(model.intent.title)
        .toolbar {
            if !inWindow {
                ToolbarItem {
                    Button("Open in window") { openWindow(id: AppShell.setupWindowID) }
                        .help("Opens this page in a window of its own, so that you can set up a world beside the one you watch.")
                }
            }
        }
        .onAppear { model.pageAppeared() }
        .onChange(of: library.state.catalog) { _, _ in model.libraryChanged() }
        .task(id: controller.repoURL) { model.guide = PresetGuideContent.load(repo: controller.repoURL) }
        .sheet(isPresented: $compare) {
            PresetCompareSheet(content: model.guide, selected: model.startingPoint.preset) { model.choose(.preset($0), applyMemory: true) }
        }
    }

    // MARK: Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(model.intent.title).font(.largeTitle.weight(.semibold))
            if let state = coordinator.plan?.state, state == .growing(stopping: false) || state == .growing(stopping: true) || state == .making(cancelling: false) || state == .making(cancelling: true) {
                Text("alife is making this for you. You can look at other pages meanwhile; the same bar stays at the top of each of them.")
                    .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            } else if coordinator.plan == nil {
                Text(model.intent.summary).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// While something is being made: what the page does not offer, and the commands that run.
    private var activeNotes: some View {
        VStack(alignment: .leading, spacing: 10) {
            commandSection(blocks: coordinator.setup?.commandBlocks(folder: controller.repoPath) ?? [], made: coordinator.setup?.madeSentence)
        }
    }

    // MARK: The form

    @ViewBuilder private var form: some View {
        if model.intent != .create {
            Button("New world instead") { model.openCreate() }
                .buttonStyle(.link).controlSize(.small)
                .help("Leaves this form and starts a new world from a starting point.")
        }
        switch model.intent {
        case .create: createForm
        case .branch: branchForm
        case .keepGrowing: keepGrowingForm
        case .makePlanet: makePlanetForm
        }
    }

    private var createForm: some View {
        VStack(alignment: .leading, spacing: 16) {
            nameField
            startingPointSection
            goalSection
            growSection(label: "Grow for", prompt: "")
            changesSection
            advancedSection
            footer
        }
    }

    private var branchForm: some View {
        VStack(alignment: .leading, spacing: 16) {
            nameField
            if let note = model.note { noteLabel(note) }
            growSection(label: "Grow for", prompt: "blank: as long as the world it comes from")
            changesSection
            advancedSection
            footer
        }
    }

    private var keepGrowingForm: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let row = model.sourceRow {
                Text("\(row.name) is at step \(WorldsGuide.grouped(row.step ?? 0))" + (row.targetSteps.map { " of the \(WorldsGuide.grouped($0)) it was asked to reach" } ?? "") + ".")
                    .font(.callout)
            }
            growSection(label: "Grow for", prompt: "blank: until it reaches the step it was asked for", more: true)
            HStack(spacing: 10) {
                Text("Its settings cannot change while it keeps growing.").font(.caption).foregroundStyle(.secondary)
                if let world = model.intent.world {
                    Button("Branch…") { model.openBranch(from: world) }.buttonStyle(.link).controlSize(.small)
                }
            }
            DisclosureGroup("Advanced", isExpanded: $advancedOpen) {
                VStack(alignment: .leading, spacing: 12) {
                    DirectorSettingsBox(controller: controller, title: "Autopilot")
                    timerBox
                }
                .padding(.top, 8)
            }
            footer
        }
    }

    // MARK: Fields

    private var nameField: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Text("Name").frame(width: 120, alignment: .leading)
                TextField("", text: $model.name)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel("Name")
                    .onChange(of: model.name) { _, _ in model.clearRefusal() }
            }
            if !model.nameProblems.isEmpty {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(model.nameProblems, id: \.self) { Text($0).font(.caption).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true) }
                    if let free = model.freeAlternative {
                        Button("Use \(free)") { model.name = free }.buttonStyle(.link).controlSize(.small)
                    }
                }
                .padding(.leading, 128)
            }
        }
    }

    private var startingPointSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text("Starting point").frame(width: 120, alignment: .leading)
                Picker("Starting point", selection: Binding(get: { model.startingPoint }, set: { model.choose($0, applyMemory: true) })) {
                    ForEach(Recipes.all) { Text($0.title).tag(StartingPoint.recipe($0.id)) }
                    Divider()
                    Section("Other presets") {
                        ForEach(Preset.allCases.filter { Recipes.recipe(for: $0) == nil }) { Text($0.rawValue).tag(StartingPoint.preset($0)) }
                    }
                }
                .labelsHidden()
                .fixedSize()
                Spacer(minLength: 0)
            }
            Text(startingPointBlurb).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true).padding(.leading, 128)
            if let entry = model.guide.entry(for: model.startingPoint.preset) {
                DisclosureGroup("About this starting point") {
                    PresetCard(entry: entry, machine: model.guide.machine) { compare = true }.padding(.top, 6)
                }
                .padding(.leading, 128)
                .font(.callout)
            }
        }
    }

    private var startingPointBlurb: String {
        if let recipe = model.startingPoint.recipe { return recipe.blurb }
        return model.guide.entry(for: model.startingPoint.preset)?.guide.what ?? "A preset of the engine."
    }

    private var goalSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text("Goal").frame(width: 120, alignment: .leading)
                Picker("Goal", selection: $model.goal) {
                    ForEach(Goal.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
                .disabled(model.goalsAllowed.count < 2)
                .onChange(of: model.goal) { _, _ in model.clearRefusal() }
                Spacer(minLength: 0)
            }
            Text(model.goalReason ?? model.goal.detail).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                .padding(.leading, 128)
        }
    }

    private func growSection(label: String, prompt: String, more: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Text(label).frame(width: 120, alignment: .leading)
                TextField(prompt, text: Binding(get: { model.growSteps }, set: { model.setGrowSteps($0) }))
                    .textFieldStyle(.roundedBorder)
                    .multilineTextAlignment(.trailing)
                    .frame(width: prompt.isEmpty ? 110 : (more ? 290 : 250))
                    .accessibilityLabel(label)
                Text(more ? "more steps" : "steps")
                Spacer(minLength: 0)
            }
            if let line = model.timeSentence {
                Text(line).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true).padding(.leading, 128)
            }
            if let note = model.timeNote {
                Text(note).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true).padding(.leading, 128)
            }
        }
    }

    // MARK: Changes and remembered values

    @ViewBuilder private var changesSection: some View {
        let changes = model.changes
        let others = model.otherChanges
        let remembered = model.rememberedShown
        if !remembered.isEmpty {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Image(systemName: "clock.arrow.circlepath").foregroundStyle(.orange).accessibilityHidden(true)
                Text("Remembered from last time: \(remembered.count == 1 ? "1 setting" : "\(remembered.count) settings") (\(remembered.map(\.title).joined(separator: ", ")))")
                    .font(.callout).fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 8)
                Button("Reset all") { model.resetRemembered() }.controlSize(.small)
            }
            .padding(10)
            .background(.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
        }
        if !changes.isEmpty || !others.isEmpty {
            VStack(alignment: .leading, spacing: 3) {
                if let sentence = ChangeList.sentence(changes) {
                    Text(sentence).font(.callout).fixedSize(horizontal: false, vertical: true)
                }
                if !others.isEmpty { Text(others.joined(separator: ". ") + ".").font(.callout).fixedSize(horizontal: false, vertical: true) }
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
            .accessibilityElement(children: .combine)
        }
    }

    // MARK: Advanced

    private var advancedSection: some View {
        DisclosureGroup("Advanced", isExpanded: $advancedOpen) {
            VStack(alignment: .leading, spacing: 14) {
                seedRow
                if model.intent == .create {
                    Toggle("Lighter brains", isOn: $controller.settings.fast)
                        .disabled(model.lighterBrainsObstacle != nil)
                        .help("Smaller brains: a lighter and faster world. The engine's --overlay fast.")
                    if let obstacle = model.lighterBrainsObstacle {
                        Text(obstacle).font(.caption).foregroundStyle(.secondary)
                    }
                }
                if model.intent == .create, model.goal == .play { poweredToggle; housesPicker }
                leverGroups
                FeatureLeverSections(controller: controller)
                DirectorSettingsBox(controller: controller, title: "Autopilot")
                timerBox
            }
            .padding(.top, 8)
        }
        .onChange(of: model.lighterBrainsObstacle) { model.dropLighterBrainsIfRefused() }
        .onAppear { model.dropLighterBrainsIfRefused() }
    }

    private var seedRow: some View {
        HStack(alignment: .firstTextBaseline) {
            Text("Seed").frame(width: 120, alignment: .leading)
            TextField(model.intent == .create ? "blank: the starting point's" : "blank: keep the source's", text: $controller.settings.seed)
                .textFieldStyle(.roundedBorder).frame(width: 170)
            Text("The same seed gives the same world again.").font(.caption).foregroundStyle(.secondary)
        }
    }

    private var poweredToggle: some View {
        Toggle("Make a world that takes no powers", isOn: $model.withoutPowers)
            .help("A planet is made so that you can act on it. This leaves the Powers tab empty in the planet.")
    }

    private var housesPicker: some View {
        HStack {
            Text("Rebuild the family lines").frame(width: 270, alignment: .leading)
            Picker("", selection: $model.houses) {
                ForEach([PlayHouses.auto, .exact, .clade], id: \.self) { Text($0.plainTitle).tag($0) }
            }
            .pickerStyle(.segmented).labelsHidden().fixedSize()
            .help(model.houses.plainHelp)
        }
    }

    private var leverGroups: some View {
        let isFork = controller.settings.mode == .fork
        return VStack(alignment: .leading, spacing: 10) {
            GroupBox("Settings") {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(LeverKeys.dynamics.filter { $0.key != "fields.season_mode" }, id: \.key) { item in
                        LeverRow(model: model, controller: controller, key: item.key, title: item.title)
                    }
                    HStack {
                        Text("Seasons arrive").frame(width: 270, alignment: .leading)
                        Picker("", selection: $controller.settings.levers.seasonMode) {
                            Text("Everywhere at once").tag("global")
                            Text("Patch by patch").tag("per_patch")
                        }
                        .labelsHidden().pickerStyle(.segmented).fixedSize()
                        if model.isRemembered("fields.season_mode") { RememberedChip() }
                    }
                }
                .padding(6)
                .frame(maxWidth: .infinity, alignment: .leading)
                HStack {
                    Spacer()
                    Button("Put every setting back to its default") {
                        controller.settings.levers.reset(world3D: controller.settings.world3DBaseline, features: controller.settings.featureLeverBaseline)
                    }
                    .controlSize(.small)
                    .disabled(model.changes.isEmpty)
                }
            }
            if controller.settings.worldDims == 3 {
                GroupBox("World shape and gravity") {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(LeverKeys.world3D, id: \.key) { item in
                            LeverRow(model: model, controller: controller, key: item.key, title: item.title, disabled: isFork && LeverKeys.shapeKeys.contains(item.key))
                        }
                        if controller.settings.levers.world3D.sphere.isSphere {
                            Divider()
                            ForEach(LeverKeys.sphere, id: \.key) { item in
                                LeverRow(model: model, controller: controller, key: item.key, title: item.title, disabled: isFork && LeverKeys.shapeKeys.contains(item.key))
                            }
                            if isFork { Text("A branch keeps the planet's shape.").font(.caption).foregroundStyle(.secondary) }
                        }
                        HStack {
                            Spacer()
                            Button("Put the world shape and gravity back") { controller.settings.levers.world3D = controller.settings.world3DBaseline }
                                .controlSize(.small)
                                .disabled(model.changes.allSatisfy { !LeverKeys.world3D.map(\.key).contains($0.key) && !LeverKeys.sphere.map(\.key).contains($0.key) })
                        }
                    }
                    .padding(6)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            if !model.rememberedSkipped.isEmpty {
                Text("Not used here: " + model.rememberedSkipped.compactMap { LeverKeys.title($0) }.joined(separator: ", ")
                     + ". They stay remembered for a world they fit.").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var timerBox: some View {
        GroupBox("Stop automatically after") {
            VStack(alignment: .leading, spacing: 6) {
                Toggle("Stop after a time", isOn: $controller.settings.timerEnabled)
                if controller.settings.timerEnabled {
                    HStack(spacing: 10) {
                        timerField(value: $controller.settings.timerHours, range: 0...RunSettings.timerHoursMax, unit: "h")
                        timerField(value: $controller.settings.timerMinutes, range: 0...RunSettings.timerMinutesMax, unit: "min")
                    }
                    Text("When the time is up it stops at a save point, as Stop does. A planet is not made after a stop; you are asked.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    /// A typeable number with a stepper; whatever is typed is brought into `range` (the Lab's field did the same).
    private func timerField(value: Binding<Int>, range: ClosedRange<Int>, unit: String) -> some View {
        let clamped = Binding<Int>(
            get: { min(max(value.wrappedValue, range.lowerBound), range.upperBound) },
            set: { value.wrappedValue = min(max($0, range.lowerBound), range.upperBound) })
        return HStack(spacing: 4) {
            TextField(unit, value: clamped, format: .number).frame(width: 44).multilineTextAlignment(.trailing).accessibilityLabel("Stop after, \(unit)")
            Stepper(unit, value: clamped, in: range)
        }
    }

    // MARK: Make planet

    private var makePlanetForm: some View {
        VStack(alignment: .leading, spacing: 16) {
            if launcher.catalog.sources.isEmpty {
                Text("There is nothing to start from yet: a planet is made from a world that has a save point.").foregroundStyle(.secondary)
                Button("New world") { model.openCreate() }
            } else {
                labelled("Start from") {
                    Picker("Start from", selection: Binding(get: { launcher.planet.source }, set: { model.chooseSource($0) })) {
                        ForEach(launcher.catalog.sources) { Text(WorldLauncherModel.title(of: $0)).tag($0.name) }
                    }
                    .labelsHidden().fixedSize()
                }
                if let source = launcher.selectedSource {
                    labelled("Save point") {
                        Picker("Save point", selection: Binding(get: { launcher.planet.sourceStep ?? source.checkpoints.last ?? 0 }, set: { launcher.selectCheckpoint($0) })) {
                            ForEach(source.checkpoints, id: \.self) { Text("step \(WorldsGuide.grouped($0))").tag($0) }
                        }
                        .labelsHidden().fixedSize()
                    }
                }
                if let source = launcher.selectedSource, source.geometry == "sphere" { noteLabel(SetupModel.shapeNote(source)) }
                if let note = model.note { noteLabel(note) }
                nameField
                Toggle("Turn the weather on", isOn: Binding(get: { launcher.planet.storm }, set: { launcher.setStorm($0) }))
                    .disabled(launcher.selectedSource?.weatherOn == true || launcher.weatherObstacle != nil)
                if let obstacle = launcher.weatherNote { noteLabel(obstacle) }
                if launcher.planet.storm {
                    Stepper("Weather warm-up: \(WorldsGuide.grouped(launcher.planet.weatherSpinup)) steps", value: $launcher.planet.weatherSpinup, in: 1...200_000, step: 1000)
                    noteLabel("The weather needs time to settle before play starts. Repeating the same world, save point and length reuses the saved weather and starts at once.")
                }
                DisclosureGroup("Advanced", isExpanded: $advancedOpen) {
                    VStack(alignment: .leading, spacing: 10) {
                        poweredToggle
                        housesPicker
                        if let housesNote = launcher.housesMessage { noteLabel(housesNote) }
                    }
                    .padding(.top, 8)
                }
                footer
            }
        }
    }

    // MARK: Footer

    private var footer: some View {
        let reasons = model.reasons
        return VStack(alignment: .leading, spacing: 10) {
            commandSection(blocks: model.commandBlocks, made: model.plan.madeSentence)
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                VStack(alignment: .leading, spacing: 3) {
                    // (what is already said under the name field is not said twice)
                    ForEach(reasons.filter { !model.nameProblems.contains($0) }, id: \.self) { Text($0).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true) }
                    if let refusal = model.refusal { Text(refusal).font(.callout).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true) }
                }
                Spacer(minLength: 12)
                Button(model.intent.buttonLabel) { model.start() }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .keyboardShortcut("r", modifiers: .command)
                    .disabled(!reasons.isEmpty)
            }
        }
    }

    private func commandSection(blocks: [CommandBlock], made: String?) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if let made { Text(made).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true) }
            DisclosureGroup("Show command", isExpanded: $commandOpen) {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                        VStack(alignment: .leading, spacing: 2) {
                            if blocks.count > 1 { Text(block.title).font(.caption.weight(.semibold)) }
                            Text(block.command).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    Button("Copy command") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(coordinator.isActive ? (coordinator.setup?.copyScript(folder: controller.repoPath) ?? "") : model.plan.copyScript(folder: controller.repoPath), forType: .string)
                    }
                    .controlSize(.small)
                }
                .padding(8)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
            }
            .font(.callout)
        }
    }

    // MARK: Small pieces

    private func noteLabel(_ text: String) -> some View {
        Label(text, systemImage: "info.circle").font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
    }

    private func labelled<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title).frame(width: 120, alignment: .leading)
            content()
            Spacer(minLength: 0)
        }
    }
}

/// A small tag on a row whose value was remembered from the last world.
struct RememberedChip: View {
    var body: some View {
        Text("remembered from last time")
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(.orange.opacity(0.2), in: Capsule())
            .accessibilityLabel("Remembered from last time")
    }
}

/// One lever of the form: its plain name, the field, a tag when the value was remembered, an arrow that puts it back, and the plain message
/// under it when it is not valid. The engine's key is the tooltip.
struct LeverRow: View {
    @ObservedObject var model: SetupModel
    @ObservedObject var controller: RunController
    let key: String
    let title: String
    var disabled = false

    private var current: String { LeverKeys.text(key, in: controller.settings.levers) ?? "" }
    private var baseline: String { LeverKeys.text(key, in: LeverKeys.baseline(of: controller.settings)) ?? "" }
    private var changed: Bool { !LeverKeys.same(key, current, baseline) }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 8) {
                Text(title).frame(width: 270, alignment: .leading)
                TextField("", text: Binding(get: { current }, set: { LeverKeys.set(key, to: $0, in: &controller.settings.levers) }))
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 120)
                    .multilineTextAlignment(.trailing)
                    .disabled(disabled)
                    .accessibilityLabel(title)
                if model.isRemembered(key) { RememberedChip() }
                else if changed { Text("changed").font(.caption2).foregroundStyle(.secondary) }
                else { Text("default").font(.caption2).foregroundStyle(.tertiary) }
                if changed {
                    Button {
                        LeverKeys.set(key, to: baseline, in: &controller.settings.levers)
                    } label: { Image(systemName: "arrow.uturn.backward") }
                        .buttonStyle(.borderless)
                        .help("Put back \(baseline)")
                        .accessibilityLabel("Put back \(title)")
                }
            }
            if let message = model.fieldError(key) {
                Text(message).font(.caption).foregroundStyle(.red).padding(.leading, 278).fixedSize(horizontal: false, vertical: true)
            }
        }
        .help(LeverKeys.explanation(key).map { $0 + " (" + key + ")" } ?? key)
    }
}
