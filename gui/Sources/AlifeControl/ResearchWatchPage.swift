import AlifeCore
import SwiftUI

/// The chip in the toolbar that says which Watch this is: *Live* (a played world) or *Observer* (a research world). The keys, the World menu and
/// Help follow it.
struct WatchFormChip: View {
    let form: WatchForm

    var body: some View {
        Text(form.chip)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 8).padding(.vertical, 3)
            .background((form == .played ? Color.green : Color.blue).opacity(0.18), in: Capsule())
            .help(form.chipMeaning)
            .accessibilityLabel("\(form.chip) view")
            .accessibilityHint(form.chipMeaning)
    }
}

/// The research form of Watch (docs/APP.md §3.4, W5): the observer's picture of a research world with the Run panel docked beside it. The
/// picture is made by `alife stream`: it follows the run while this app grows it, and re-simulates the latest save point otherwise. It cannot be
/// played (the page says so in one line, with the way to a planet); the keys and the World menu are the observer's own table, never the played
/// world's. The stream ends when the page is left; the world goes on.
struct ResearchWatchPage: View {
    let world: String
    @ObservedObject var shell: ShellModel
    @ObservedObject var observer: ObserverSession
    @ObservedObject var stream: StreamController
    @ObservedObject var controller: RunController
    @ObservedObject var library: WorldLibrary
    @ObservedObject var reader: WorldReader
    @ObservedObject var setup: SetupModel
    @AppStorage("world.dockSide") private var sideRaw = DockSide.right.rawValue
    @AppStorage("world.dockExtent") private var sideExtent = Double(DockLayout.defaultExtent)
    @AppStorage("world.dockExtentBottom") private var bottomExtent = Double(DockLayout.defaultBottomExtent)
    @AppStorage("world.dockCollapsed") private var collapsed = false

    private var side: DockSide { DockSide(rawValue: sideRaw) ?? .right }
    private var extent: CGFloat { CGFloat(side == .bottom ? bottomExtent : sideExtent) }
    private var row: LibraryRow? { library.state.row(named: world) }
    private var growing: Bool { observer.isGrowing(world) }

    var body: some View {
        GeometryReader { geometry in
            let frames = DockLayout.frames(in: geometry.size, side: side, extent: extent, collapsed: collapsed)
            ZStack(alignment: .topLeading) {
                Color(nsColor: .windowBackgroundColor)
                canvasArea
                    .frame(width: frames.free.width, height: frames.free.height)
                    .position(x: frames.free.midX, y: frames.free.midY)
                RunPanel(world: world, reader: reader, controller: controller, library: library, collapsed: $collapsed, side: side,
                         showDetails: showDetails)
                    .frame(width: frames.dock.width, height: frames.dock.height)
                    .contentShape(Rectangle())
                    .position(x: frames.dock.midX, y: frames.dock.midY)
                if !collapsed { DockDivider(rect: frames.divider, side: side, size: geometry.size) }
            }
            .coordinateSpace(name: "world")
        }
        .navigationTitle(DetailsGuide.title(world))
        .navigationSubtitle(WatchGuide.observerLine(follows: stream.source.map(\.follow), step: stream.frame?.step))
        .toolbar {
            ToolbarItemGroup {
                Picker("Dock", selection: $sideRaw) {
                    ForEach(DockSide.allCases) { Image(systemName: Self.symbol($0)).tag($0.rawValue) }
                }
                .pickerStyle(.segmented)
                .help("Dock the Run panel on the left, the right or the bottom")
                WatchFormChip(form: .research)
                Button("Details", action: showDetails)
                    .help("What this world is, how it is doing, and what you can do with it")
            }
        }
        .onAppear {
            reader.retain()
            start(world)
        }
        .onChange(of: world) { _, name in start(name) }
        .onDisappear {
            observer.leave()
            reader.release()
        }
        // Every second: the stream starts when the world has its first save point, follows a run while it grows and re-simulates it afterwards.
        .task(id: world) {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                observer.tick()
            }
        }
    }

    private func start(_ name: String) {
        reader.follow(name)
        library.select(name, scroll: false)
        observer.show(name)
    }

    private func showDetails() {
        library.select(world, scroll: false)
        shell.select(.details)
    }

    private static func symbol(_ side: DockSide) -> String {
        switch side {
        case .left: return "sidebar.left"
        case .right: return "sidebar.right"
        case .bottom: return "rectangle.bottomhalf.inset.filled"
        }
    }

    // MARK: The picture

    private var canvasArea: some View {
        VStack(spacing: 0) {
            ViewerCompactControls(stream: stream, model: observer.viewer, canFollow: growing) { observer.choose(follow: $0) }
            Divider()
            noteStrip
            ZStack {
                Color(red: 0.035, green: 0.04, blue: 0.055)
                picture
                overlay
            }
            Divider()
            ViewerStatusBar(stream: stream, model: observer.viewer, sourceLine: Self.sourceLine)
        }
    }

    @ViewBuilder private var picture: some View {
        if stream.phase == .idle {
            EmptyView()
        } else if stream.header?.isSphere == true {
            SphereLabMetalView(stream: stream, model: observer.viewer)
        } else {
            ViewerMetalView(stream: stream, model: observer.viewer)
        }
    }

    /// One line on what this is, and the way to a planet.
    private var noteStrip: some View {
        HStack(spacing: 10) {
            Image(systemName: "eye").foregroundStyle(.secondary).accessibilityHidden(true)
            Text(WatchGuide.note(forShape: row?.shape)).font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 8)
            if let row, row.shape != .flat {
                let actions = WorldActionSet.make(row: row, controller: controller, play: shell.session.play, library: library, setup: setup)
                Button("Make planet…") { setup.openMakePlanet(from: world) }
                    .controlSize(.small)
                    .disabled(actions.makePlanet.reason != nil)
                    .help(actions.makePlanet.reason ?? "Make a played world from this world's newest save point")
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 5)
        .background(.quaternary.opacity(0.5))
    }

    /// What the picture's place says while there is no picture, in the page's words.
    @ViewBuilder private var overlay: some View {
        if let failure = observer.viewer.drawingFailure {
            card(Text("Cannot draw: \(failure)").foregroundStyle(.red))
        } else {
            switch stream.phase {
            case .live:
                EmptyView()
            case .starting:
                VStack(spacing: 8) {
                    ProgressView()
                    Text(WatchGuide.streamText(.starting, growing: growing, failure: nil) ?? "").foregroundStyle(.secondary)
                }
                .padding(16)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
            case .idle:
                if let text = idleText { card(Text(text).foregroundStyle(.secondary)) }
            case .ended(let message):
                VStack {
                    Text(message).padding(6).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 6))
                    Spacer()
                }
                .padding(8)
            case .failed(let message):
                VStack(spacing: 8) {
                    Text(observer.failures >= WatchGuide.startRetries ? "The viewer could not start." : "The viewer stopped; trying again.")
                        .font(.headline)
                    Text(message).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                    if observer.failures >= WatchGuide.startRetries {
                        HStack {
                            Button("Try again") { observer.tryAgain() }.buttonStyle(.borderedProminent)
                            Button("Show log") { shell.select(.activity) }
                        }
                    }
                }
                .padding(14)
                .frame(maxWidth: 520)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
            }
        }
    }

    private var idleText: String? {
        if case .wait(let text) = observer.step { return text }
        if row == nil { return "This world is not in the library any more. It may have been moved or removed." }
        return nil
    }

    private func card(_ content: some View) -> some View {
        content
            .multilineTextAlignment(.center)
            .padding(14)
            .frame(maxWidth: 520)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
            .textSelection(.enabled)
    }

    /// The line under the status numbers, in the page's words.
    static func sourceLine(_ source: StreamSource) -> String {
        source.follow
            ? "Following the run: the picture moves forward each time the world writes a save point, and shows what the autopilot did."
            : "Re-simulating from the latest save point in a separate process. It leaves the autopilot out, so a world that had one drifts from what you see after the autopilot's first move."
    }
}
