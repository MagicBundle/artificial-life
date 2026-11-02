import AlifeCore
import SwiftUI

/// The content of the "World" window (DESIGN §33.4.4): the map and the organisms in the free rectangle, the control center
/// docked beside it. The map view is inset to the free rectangle by its frame, in the same place in the hierarchy
/// whichever side the dock is on, so moving the dock or dragging the divider changes a frame and never the identity of the
/// view (and so never re-creates the Metal view and its device state).
struct WorldView: View {
    @ObservedObject var play: PlayController
    @StateObject private var model: WorldModel
    @AppStorage("world.dockSide") private var sideRaw = DockSide.right.rawValue
    @State private var launcher: WorldLauncherModel?

    init(play: PlayController) {
        self.play = play
        _model = StateObject(wrappedValue: {
            let model = WorldModel(play: play)
            model.adoptGlobeDefault()  // (Home is the globe overview until the player turns it off)
            return model
        }())
    }

    var body: some View {
        WorldDockLayout(play: play, model: model)
            .frame(minWidth: 1100, minHeight: LauncherLayout.windowMinHeight)
            .navigationTitle(WorldTitle.window(world: play.world, header: play.store.play))
            .background(WindowCloseObserver(restorable: false, group: "world") { play.stop() })
            .sheet(isPresented: $model.launcherOpen) {
                if let launcher { WorldLauncherView(model: launcher, play: play, close: { model.launcherOpen = false }) }
            }
            .onAppear {
                if launcher == nil { launcher = Self.makeLauncher() }
                if play.phase == .idle { model.launcherOpen = true }
            }
            .toolbar {
                ToolbarItemGroup {
                    Picker("Dock", selection: $sideRaw) {
                        ForEach(DockSide.allCases) { Image(systemName: Self.symbol($0)).tag($0.rawValue) }
                    }
                    .pickerStyle(.segmented)
                    .help("Dock the control center on the left, the right or the bottom")
                    Button { model.home() } label: { Label("Home", systemImage: "house") }
                        .help("Fly to the Home view: the whole width of the world, north up (H)")
                        .disabled(!play.isActive)
                    Button("Worlds…") { model.launcherOpen = true }
                    Button("Close world") {
                        play.stop()
                        model.launcherOpen = true
                    }
                    .disabled(!play.isActive)
                    .help("Stop the world (it saves a checkpoint first) and open the launcher")
                }
            }
    }

    private static func symbol(_ side: DockSide) -> String {
        switch side {
        case .left: return "sidebar.left"
        case .right: return "sidebar.right"
        case .bottom: return "rectangle.bottomhalf.inset.filled"
        }
    }

    /// The launcher for the repository the Lab is set to.
    @MainActor
    private static func makeLauncher() -> WorldLauncherModel {
        WorldLauncherModel(
            runsDirectory: { RunController.shared.repoReady ? RunController.shared.repoURL.appendingPathComponent("runs") : nil },
            launch: { request in
                let run = RunController.shared
                PlayController.shared.start(request, executable: run.alifeURL, directory: run.repoURL)
            })
    }
}

/// The map and the control center docked beside it (DESIGN §33.4.4): the free rectangle holds the canvas, the dock the control center
/// and the draggable divider. The map view is inset to the free rectangle by its frame, in the same place in the hierarchy
/// whichever side the dock is on, so moving the dock or dragging the divider changes a frame and never the identity of the
/// view (and so never re-creates the Metal view and its device state). Shared by the World window and the `alife` window's
/// Watch layer (`showsDock` is false while another page is shown: the canvas stays, the dock goes).
struct WorldDockLayout: View {
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldModel
    var showsDock = true
    /// "No world is open. Choose one with Worlds..." over the canvas while idle; the `alife` window says it with a button instead.
    var showsIdleBanner = true
    @AppStorage("world.dockSide") private var sideRaw = DockSide.right.rawValue
    @AppStorage("world.dockExtent") private var sideExtent = Double(DockLayout.defaultExtent)
    @AppStorage("world.dockExtentBottom") private var bottomExtent = Double(DockLayout.defaultBottomExtent)
    @AppStorage("world.dockCollapsed") private var collapsed = false

    private var side: DockSide { DockSide(rawValue: sideRaw) ?? .right }

    private var extent: CGFloat { CGFloat(side == .bottom ? bottomExtent : sideExtent) }

    var body: some View {
        GeometryReader { geometry in
            let frames = DockLayout.frames(in: geometry.size, side: side, extent: extent, collapsed: collapsed)
            ZStack(alignment: .topLeading) {
                WorldCanvasView(play: play, model: model, dockCollapsed: collapsed, alertsOverMap: collapsed || side == .bottom,
                                showsIdleBanner: showsIdleBanner)
                    .frame(width: frames.free.width, height: frames.free.height)
                    .position(x: frames.free.midX, y: frames.free.midY)
                if showsDock {
                    ControlCenterView(play: play, model: model, collapsed: $collapsed, side: side, height: frames.dock.height)
                        .frame(width: frames.dock.width, height: frames.dock.height)
                        // (the dock's materials give its hit area a margin of 13 points that reaches into the map: its hit shape is its frame)
                        .contentShape(Rectangle())
                        .position(x: frames.dock.midX, y: frames.dock.midY)
                    if !collapsed {
                        DockDivider(rect: frames.divider, side: side, size: geometry.size)
                    }
                }
            }
            .coordinateSpace(name: "world")
            .onChange(of: frames.free, initial: true) { _, free in model.setFreeRect(free) }
        }
    }
}

/// What the free rectangle shows: the Metal view (created once a world is live and kept), the overlay over it, and the
/// banners for a world that is starting, has ended or failed, or a renderer that cannot draw.
struct WorldCanvasView: View {
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldModel
    /// The dock is collapsed: its clock strip is hidden, so the clock is shown over the map.
    var dockCollapsed = false
    /// The alert rows are over the map (a bottom dock has no room for them, and a collapsed dock hides them).
    var alertsOverMap = false
    /// "No world is open. Choose one with Worlds..." while idle (the `alife` window says it with a button).
    var showsIdleBanner = true

    var body: some View {
        let _ = play.revision
        ZStack {
            Color(red: 0.035, green: 0.04, blue: 0.055)
            // No Metal device is created until a world is live.
            if play.phase == .live || play.store.header != nil {
                if model.sphereActive {
                    // A planet: its own renderer, overlay and controls (DESIGN §46.8); the box world's below are untouched.
                    SphereMetalView(model: model)
                    SphereOverlayView(model: model, overlay: model.overlay, play: play)
                    SphereHintView(model: model)
                } else {
                    WorldMetalView(model: model)
                    WorldOverlayView(model: model, overlay: model.overlay, play: play)
                    WorldHUDView(model: model, overlay: model.overlay, play: play)
                    ControlsHintView(model: model)
                }
                if alertsOverMap { AlertStackOverlay(play: play, model: model) }
                ChapterCardView(play: play, model: model)
                ToastStackView(model: model)
                TickerView(model: model)
                FollowBadge(model: model, play: play)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
                if dockCollapsed {
                    // (the view controls are a column down the right edge, which starts under it: `WorldHUD.columnTop`)
                    ClockBadge(play: play, model: model)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                }
            }
            banner
        }
    }

    @ViewBuilder
    private var banner: some View {
        if let failure = model.drawingFailure {
            text("Cannot draw: \(failure)", error: true)
        } else {
            switch play.phase {
            case .idle: if showsIdleBanner { text("No world is open. Choose one with Worlds…") }
            case .starting:
                VStack(spacing: 8) {
                    ProgressView()
                    Text("Starting \(play.world ?? "the world")…").foregroundStyle(.secondary)
                    Text(play.progress.last ?? "").font(.caption.monospaced()).foregroundStyle(.secondary).lineLimit(1)
                }
                .padding(16)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
            case .live: EmptyView()
            case .ended(let message): text(message)
            case .failed(let message): text(message, error: true)
            }
        }
    }

    private func text(_ string: String, error: Bool = false) -> some View {
        Text(string)
            .foregroundStyle(error ? Color.red : Color.secondary)
            .multilineTextAlignment(.center)
            .padding(14)
            .frame(maxWidth: 520)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
            .textSelection(.enabled)
    }
}
