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
    @AppStorage("world.dockExtent") private var sideExtent = Double(DockLayout.defaultExtent)
    @AppStorage("world.dockExtentBottom") private var bottomExtent = Double(DockLayout.defaultBottomExtent)
    @AppStorage("world.dockCollapsed") private var collapsed = false
    @State private var dragStamp = Date.distantPast
    @State private var launcher: WorldLauncherModel?

    init(play: PlayController) {
        self.play = play
        _model = StateObject(wrappedValue: WorldModel(play: play))
    }

    private var side: DockSide { DockSide(rawValue: sideRaw) ?? .right }

    private var extent: CGFloat { CGFloat(side == .bottom ? bottomExtent : sideExtent) }

    var body: some View {
        GeometryReader { geometry in
            let frames = DockLayout.frames(in: geometry.size, side: side, extent: extent, collapsed: collapsed)
            ZStack(alignment: .topLeading) {
                WorldCanvasView(play: play, model: model)
                    .frame(width: frames.free.width, height: frames.free.height)
                    .position(x: frames.free.midX, y: frames.free.midY)
                ControlCenterView(play: play, model: model, collapsed: $collapsed, side: side)
                    .frame(width: frames.dock.width, height: frames.dock.height)
                    .position(x: frames.dock.midX, y: frames.dock.midY)
                if !collapsed {
                    divider(frames.divider, in: geometry.size)
                }
            }
            .coordinateSpace(name: "world")
            .onChange(of: frames.free, initial: true) { _, free in model.setFreeRect(free) }
        }
        .frame(minWidth: 1100, minHeight: 700)
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
                Button("Worlds…") { model.launcherOpen = true }
            }
        }
    }

    private func divider(_ rect: CGRect, in size: CGSize) -> some View {
        Rectangle()
            .fill(Color.clear)
            .frame(width: rect.width, height: rect.height)
            .contentShape(Rectangle())
            .position(x: rect.midX, y: rect.midY)
            .onHover { inside in
                if inside { (side == .bottom ? NSCursor.resizeUpDown : NSCursor.resizeLeftRight).push() } else { NSCursor.pop() }
            }
            .gesture(
                DragGesture(minimumDistance: 0, coordinateSpace: .named("world")).onChanged { value in
                    // The drawable resizes with each step of the divider: at most 30 a second.
                    guard Date().timeIntervalSince(dragStamp) >= 1.0 / 30 else { return }
                    dragStamp = Date()
                    let e = Double(DockLayout.extent(draggingTo: value.location, side: side, in: size))
                    if side == .bottom { bottomExtent = e } else { sideExtent = e }
                })
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

/// What the free rectangle shows: the Metal view (created once a world is live and kept), the overlay over it, and the
/// banners for a world that is starting, has ended or failed, or a renderer that cannot draw.
struct WorldCanvasView: View {
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldModel

    var body: some View {
        let _ = play.revision
        ZStack {
            Color(red: 0.035, green: 0.04, blue: 0.055)
            // No Metal device is created until a world is live.
            if play.phase == .live || play.store.header != nil {
                WorldMetalView(model: model)
                WorldOverlayView(model: model, overlay: model.overlay, play: play)
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
            case .idle: text("No world is open. Choose one with Worlds…")
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
