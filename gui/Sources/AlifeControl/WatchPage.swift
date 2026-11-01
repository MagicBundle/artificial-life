import AlifeCore
import SwiftUI

/// The Watch page of the `alife` window, in two parts (docs/APP.md §2.4 item 3, spike S1).
///
/// `WatchLayer` is the persistent part: the map with its Metal view, and the control center when Watch is the selected page. It is
/// always in the window's hierarchy, below the pages, so the Metal view is made once and kept (S1 criterion (a)); while another page is
/// shown it is hidden, takes no input and draws nothing (`WorldModel.watchActive`). It carries no toolbar, title or size.
///
/// `WatchChrome` is the part that exists only while Watch is selected: the title, the toolbar, and what the page says when no world is
/// open. A toolbar belongs to the page that is shown.
struct WatchLayer: View {
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldModel
    var visible: Bool

    var body: some View {
        // (no control center while no world is open: it would only say "No world is live")
        WorldDockLayout(play: play, model: model, showsDock: visible && play.phase != .idle, showsIdleBanner: false)
            .opacity(visible ? 1 : 0)
            .allowsHitTesting(visible)
            .accessibilityHidden(!visible)
    }
}

struct WatchChrome: View {
    @ObservedObject var shell: ShellModel
    @ObservedObject var play: PlayController
    @ObservedObject var model: WorldModel
    @AppStorage("world.dockSide") private var sideRaw = DockSide.right.rawValue

    var body: some View {
        ZStack {
            Color.clear.allowsHitTesting(false)
            switch play.phase {
            case .idle:
                Color(nsColor: .windowBackgroundColor)
                emptyState
            case .ended, .failed: backBar
            case .starting, .live: EmptyView()
            }
        }
        .navigationTitle(play.world == nil ? "Watch" : WorldTitle.window(world: play.world, header: play.store.play))
        .navigationSubtitle(WorldTitle.subtitle(header: play.store.play) ?? "")
        .toolbar {
            ToolbarItemGroup {
                Picker("Dock", selection: $sideRaw) {
                    ForEach(DockSide.allCases) { Image(systemName: Self.symbol($0)).tag($0.rawValue) }
                }
                .pickerStyle(.segmented)
                .help("Dock the control center on the left, the right or the bottom")
                Button { model.home() } label: { Label("Overview", systemImage: "globe") }
                    .help(play.isActive ? "Fly to the overview: the whole width of the world, north up (H)" : "No world is open.")
                    .disabled(!play.isActive)
                Button { shell.closeWorld() } label: { Label("Close World", systemImage: "xmark.circle") }
                    .help(play.isActive ? "Stop the world (it saves a save point first) and go to Worlds" : "No world is open.")
                    .disabled(!play.isActive)
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "eye.slash").font(.system(size: 34)).foregroundStyle(.secondary)
            Text("No world is open").font(.title2.weight(.semibold))
            Text("Watch shows the world you are playing. Choose a world to play, or make a planet, on the Worlds page.")
                .foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: 420)
            Button("Go to Worlds") { shell.select(.worlds) }.buttonStyle(.borderedProminent)
        }
        .padding(28)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    /// Under the banner the canvas shows for a world that ended or failed.
    private var backBar: some View {
        VStack {
            Spacer()
            Button("Back to Worlds") { shell.select(.worlds) }
                .buttonStyle(.borderedProminent)
                .padding(.bottom, 28)
        }
    }

    private static func symbol(_ side: DockSide) -> String {
        switch side {
        case .left: return "sidebar.left"
        case .right: return "sidebar.right"
        case .bottom: return "rectangle.bottomhalf.inset.filled"
        }
    }
}
