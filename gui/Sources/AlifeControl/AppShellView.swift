import AlifeCore
import AppKit
import SwiftUI

/// The one window of `alife` (docs/APP.md §2.4): a sidebar of pages and the page beside it. The Watch layer (the map and its Metal
/// view) is always in the hierarchy under the pages, so it is made once and kept; the pages above it come and go, and each brings its own
/// title and toolbar. The window has one minimum size for every page, so that entering Watch never resizes it.
struct AppShellView: View {
    @ObservedObject var shell: ShellModel
    @ObservedObject var controller: RunController
    @ObservedObject var play: PlayController
    @ObservedObject var stream: StreamController
    @ObservedObject var model: WorldModel
    @StateObject private var launcher: WorldLauncherModel
    @State private var launchHandled = false
    @Environment(\.dismissWindow) private var dismissWindow
    private let arguments: [String]

    init(shell: ShellModel, controller: RunController, stream: StreamController = .shared, arguments: [String] = CommandLine.arguments) {
        self.shell = shell
        self.controller = controller
        play = shell.session.play
        self.stream = stream
        model = shell.session.model
        self.arguments = arguments
        let play = shell.session.play
        _launcher = StateObject(wrappedValue: WorldLauncherModel.forApp(run: controller, play: play))
    }

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 190, ideal: 215, max: 280)
        } detail: {
            ZStack {
                WatchLayer(play: play, model: model, visible: shell.page == .watch)
                page
            }
        }
        .frame(minWidth: AppShell.minWidth, minHeight: AppShell.minHeight)
        // The 3D viewer window (until PR-F) goes with the main window, so that closing the main window ends the app as it always did.
        .background(WindowCloseObserver(group: "main") { dismissWindow(id: AppWindow.viewer3d.rawValue) })
        .task {
            // `--world RUN` (a fresh launch): started once, through the same path as Play.
            guard !launchHandled else { return }
            launchHandled = true
            shell.handleLaunch(arguments: arguments, controller: controller)
        }
    }

    @ViewBuilder private var page: some View {
        switch shell.page {
        case .watch:
            WatchChrome(shell: shell, play: play, model: model)
        case .worlds:
            WorldsPage(shell: shell, controller: controller, play: play, model: launcher)
                .navigationTitle("Worlds")
        case .activity:
            ActivityPage(shell: shell, controller: controller, play: play, stream: stream)
                .navigationTitle("Activity")
        case .lab:
            LabPage(controller: controller) { shell.select($0) }
                .navigationTitle("Lab")
        }
    }

    // MARK: Sidebar

    private var sidebar: some View {
        List(selection: Binding(get: { shell.page }, set: { if let page = $0 { shell.select(page) } })) {
            ForEach(ShellPage.allCases) { page in
                SidebarRow(page: page, detail: detail(of: page))
                    .tag(page)
                    .help(page.purpose)
            }
        }
        .safeAreaInset(edge: .bottom) { engineStatus }
    }

    private func detail(of page: ShellPage) -> String? {
        switch page {
        case .worlds: return nil
        case .lab: return nil
        case .watch:
            guard let world = play.world, play.phase != .idle else { return "No world open" }
            switch play.phase {
            case .starting: return "\(world) · starting"
            case .live: return "\(world) · live"
            case .ended: return "\(world) · ended"
            case .failed: return "\(world) · failed"
            case .idle: return "No world open"
            }
        case .activity:
            return ActivityGuide.sidebarDetail(ActivityPage.facts(controller: controller, play: play, stream: stream))
        }
    }

    private var engineStatus: some View {
        HStack(spacing: 6) {
            Circle().fill(controller.repoReady ? Color.green : Color.orange).frame(width: 8, height: 8)
            Text(controller.repoReady ? "Engine ready" : (controller.repoFolderExists ? "Engine not set up" : "alife folder not found"))
                .font(.caption).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .help(controller.repoReady ? controller.repoPath : "Open Worlds or Settings to fix it.")
    }
}

private struct SidebarRow: View {
    let page: ShellPage
    let detail: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            Label(page.title, systemImage: page.symbol)
            if let detail {
                Text(detail).font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle).padding(.leading, 26)
            }
        }
        .padding(.vertical, 2)
    }
}

/// The Lab as a page: today's form, kept until the set-up page replaces it (PR-D). It keeps its own labels (Resume, Fork, Open viewer)
/// until then, and says so.
struct LabPage: View {
    @ObservedObject var controller: RunController
    var openPage: (ShellPage) -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Image(systemName: "info.circle").foregroundStyle(.secondary)
                Text("The research Lab as it was. Its words (Resume, Fork) change when the set-up page replaces it.")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 6)
            .background(.quaternary.opacity(0.4))
            ContentView(controller: controller, embedded: true, openPage: openPage)
        }
    }
}
