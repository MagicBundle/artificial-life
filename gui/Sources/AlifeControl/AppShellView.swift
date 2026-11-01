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
    @Environment(\.dismissWindow) private var dismissWindow
    @Environment(\.openWindow) private var openWindow
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
        // The window is the app's: closing it is closing the last window (the Settings window and the 3D Viewer go with it, so that the app
        // ends through the question and the wait of the quit, as the old apps do).
        .background(WindowCloseObserver(group: "main") {
            dismissWindow(id: AppWindow.viewer3d.rawValue)
            AppDelegate.current?.mainWindowClosed()
        })
        .onAppear {
            // A click on the Dock icon brings a closed window back (`AppDelegate.applicationShouldHandleReopen`).
            AppDelegate.current?.reopenMainWindow = { openWindow(id: AppShell.mainWindowID) }
        }
        .task {
            // `--world RUN` (a fresh launch): started once, through the same path as Play.
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
                let row = rowText(of: page)
                SidebarRow(page: page, detail: row.text, marked: row.marked)
                    .tag(page)
                    .help(page.purpose)
            }
        }
        .safeAreaInset(edge: .bottom) { engineStatus }
    }

    /// The second line of a sidebar row, and whether the row is marked (something there waits for the player).
    private func rowText(of page: ShellPage) -> (text: String?, marked: Bool) {
        switch page {
        case .lab: return (nil, false)
        case .worlds:
            if shell.startFailure != nil { return ("A world did not open", true) }
            return (nil, false)
        case .watch:
            let cue = model.watchCue(startFailed: shell.startFailure != nil)
            return (cue.text, cue.needsPlayer)
        case .activity:
            return (ActivityGuide.sidebarDetail(ActivityPage.facts(controller: controller, play: play, stream: stream)), false)
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
    var marked = false

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            HStack(spacing: 6) {
                Label(page.title, systemImage: page.symbol)
                if marked {
                    Circle().fill(Color.orange).frame(width: 8, height: 8)
                        .accessibilityLabel("Waiting for you")
                }
            }
            if let detail {
                Text(detail).font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle).padding(.leading, 26)
            }
        }
        .padding(.vertical, 2)
    }
}
