import AlifeCore
import SwiftUI

/// The Activity page (docs/APP.md §2.4): what the app has going, each with the control that ends it, and the log of everything the app
/// runs (`RunController.logLines`). The numbers and the autopilot of the run that grows are here until the Run panel of PR-F.
struct ActivityPage: View {
    @ObservedObject var shell: ShellModel
    @ObservedObject var controller: RunController
    @ObservedObject var play: PlayController
    @ObservedObject var stream: StreamController

    static func facts(controller: RunController, play: PlayController, stream: StreamController) -> ActivityGuide.Facts {
        ActivityGuide.Facts(
            runName: controller.hasRunJob ? (controller.runningRunName ?? "being started") : nil,
            runStopping: controller.hasRunJob && controller.stopRequested,
            setupRunning: controller.job == .setup, checkRunning: controller.job == .inspect,
            worldName: play.isActive ? (play.world ?? "") : nil, worldStarting: play.phase == .starting,
            saving: play.retiredWorlds, streamRun: stream.phase == .idle ? nil : (stream.source?.run ?? ""),
            classicViewerOpen: controller.viewerRunning)
    }

    var body: some View {
        let items = ActivityGuide.items(Self.facts(controller: controller, play: play, stream: stream))
        VStack(alignment: .leading, spacing: 16) {
            Text("Activity").font(.largeTitle.weight(.semibold))
            if items.isEmpty {
                HStack(spacing: 12) {
                    Image(systemName: "checkmark.circle").font(.title2).foregroundStyle(.secondary)
                    Text(ActivityGuide.emptyAdvice).foregroundStyle(.secondary)
                    Spacer()
                    Button("Worlds") { shell.select(.worlds) }
                    Button("New world") { shell.select(.setup) }
                }
                .padding(14)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
            } else {
                VStack(spacing: 0) {
                    ForEach(items) { item in
                        if item.id != items.first?.id { Divider() }
                        row(item)
                    }
                }
                .padding(.vertical, 4)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
            }
            if controller.hasRunJob {
                RunNumbersView(controller: controller)
                DirectorLiveView(controller: controller)
            }
            LogView(controller: controller)
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private func row(_ item: ActivityGuide.Item) -> some View {
        HStack(spacing: 12) {
            if item.controls.isEmpty || item.kind == .saving { ProgressView().controlSize(.small) }
            VStack(alignment: .leading, spacing: 2) {
                Text(item.title).font(.body.weight(.medium))
                Text(item.detail).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            ForEach(item.controls, id: \.self) { control in
                Button(control.label) { perform(control) }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    private func perform(_ control: ActivityGuide.Control) {
        switch control {
        case .stop: controller.stop()
        case .forceStop, .cancel: controller.forceStop()
        case .closeWorld: shell.closeWorld()
        case .closeViewer: controller.closeViewer()
        }
    }
}
