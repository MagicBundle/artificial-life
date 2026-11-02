import AppKit
import SwiftUI

/// The log of what runs, set-ups and checks print (Activity, and the Run panel of a research Watch).
struct LogView: View {
    @ObservedObject var controller: RunController

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 4) {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 0) {
                            ForEach(controller.logLines) { line in
                                Text(line.text)
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundStyle(line.isError ? Color.red : Color.primary)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .id(line.id)
                            }
                        }
                        .textSelection(.enabled)
                        .padding(6)
                    }
                    .background(Color(nsColor: .textBackgroundColor))
                    .overlay {
                        if controller.logLines.isEmpty {
                            Text("Nothing logged yet. What runs, setups and checks print appears here.")
                                .font(.callout).foregroundStyle(.secondary).padding()
                        }
                    }
                    .onChange(of: controller.logLines.last?.id) { _, lastID in
                        if let lastID { proxy.scrollTo(lastID, anchor: .bottom) }
                    }
                }
                HStack {
                    Button("Clear") { controller.clearLog() }
                    Button("Copy") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(controller.logText, forType: .string)
                    }
                    Spacer()
                    Text("\(controller.logLines.count) lines").font(.caption).foregroundStyle(.secondary)
                }
                .font(.caption)
            }
        } label: {
            Text("Log")
        }
        .frame(maxHeight: .infinity)
    }
}
