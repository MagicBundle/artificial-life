import Foundation
import Testing

/// A test that makes a window any way but through `QuietWindow` (Support.swift) is a test that can leak a blocked AppKit animation while the
/// display sleeps, and starve the dispatch pool of the whole run (the review of PR-F measured 79 blocked threads against a limit of 90). The
/// check reads the test sources, so a new test cannot forget.
@Suite struct TestWindowLintTests {
    @Test func everyWindowOfATestIsAQuietOne() throws {
        let dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        var offenders: [String] = []
        for file in try FileManager.default.contentsOfDirectory(atPath: dir.path).sorted() where file.hasSuffix(".swift") && file != "Support.swift" && file != "TestWindowLintTests.swift" {
            let text = try String(contentsOf: dir.appendingPathComponent(file), encoding: .utf8)
            for (i, line) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
                let code = line.split(separator: "/", maxSplits: 1, omittingEmptySubsequences: false).first.map(String.init) ?? ""  // (not a comment)
                if code.contains("NSWindow(contentRect") || code.contains("NSPanel(contentRect") || code.contains(": NSWindow {") || code.contains(": NSPanel {")
                    || code.contains("NSWindow()") || code.contains("NSPanel()") {
                    if line.trimmingCharacters(in: .whitespaces).hasPrefix("//") { continue }
                    offenders.append("\(file):\(i + 1)")
                }
            }
        }
        #expect(offenders.isEmpty, "make windows with QuietWindow: \(offenders)")
    }

    @Test func theHelperItselfTurnsTheAnimationOff() throws {
        let text = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Support.swift"), encoding: .utf8)
        #expect(text.contains("class QuietWindow: NSWindow") && text.contains("animationBehavior = .none"))
    }
}
