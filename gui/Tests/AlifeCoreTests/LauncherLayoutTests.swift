import CoreGraphics
import Testing

@testable import AlifeCore

@Suite struct LauncherLayoutTests {
    @Test func theSheetNeverOutgrowsTheMinimumWorldWindow() {
        #expect(LauncherLayout.maxHeight + LauncherLayout.windowMargin <= LauncherLayout.windowMinHeight)
        #expect(LauncherLayout.minHeight <= LauncherLayout.idealHeight && LauncherLayout.idealHeight <= LauncherLayout.maxHeight)
    }

    @Test func theResumeListShowsAFewRowsAndScrollsBeyond() {
        #expect(LauncherLayout.resumeListHeight(count: 0) == 0)
        #expect(LauncherLayout.resumeListHeight(count: 1) == LauncherLayout.resumeRowHeight)
        let three = LauncherLayout.resumeListHeight(count: LauncherLayout.resumeRowsShown)
        #expect(three == 3 * LauncherLayout.resumeRowHeight + 2 * LauncherLayout.resumeRowSpacing)
        #expect(LauncherLayout.resumeListHeight(count: 40) == three, "a long list is a window of the same height")
        #expect(!LauncherLayout.resumeListScrolls(count: 3) && LauncherLayout.resumeListScrolls(count: 4))
    }
}
