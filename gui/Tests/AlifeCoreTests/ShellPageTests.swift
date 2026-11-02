import Foundation
import Testing

@testable import AlifeCore

/// The pages of the window and their Command keys (docs/APP.md §2.4).
@Suite struct ShellPageTests {
    @Test func theCommandKeysFollowTheSidebarOrder() {
        #expect(ShellPage.allCases.map(\.key) == ["1", "2", "3", "4", "5"])
        #expect(ShellPage.allCases == [.worlds, .watch, .activity, .setup, .details])
        #expect(ShellPage.sidebarPages == [.worlds, .watch, .activity, .setup], "Details has no row of its own: a world's row opens it")
        #expect(Set(ShellPage.allCases.map(\.key)).count == ShellPage.allCases.count)
        #expect(Set(ShellPage.allCases.map(\.title)).count == ShellPage.allCases.count)
    }

    @Test func theWatchAndDetailsPagesAreNeverRestoredBecauseTheyHaveNoWorldToShowAtALaunch() {
        #expect(ShellPage.restored(from: "details") == .worlds && ShellPage.restored(from: "watch") == .worlds)
        #expect(ShellPage.restored(from: "activity") == .activity && ShellPage.restored(from: "setup") == .setup && ShellPage.restored(from: nil) == .worlds)
    }

    @Test func theNamesAreTheSpecsOnes() {
        #expect(ShellPage.allCases.map(\.title) == ["Worlds", "Watch", "Activity", "New world", "Details"])
        // Resume, Home and Continue are not page names; "Watch" is the picture
        for page in ShellPage.allCases { #expect(!["Resume", "Home", "World", "3D Viewer"].contains(page.title)) }
    }

    @Test func aRememberedPageIsRestoredExceptWatch() {
        #expect(ShellPage.restored(from: "setup") == .setup)
        #expect(ShellPage.restored(from: "activity") == .activity)
        #expect(ShellPage.restored(from: "watch") == .worlds)  // opening the app starts nothing, so no world is open to watch
        #expect(ShellPage.restored(from: nil) == .worlds)
        #expect(ShellPage.restored(from: "nonsense") == .worlds)
    }
}
