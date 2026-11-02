import Foundation
import Testing

@testable import AlifeCore

@Suite struct ActivityGuideTests {
    @Test func nothingRunningSaysWhereToGo() {
        #expect(ActivityGuide.items(.init()).isEmpty)
        #expect(ActivityGuide.sidebarDetail(.init()) == "Nothing running")
        #expect(ActivityGuide.emptyAdvice.contains("Worlds") && ActivityGuide.emptyAdvice.contains("new"))
    }

    @Test func eachJobHasTheControlThatEndsIt() {
        let run = ActivityGuide.items(.init(runName: "r1"))
        #expect(run.map(\.kind) == [.run] && run[0].controls == [.details, .stop] && run[0].title == "Research run r1")
        let stopping = ActivityGuide.items(.init(runName: "r1", runStopping: true))
        #expect(stopping[0].controls == [.details, .forceStop], "a run has its Details; Stop becomes Force stop while it stops")
        #expect(ActivityGuide.items(.init(setupRunning: true))[0].controls == [.cancel])
        #expect(ActivityGuide.items(.init(checkRunning: true))[0].controls == [.cancel])
        #expect(ActivityGuide.items(.init(worldName: "w"))[0].controls == [.closeWorld])
        #expect(ActivityGuide.items(.init(classicViewerOpen: true))[0].controls == [.closeViewer])
    }

    @Test func aWorldThatIsSavingHasNoControlAndSaysItEndsByItself() {
        let items = ActivityGuide.items(.init(saving: ["w1"]))
        #expect(items.count == 1 && items[0].controls.isEmpty && items[0].detail.contains("ends by itself"))
        #expect(ActivityGuide.sidebarDetail(.init(saving: ["w1"])) == "Saving a world")
    }

    @Test func theSidebarSummarisesWithoutRepeatingTheRows() {
        #expect(ActivityGuide.sidebarDetail(.init(runName: "r1")) == "Research run r1")
        #expect(ActivityGuide.sidebarDetail(.init(runName: "r1", worldName: "w")) == "2 running")
        #expect(ActivityGuide.sidebarDetail(.init(runName: "r1", saving: ["w"])) == "Research run r1")
    }

    @Test func theObserverIsWatchesAndEndsWhenWatchIsLeft() {
        let item = ActivityGuide.items(.init(streamRun: "planet-1-grow"))[0]
        #expect(item.kind == .stream && item.title == "Watching the grown world of planet-1" && item.controls.isEmpty)
        #expect(item.detail.contains("leave Watch") && !item.detail.contains("3D Viewer window"))
    }

    @Test func rowsHaveDistinctIds() {
        let items = ActivityGuide.items(.init(runName: "a", setupRunning: true, worldName: "b", saving: ["c", "d"], streamRun: "e", classicViewerOpen: true))
        #expect(Set(items.map(\.id)).count == items.count)
    }

    @Test func buttonWordsAreTheSpecsOnes() {
        #expect(ActivityGuide.Control.stop.label == "Stop" && ActivityGuide.Control.forceStop.label == "Force stop")
        #expect(ActivityGuide.Control.closeWorld.label == "Close World")
    }
}
