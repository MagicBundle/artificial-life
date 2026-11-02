import Foundation
import Testing

@testable import AlifeCore

@Suite struct HelpTopicsTests {
    @Test func theTopicsAreDistinctAndWorldControlsIsLast() {
        let ids = HelpTopics.all.map(\.id)
        #expect(Set(ids).count == ids.count && ids.last == HelpTopics.controlsID)
        #expect(HelpTopics.topic("watch")?.title == "Watch" && HelpTopics.topic("nope") == nil)
        #expect(Set(HelpTopics.all.map(\.title)).count == HelpTopics.all.count)
        // the key tables are drawn from the tables, not written here
        #expect(HelpTopics.topic(HelpTopics.controlsID)?.paragraphs.isEmpty == true)
    }

    @Test func everyTopicHasWordsAndNoneUsesAWordThatIsRetired() {
        let banned = ["Resume", "Checkpoint", "checkpoint", "Fork", "Home", "spin-up", "Storm Worlds", "Source", "Repository", "repository", "its seed",
                      "the Lab", "Director", "director", "Inspect"]
        for topic in HelpTopics.all where topic.id != HelpTopics.controlsID {
            #expect(!topic.paragraphs.isEmpty, Comment(rawValue: topic.id))
            for text in topic.paragraphs { for word in banned { #expect(!text.contains(word), "'\(word)' in \(topic.id): \(text)") } }
        }
    }

    @Test func helpSaysWhatBothKindsOfWorldCanDo() {
        let watch = HelpTopics.topic("watch")?.paragraphs.joined(separator: " ") ?? ""
        #expect(watch.contains("observer") && watch.contains("cannot use powers") && watch.contains("Live or Observer"))
        let kinds = HelpTopics.topic("kinds")?.paragraphs.joined(separator: " ") ?? ""
        #expect(kinds.contains("Played") && kinds.contains("Research") && kinds.contains("Make planet"))
    }

    @Test func theNamesOfTheActionsAreTheOnesOfTheApp() {
        let actions = HelpTopics.topic("actions")?.paragraphs.joined(separator: " ") ?? ""
        for word in ["Create", "Branch", "Keep growing", "Make planet", "Play", "Check world", "Close World", "Stop", "Force stop"] { #expect(actions.contains(word), Comment(rawValue: word)) }
    }

    @Test func theTwoTablesAreHeadedByWhatTheyAreFor() {
        #expect(HelpTopics.controlsHeading(.played) == "When you play a world")
        #expect(HelpTopics.controlsHeading(.research) == "When you watch a research world")
        #expect(HelpTopics.controlsIntro.contains("two sets of keys"))
    }
}
