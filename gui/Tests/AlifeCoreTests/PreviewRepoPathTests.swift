import Foundation
import Testing

@testable import AlifeCore

/// The alife preview reads the shared repoPath suite and never writes it (docs/APP.md §6; `RepoPathStoreTests` stay as they were).
@Suite struct PreviewRepoPathTests {
    private func preview(shared: MemoryDomain, own: MemoryDomain, lab: MemoryDomain) -> RepoPathStore {
        RepoPathStore.forApp(info: "alife", environment: [:], standard: own, suite: { name in
            name == RepoPathStore.sharedSuite ? shared : (name == RepoPathStore.labDomain ? lab : nil)
        })
    }

    @Test func thePreviewReadsTheSharedFolderAndNeverWritesTheSuite() {
        let shared = MemoryDomain(), own = MemoryDomain(), lab = MemoryDomain()
        shared.set("/owners/folder", forKey: "repoPath")
        shared.set("other", forKey: "unrelated")
        let before = shared.all.mapValues { "\($0)" }
        let store = preview(shared: shared, own: own, lab: lab)
        #expect(store.load() == "/owners/folder")  // read, as a source
        store.save("/chosen/in/the/preview")
        #expect(store.load() == "/chosen/in/the/preview", "the preview's own choice comes first")
        #expect(own.string(forKey: "repoPath") == "/chosen/in/the/preview")
        #expect(shared.all.mapValues { "\($0)" } == before, "the shared domain is unchanged after a load and a save")
    }

    @Test func thePreviewFromTheLabsFolderDoesNotCopyItIntoTheSuiteEither() {
        let shared = MemoryDomain(), own = MemoryDomain(), lab = MemoryDomain()
        lab.set("/labs/folder", forKey: "repoPath")
        let store = preview(shared: shared, own: own, lab: lab)
        #expect(store.load() == "/labs/folder")
        #expect(shared.all.isEmpty, "the Lab's own load copies into the suite; the preview does not")
    }

    @Test func theOldAppsStillShareAndWriteTheSuite() {
        let shared = MemoryDomain(), own = MemoryDomain(), lab = MemoryDomain()
        let store = RepoPathStore.forApp(info: "lab", environment: [:], standard: own, suite: { name in
            name == RepoPathStore.sharedSuite ? shared : (name == RepoPathStore.labDomain ? lab : nil)
        })
        store.save("/x")
        #expect(shared.string(forKey: "repoPath") == "/x")
    }
}
