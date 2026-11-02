import AlifeCore
import AppKit
import SwiftUI
import Testing

@testable import AlifeControl

/// The preset guide's views, drawn by AppKit in a window that is never shown (`ImageRenderer` draws neither a `ScrollView` nor a control).
/// They draw from the repository's real presets, manifest and snapshots. With `ALIFE_PRESET_SHOTS_OUT=DIR` the pictures are also written to
/// DIR as PNG, in light and dark, for looking at; without it nothing is written.
@MainActor
@Suite("The preset guide's views", .serialized)
struct PresetGuideViewTests {
    private var out: URL? { ProcessInfo.processInfo.environment["ALIFE_PRESET_SHOTS_OUT"].map { URL(fileURLWithPath: $0) } }

    private func controller() -> RunController {
        RunController(storedPath: Repo.root.path, savePath: { _ in }, candidates: [Repo.root.path], isApp: false)
    }

    /// The view drawn at `width` x `height` in the given appearance; written as `name-light.png` or `name-dark.png` when asked for.
    private func draw<V: View>(_ view: V, width: CGFloat, height: CGFloat, dark: Bool, name: String) throws -> NSBitmapImageRep {
        let host = NSHostingView(rootView: view.frame(width: width, height: height, alignment: .topLeading))
        let window = QuietWindow(contentRect: NSRect(x: 0, y: 0, width: width, height: height), styleMask: [.borderless], backing: .buffered, defer: true)
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        window.contentView = host
        window.alphaValue = 0
        window.ignoresMouseEvents = true
        window.setFrameOrigin(NSPoint(x: -20000, y: -20000))
        host.frame = NSRect(x: 0, y: 0, width: width, height: height)
        host.layoutSubtreeIfNeeded()
        let rep = try #require(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: rep)
        if let out {
            try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
            try rep.representation(using: .png, properties: [:])?.write(to: out.appendingPathComponent("\(name)-\(dark ? "dark" : "light").png"))
        }
        return rep
    }

    /// The number of distinct colours among a sparse grid of pixels: a blank picture has one or two.
    private func colours(_ rep: NSBitmapImageRep) -> Int {
        var seen = Set<UInt32>()
        let step = max(1, rep.pixelsWide / 160)
        for y in stride(from: 0, to: rep.pixelsHigh, by: step) {
            for x in stride(from: 0, to: rep.pixelsWide, by: step) {
                guard let c = rep.colorAt(x: x, y: y)?.usingColorSpace(.sRGB) else { continue }
                seen.insert(UInt32(c.redComponent * 31) << 10 | UInt32(c.greenComponent * 31) << 5 | UInt32(c.blueComponent * 31))
            }
        }
        return seen.count
    }

    private var content: PresetGuideContent { PresetGuideContent.load(repo: Repo.root) }

    // MARK: The card and the sheet draw

    @Test(arguments: [false, true])
    func theCardDrawsEveryPresetWithItsSnapshot(dark: Bool) throws {
        let content = content
        for entry in content.entries {
            let rep = try draw(PresetCard(entry: entry, machine: content.machine) {}.padding(12).frame(width: 380).background(Color(nsColor: .windowBackgroundColor)),
                               width: 380, height: 520, dark: dark, name: "card-\(entry.preset.rawValue)")
            #expect(colours(rep) > 30, "\(entry.preset.rawValue): the card is blank")
        }
    }

    @Test(arguments: [false, true])
    func theCardWithNoSnapshotShowsAPlaceholderAndTheWords(dark: Bool) throws {
        let entry = PresetGuideEntry(preset: .mini3s, facts: PresetFacts.preset(.mini3s, repo: Repo.root), snapshot: nil,
                                     imageURL: URL(fileURLWithPath: "/nonexistent/mini3s.jpg"))
        let rep = try draw(PresetCard(entry: entry, machine: "") {}.padding(12).frame(width: 380).background(Color(nsColor: .windowBackgroundColor)),
                           width: 380, height: 520, dark: dark, name: "card-no-snapshot")
        #expect(colours(rep) > 10)
        #expect(PresetSnapshotImages.image(at: entry.imageURL) == nil)
    }

    @Test(arguments: [false, true])
    func theSheetDrawsAtItsOwnSize(dark: Bool) throws {
        let content = content
        let rep = try draw(PresetCompareSheet(content: content, selected: .tiny3d) { _ in }.background(Color(nsColor: .windowBackgroundColor)),
                           width: 740, height: 580, dark: dark, name: "sheet")
        #expect(colours(rep) > 60)
    }

    @Test(arguments: [false, true])
    func theWholeListOfTenDrawsInThreeGroups(dark: Bool) throws {
        let content = content
        let list = PresetCompareList(content: content, selected: .k64_3d) { _ in }.padding(16).background(Color(nsColor: .windowBackgroundColor))
        let rep = try draw(list, width: 740, height: 2700, dark: dark, name: "list")
        #expect(colours(rep) > 100)
        let host = NSHostingView(rootView: PresetCompareList(content: content, selected: .dev) { _ in }.padding(16).frame(width: 740))
        host.layoutSubtreeIfNeeded()
        #expect(host.fittingSize.height > 1500, "ten rows need scrolling in a sheet of 580 points: \(host.fittingSize.height)")
    }

    @Test func theSheetFitsTheSmallestMainWindow() {
        // AlifeControlApp gives the window a minimum of 960 x 700; the sheet's ideal size is inside it, and it scrolls.
        let host = NSHostingView(rootView: PresetCompareSheet(content: content, selected: .dev) { _ in })
        host.layoutSubtreeIfNeeded()
        #expect(host.fittingSize.width <= 960 && host.fittingSize.height <= 700, "\(host.fittingSize)")
    }

    // MARK: In the Lab

    @Test(arguments: [false, true])
    func theLabShowsTheCardUnderThePickerAndTheSheetOpens(dark: Bool) throws {
        let c = controller()
        c.settings.mode = .new
        c.settings.preset = .tiny3d
        // the window of the app has its own background; the hosted view has none, so it is given the system's here
        let rep = try draw(ContentView(controller: c, presetGuide: content).background(Color(nsColor: .windowBackgroundColor)), width: 1000, height: 760, dark: dark, name: "lab")
        #expect(colours(rep) > 60)
    }

    @Test func aPickInTheLabRunsWhatThePickerDoesOnAChange() async throws {
        // The sheet sets `controller.settings.preset`, the picker's own binding: the 3D levers follow through the view's `onChange`.
        let c = controller()
        c.settings.mode = .new
        let host = NSHostingView(rootView: ContentView(controller: c, presetGuide: content).frame(width: 1000, height: 760))
        let window = QuietWindow(contentRect: NSRect(x: 0, y: 0, width: 1000, height: 760), styleMask: [.borderless], backing: .buffered, defer: true)
        window.contentView = host
        window.alphaValue = 0  // (this one was ordered in at the origin of the display, at full opacity, for as long as the test took)
        window.ignoresMouseEvents = true
        window.setFrameOrigin(NSPoint(x: -20000, y: -20000))
        window.orderBack(nil)
        defer { window.orderOut(nil) }
        host.layoutSubtreeIfNeeded()
        c.settings.preset = .mini3d  // what a click in the sheet does
        let followed = await eventually(timeout: .seconds(5)) { c.settings.worldDims == 3 && c.settings.presetWorld3D.height == "80" }
        #expect(followed, "the 3D levers did not follow the preset: dims \(c.settings.worldDims), height \(c.settings.presetWorld3D.height)")
    }

    // MARK: The image cache

    @Test func aSnapshotMadeAgainIsSeenWithoutARestart() throws {
        let scratch = try Scratch()
        defer { scratch.cleanup() }
        let file = scratch.url.appendingPathComponent("tiny.jpg")
        let dir = Repo.root.appendingPathComponent("gui/PresetSnapshots")
        try FileManager.default.copyItem(at: dir.appendingPathComponent("tiny.jpg"), to: file)
        let first = try #require(PresetSnapshotImages.image(at: file))
        #expect(PresetSnapshotImages.image(at: file) === first, "the same file is served from the cache")
        try Data(contentsOf: dir.appendingPathComponent("dev.jpg")).write(to: file)
        try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(60)], ofItemAtPath: file.path)
        let second = try #require(PresetSnapshotImages.image(at: file))
        #expect(second !== first, "a file made again is read again")
        try FileManager.default.removeItem(at: file)
        #expect(PresetSnapshotImages.image(at: file) == nil, "a missing file is no image")
    }
}
