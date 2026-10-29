import AlifeCore
import Foundation
import AppKit
import SwiftUI
import Testing

@testable import AlifeControl

/// The controller's side of the feature levers (the overlay toggles read the overlay files of the repository) and a render of the boxes.
@MainActor
@Suite struct FeatureLeverViewTests {
    private func controller() -> RunController {
        RunController(storedPath: Repo.root.path, savePath: { _ in }, defaultPath: Repo.root.path, isApp: false)
    }

    @Test func anOverlayToggleReadsTheOverlayFileAndPassesTheOverlay() {
        let c = controller()
        c.settings.mode = .new
        c.settings.name = "t"
        c.settings.preset = .tiny
        c.setFeatureOverlay("bonds", on: true)
        #expect(c.settings.featureOverlays == ["bonds"])
        #expect(c.settings.levers.features.values.isOn("roles.enabled") && c.settings.levers.features["bonds.share_mode"] == "2")
        #expect(c.settings.arguments(existingRuns: []).contains("bonds") && c.settings.arguments(existingRuns: []).filter { $0 == "--set" }.isEmpty)
        c.setFeatureOverlay("bonds", on: false)
        #expect(c.settings.featureOverlays.isEmpty && c.settings.levers.features == FeatureLevers())
        // switching the preset to a 3D one asks for the 3D variant of the farming overlay, where the engine has one
        c.settings.preset = .tiny3d
        c.refreshPresetWorld3D()
        #expect(c.settings.worldDims == 3)
        c.setFeatureOverlay("comm", on: true)
        #expect(c.settings.overlayArguments == ["--overlay", "comm"])
    }

    @Test func aChangedPresetKeepsTheChosenOverlays() {
        let c = controller()
        c.settings.mode = .new
        c.setFeatureOverlay("structures", on: true)
        c.settings.levers.features["structures.cap"] = "2048"
        c.refreshFeatureBaseline()
        #expect(c.settings.levers.features["structures.cap"] == "2048", "a refresh that finds the same baseline keeps the edits")
    }

    @Test func theBoxesRender() throws {
        let c = controller()
        c.settings.mode = .new
        c.setFeatureOverlay("comm", on: true)
        let renderer = ImageRenderer(content: VStack { FeatureLeverSections(controller: c) }.frame(width: 400))
        renderer.scale = 1
        let image = try #require(renderer.cgImage, "the lever boxes draw")
        if let dir = ProcessInfo.processInfo.environment["ALIFE_GPU_TESTS_OUT"] {
            let rep = NSBitmapImageRep(cgImage: image)
            try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: dir).appendingPathComponent("feature-levers.png"))
        }
        // (ImageRenderer draws a GroupBox's title and not its content: five titles are five boxes, and the body built without a crash)
        #expect(image.width >= 380 && image.height >= 90, "five boxes: \(image.width) x \(image.height)")
        c.settings.mode = .fork
        c.settings.sourceRun = "none"
        #expect(ImageRenderer(content: VStack { FeatureLeverSections(controller: c) }.frame(width: 400)).cgImage != nil)
    }
}
