import AlifeCore
import AppKit
import CoreGraphics
import Foundation
import SwiftUI
import Testing
import simd

@testable import AlifeControl

/// The fixes of the review of the first A1 build (the findings in `docs/DESIGN.md` §36.5, "review fixes"). Every draw or layout test
/// goes through the hosted `WorldView` (`HostedWorld`: the renderer the Metal view owns, the model the view made), never a `Renderer` or a
/// `WorldModel` in a local, which is how I61 got past every test. Nothing here needs the GPU: the overlay and the controls are SwiftUI
/// `Canvas` views, drawn by `ImageRenderer` over the hosted model.
///
/// They are tests of `WorldInputTests`' suite on purpose: that suite is serialized and moves the dock about in the process's defaults
/// (`world.dockSide`, which `HostedWorld` sets too), and two suites that do so at once, each at its own `await`, see each other's dock.
extension WorldInputTests {
    // MARK: Helpers

    /// The pixels of `image` as RGBA bytes, premultiplied, top row first.
    private func rgba(_ image: CGImage) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: image.width * image.height * 4)
        bytes.withUnsafeMutableBytes { buffer in
            let context = CGContext(data: buffer.baseAddress, width: image.width, height: image.height, bitsPerComponent: 8,
                                    bytesPerRow: 4 * image.width, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        }
        return bytes
    }

    /// How many pixels of `image` (drawn at `scale` pixels per point) inside `rect` (points, origin at the top left) are not transparent.
    private func opaque(_ image: CGImage, in rect: CGRect, scale: Int) -> Int {
        let bytes = rgba(image)
        var count = 0
        let (x0, x1) = (max(Int(rect.minX) * scale, 0), min(Int(rect.maxX) * scale, image.width))
        let (y0, y1) = (max(Int(rect.minY) * scale, 0), min(Int(rect.maxY) * scale, image.height))
        guard x0 < x1, y0 < y1 else { return 0 }
        for y in y0..<y1 { for x in x0..<x1 where bytes[4 * (y * image.width + x) + 3] > 8 { count += 1 } }
        return count
    }

    private func render<V: View>(_ view: V, size: CGSize, scale: Int = 2) throws -> CGImage {
        let renderer = ImageRenderer(content: view.frame(width: size.width, height: size.height))
        renderer.scale = CGFloat(scale)
        return try #require(renderer.cgImage)
    }

    private func resize(_ world: HostedWorld, to size: CGSize) {
        world.window.setContentSize(size)
        world.host.frame = NSRect(origin: .zero, size: size)
        world.settle(20)
    }

    // MARK: A window that changes shape

    @Test func aWindowResizeAwayFromHomeRefitsTheViewToTheNewShape() async throws {
        let world = try await HostedWorld(kind: "3d")
        defer { world.finish() }
        let model = world.model
        let (L, H) = (model.worldSize, model.height)
        model.reduceMotion = { true }
        // Off Home: a pan, and a tilt, at the farthest the narrow window allows.
        model.editPose { $0.x += 20; $0.y += 6; $0.tilt = 0.3 }
        #expect(!model.cameraIsHome && model.pose.fits(aspect: model.aspect, size: L, height: H))
        let narrowAspect = model.aspect
        resize(world, to: CGSize(width: 1700, height: 700))
        #expect(model.aspect > narrowAspect * 1.5, "the window did get wider: \(narrowAspect) to \(model.aspect)")
        #expect(model.pose.distance <= model.distanceLimit * 1.0001, "the distance is within the new limit \(model.distanceLimit): \(model.pose.distance)")
        #expect(model.pose.fits(aspect: model.aspect, size: L, height: H), "the picture is within the one-world patch in the wider window")
        resize(world, to: CGSize(width: 1100, height: 700))
        #expect(model.pose.fits(aspect: model.aspect, size: L, height: H), "and back")
        #expect(!model.cameraIsHome, "a refit is not Home: the player's view stays where it is")
    }

    @Test func aWindowResizeAtHomeStillShowsTheWholeWorldInTheNewShape() async throws {
        let world = try await HostedWorld(kind: "3d")
        defer { world.finish() }
        let model = world.model
        #expect(model.cameraIsHome)
        resize(world, to: CGSize(width: 1700, height: 700))
        #expect(model.cameraIsHome && abs(model.pose.distance - model.distanceLimit) < 1e-3)
        resize(world, to: CGSize(width: 1100, height: 900))
        #expect(model.cameraIsHome && abs(model.pose.distance - model.distanceLimit) < 1e-3)
    }

    // MARK: The minimap's drag

    @Test func draggingTheMinimapKeepsTheCentreMarkerUnderThePointerAtAnyTiltAndHeight() async throws {
        let world = try await HostedWorld(kind: "3d")
        defer { world.finish() }
        let model = world.model
        model.reduceMotion = { true }
        model.display.minimap = true
        world.settle()
        let box = try #require(model.inset?.rect)
        model.editPose { $0.distance = 18; $0.tilt = 0.6; $0.z = 12 }  // (the footprint's limit takes the tilt down to 0.37 here)
        #expect(model.pose.tilt > 0.3 && model.pose.z > 11, "the view really is tilted and above the floor: \(model.pose) L \(model.worldSize) H \(model.height) limit \(model.distanceLimit)")
        model.hudPress(.minimap)
        for fraction in [(0.3, 0.7), (0.5, 0.5), (0.8, 0.25)] {
            let pointer = CGPoint(x: box.minX + box.width * fraction.0, y: box.minY + box.height * fraction.1)
            model.hudDrag(.minimap, to: pointer)
            let inset = try #require(model.inset)
            let marker = inset.centre(of: model.pose, aspect: model.aspect)
            // (the marker is unwrapped about the view's centre: compare modulo the box)
            let side = box.width
            var (dx, dy) = (marker.x - pointer.x, marker.y - pointer.y)
            dx -= side * (dx / side).rounded()
            dy -= side * (dy / side).rounded()
            #expect(hypot(dx, dy) < 0.6, "pointer \(pointer), marker \(marker) (tilt \(model.pose.tilt), z \(model.pose.z))")
        }
        model.hudRelease(.minimap, at: box.origin, clicks: 1, dragged: true)
    }

    // MARK: What the pointer reads

    @Test func overAnOrganismThePointerReadsItsOwnPlaceAndHeightAndOverTheFloorItSaysFloor() async throws {
        let world = try await HostedWorld(kind: "3d")
        defer { world.finish() }
        let model = world.model
        model.reduceMotion = { true }
        let store = world.play.store
        let frame = try #require(store.frame)
        let H = model.height
        var best = 0
        for i in 0..<frame.n where frame.position(i).z > frame.position(best).z { best = i }
        let p = frame.position(best)
        #expect(p.z > 0.25 * H, "the fixture has an organism well above the floor: z = \(p.z) of \(H)")
        // Look at the organism's column from a little to its west, tilted, so that the floor under it is nowhere near it on the screen.
        model.editPose { $0.x = p.x - 3; $0.y = p.y; $0.z = 0; $0.tilt = 0.5; $0.distance = 20 }
        let projector = model.projector(size: model.freeRect.size)
        let screen = try #require(projector.screen(p))
        let floor = try #require(model.pointerReadout(at: screen))
        #expect(floor.hasPrefix("floor "), "no organism named: the floor under the pointer, and it says so: \(floor)")
        let own = try #require(model.pointerReadout(at: screen, over: .organism(uid: frame.uid[best], index: best)))
        #expect(own.hasPrefix("organism ") && own.contains("height"), "\(own)")
        #expect(own == WorldReadout.organismText(p, size: model.worldSize), "its own place, not the floor behind it")
        #expect(own != floor.replacingOccurrences(of: "floor", with: "organism"), "the two differ when the organism is above the floor")
        // The hover itself picks it when it can: the readout of the hover is the organism's.
        model.hover(at: screen)
        if case .organism(let uid, _)? = model.overlay.hover?.pick, uid == frame.uid[best] {
            #expect(model.overlay.pointerText == own)
        } else if model.overlay.hover == nil {
            #expect(model.overlay.pointerText?.hasPrefix("floor ") == true)
        }
        model.hover(at: nil)
        #expect(model.overlay.pointerText == nil)
    }

    // MARK: The overlay keeps off the controls

    @Test func theMapsOverlayDrawsNothingOverTheMinimapTheCompassOrTheButtons() async throws {
        let world = try await HostedWorld(kind: "3d")
        defer { world.finish() }
        let model = world.model
        model.reduceMotion = { true }
        model.display.minimap = true
        world.settle()
        let frame = try #require(world.play.store.frame)
        let free = model.freeRect.size
        let hud = model.hud
        let box = try #require(hud.minimap)
        // Select an organism and look at it so that it is in the middle of the minimap's box: its selection ring is entirely under it.
        let p = frame.position(0)
        model.setSelection(.organism(uid: frame.uid[0]))
        model.editPose { $0.x = p.x; $0.y = p.y; $0.z = p.z; $0.tilt = 0; $0.distance = 20 }
        let ndc = WorldPose.ndc(of: CGPoint(x: box.midX, y: box.midY), in: free)
        var pose = model.pose
        pose.grab(anchor: SIMD2(p.x, p.y), at: ndc, aspect: model.aspect, size: model.worldSize)
        model.editPose { $0 = pose }
        let at = try #require(model.projector(size: free).screen(p))
        #expect(box.contains(at), "the selected organism is drawn under the minimap's box: \(at) in \(box)")
        func overlayImage() throws -> CGImage {
            try render(WorldOverlayView(model: model, overlay: model.overlay, play: world.play), size: free)
        }
        var image = try overlayImage()
        for (target, rect) in hud.rects {
            #expect(opaque(image, in: rect, scale: 2) == 0, "the overlay drew over the \(target): \(opaque(image, in: rect, scale: 2)) pixels")
        }
        // The same ring, with the organism in the middle of the free rectangle, is drawn (the test sees what it is meant to see).
        pose.grab(anchor: SIMD2(p.x, p.y), at: .zero, aspect: model.aspect, size: model.worldSize)
        model.editPose { $0 = pose }
        image = try overlayImage()
        // The ring's radius: 1.8 organism radii at this scale (at least 9 points); it is the pixels just outside the ring's edge that
        // are transparent, and the ring's own that are not.
        let projector = model.projector(size: free)
        let centre = try #require(projector.screen(p))
        let radius = max(projector.points(ofUnits: Float(try #require(world.play.store.header).radius) * 1.8, at: p) ?? 0, 9)
        let onRing = CGRect(x: centre.x + radius - 1.5, y: centre.y - 3, width: 3, height: 6)
        #expect(opaque(image, in: onRing, scale: 2) > 10, "the selection ring (radius \(radius)) is drawn where it is not under a control")
    }

    // MARK: The scale plate

    @Test func theScalePlateNamesItsPlaneAndShowsTheOrganismAsASegmentOnTheBarsOwnScale() async throws {
        let world = try await HostedWorld(kind: "3d")
        defer { world.finish() }
        let model = world.model
        model.reduceMotion = { true }
        model.editPose { $0.distance = 40; $0.tilt = 0 }
        let scale = try #require(model.scaleBar)
        #expect(scale.plane == "on the floor" && scale.bar.label(on: scale.plane).hasSuffix("units on the floor"))
        model.editPose { $0.z = 12 }
        #expect(model.scaleBar?.plane == "at height 12", "\(String(describing: model.scaleBar?.plane)) z \(model.pose.z) H \(model.height)")
        model.editPose { $0.z = 0 }
        // Drawn by the hosted model's HUD: a bar, and under it the organism's segment in yellow, as long as the organism is at this scale.
        let free = model.freeRect.size
        let image = try render(WorldHUDView(model: model, overlay: model.overlay, play: world.play), size: free)
        let organism = try #require(model.scaleBar?.organism)
        let segment = organism.segment(room: 220)
        let bytes = rgba(image)
        var yellow: [Int] = []  // x (in pixels) of the yellow pixels of the plate's rows
        let top = Int(free.height - WorldHUDView.scalePlateBottomInset - WorldHUDView.scalePlateHeight)
        for y in (top * 2)..<Int((free.height - WorldHUDView.scalePlateBottomInset) * 2) {
            for x in 0..<min(600, image.width) {
                let i = 4 * (y * image.width + x)
                if bytes[i + 3] > 200, bytes[i] > 200, bytes[i + 1] > 170, bytes[i + 2] < 120 { yellow.append(x) }
            }
        }
        #expect(!yellow.isEmpty, "the organism's segment is drawn in the plate")
        let drawn = CGFloat((yellow.max() ?? 0) - (yellow.min() ?? 0)) / 2
        // (the yellow of the text is none: the label is white; the segment's end ticks add a line's width)
        if !segment.clipped, segment.points > 6 {
            #expect(abs(drawn - segment.points) < 3.5, "the segment is \(drawn) points long, the organism is \(segment.points) at this scale")
        }
    }

    // MARK: The window's layout

    @Test func theDockContentFitsItsFrameAndNothingOfItLiesOverTheMap() async throws {
        let world = try await HostedWorld(kind: "3d")  // the default 1100 x 700 window with the dock open
        defer { world.finish() }
        let model = world.model
        world.settle(20)
        let free = model.freeRect
        var offenders: [String] = []
        func scan(_ view: NSView) {
            let f = view.convert(view.bounds, to: world.host)
            let name = String(describing: type(of: view))
            if name.contains("SegmentedControl"), f.minX < free.maxX - 0.5 || f.maxX > world.host.bounds.maxX + 0.5 {
                offenders.append("\(name) \(f)")
            }
            for sub in view.subviews { scan(sub) }
        }
        scan(world.host)
        #expect(offenders.isEmpty, "controls that overflow the dock: \(offenders)")
    }

    @Test func theFirstLaunchHintIsClearOfTheScalePlateTheMinimapAndTheCompass() async throws {
        let world = try await HostedWorld(kind: "3d")
        defer { world.finish() }
        let model = world.model
        model.display.minimap = true
        model.dismissHint()
        world.settle(20)
        func snapshot() throws -> NSBitmapImageRep {
            let rep = try #require(world.host.bitmapImageRepForCachingDisplay(in: world.host.bounds))
            world.host.cacheDisplay(in: world.host.bounds, to: rep)
            return rep
        }
        let without = try snapshot()
        model.showHint()
        world.settle(20)
        let with = try snapshot()
        let free = model.freeRect
        let scale = CGFloat(with.pixelsWide) / world.host.bounds.width
        var box: CGRect?
        for y in 0..<with.pixelsHigh {
            for x in 0..<Int(free.width * scale) {
                guard let a = with.colorAt(x: x, y: y), let b = without.colorAt(x: x, y: y) else { continue }
                if abs(a.redComponent - b.redComponent) + abs(a.greenComponent - b.greenComponent) + abs(a.blueComponent - b.blueComponent) > 0.05 {
                    let point = CGRect(x: CGFloat(x) / scale, y: CGFloat(y) / scale, width: 1, height: 1)
                    box = box.map { $0.union(point) } ?? point
                }
            }
        }
        let hint = try #require(box, "the hint is drawn")
        let hud = model.hud
        for (target, rect) in hud.rects { #expect(!hint.intersects(rect), "the hint (\(hint)) lies over the \(target) \(rect)") }
        let plateTop = free.height - WorldHUDView.scalePlateBottomInset - WorldHUDView.scalePlateHeight
        #expect(hint.maxY < plateTop, "above the scale plate: \(hint) against its top at \(plateTop)")
        #expect(hint.maxX <= free.width - WorldHUD.compassSide - WorldHUD.margin, "left of the compass column")
    }
}
