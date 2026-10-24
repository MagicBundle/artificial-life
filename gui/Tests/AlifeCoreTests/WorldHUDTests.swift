import CoreGraphics
import Foundation
import Testing

@testable import AlifeCore

/// Where the controls over the map are, and what a press lands on (DESIGN §36.5, the spec's 4.1 and 5.3).
@Suite struct WorldHUDTests {
    private let free = CGSize(width: 780, height: 700)

    @Test func theMinimapIsAtTheBottomRightWithItsSideTheLeastOf220AndAQuarterOfTheShorterSide() {
        let hud = WorldHUD(free: free, minimap: true)
        let box = try! #require(hud.minimap)
        #expect(box.width == box.height && box.width == WorldHUD.minimapSide(free: free))
        #expect(box.width == 196)  // 0.28 x 700
        #expect(box.maxX == free.width - WorldHUD.margin && box.maxY == free.height - WorldHUD.margin)
        #expect(WorldHUD.minimapSide(free: CGSize(width: 1800, height: 1200)) == 220)
    }

    @Test func itIsHiddenWhenOffOrTheWindowIsShortAndTheControlsGoOnShowing() {
        #expect(WorldHUD(free: free, minimap: false).minimap == nil)
        #expect(WorldHUD(free: free, minimap: false).home == nil, "no minimap, no Home glyph")
        let short = WorldHUD(free: CGSize(width: 900, height: 159), minimap: true)
        #expect(short.minimap == nil)
        // Its labels need a side of 96 points: a quarter of the shorter side of a 343 point free rectangle.
        #expect(WorldHUD(free: CGSize(width: 900, height: 340), minimap: true).minimap == nil)
        #expect(WorldHUD(free: CGSize(width: 900, height: 343), minimap: true).minimap?.width == 96)
        #expect(short.zoomIn != nil, "the zoom buttons need only the room of the buttons")
        let tiny = WorldHUD(free: CGSize(width: 200, height: 120), minimap: true)
        #expect(tiny.rects.isEmpty && tiny.target(at: CGPoint(x: 100, y: 60)) == .map)
    }

    @Test func theViewControlsAreAColumnDownTheRightEdgeAboveTheMinimapAndATwoDWorldHasOnlyTheZoomButtons() throws {
        let hud = WorldHUD(free: free, minimap: true)
        let box = try #require(hud.minimap)
        let compass = try #require(hud.compass), toggle = try #require(hud.toggle3D), track = try #require(hud.tiltTrack)
        let zoomIn = try #require(hud.zoomIn), zoomOut = try #require(hud.zoomOut), readout = try #require(hud.readout)
        #expect(hud.presets.map(\.preset) == ViewPreset.allCases)
        #expect(compass.maxX == free.width - WorldHUD.margin, "the compass is at the right edge")
        for rect in [toggle, track, zoomIn, zoomOut] { #expect(rect.midX == compass.midX, "the buttons are centred under the compass") }
        // From the top: the readout, the strip, the compass, the 3D button, the tilt handle, the zoom buttons; all above the minimap.
        let strip = try #require(hud.presets.first).rect
        #expect(readout.maxY < strip.minY && strip.maxY <= compass.minY && compass.maxY < toggle.minY)
        #expect(toggle.maxY < track.minY && track.maxY < zoomIn.minY && zoomIn.maxY < zoomOut.minY)
        #expect(zoomOut.maxY <= box.minY - 30, "above the minimap's coordinates tag")
        #expect(readout.minY >= WorldHUD.columnTop - 0.5, "under the clock badge")
        #expect(WorldHUD.trackLength.contains(track.height))
        // The strip is one row of four, right-aligned with the column, and the readout is as wide.
        #expect(abs(hud.presets.last!.rect.maxX - (free.width - WorldHUD.margin)) < 1e-6 && readout.width == WorldHUD.stripWidth)
        for (a, b) in zip(hud.presets, hud.presets.dropFirst()) { #expect(a.rect.maxX == b.rect.minX && a.rect.minY == b.rect.minY) }
        // A 2D world has no north and no tilt: the zoom buttons take the column.
        let flat = WorldHUD(free: free, minimap: true, compass: false)
        #expect(flat.compass == nil && flat.toggle3D == nil && flat.tiltTrack == nil && flat.presets.isEmpty && flat.readout == nil)
        #expect(flat.zoomIn != nil && flat.zoomOut != nil)
    }

    @Test func theColumnKeepsWhatFitsInTheOrderOfPriorityAndTheTrackTakesTheRoomThatIsLeft() throws {
        // Without the minimap the column reaches the bottom corner. The room above the zoom buttons decides what is kept, in this order:
        // the compass, the 3D button, the handle (from 48 to 96 points), the strip with the readout.
        func column(_ height: CGFloat) -> WorldHUD { WorldHUD(free: CGSize(width: 900, height: height), minimap: false) }
        var last: (compass: Bool, toggle: Bool, track: Bool, strip: Bool) = (false, false, false, false)
        var lengths: [CGFloat] = []
        for height in stride(from: CGFloat(150), through: 700, by: 2) {
            let hud = column(height)
            let now = (hud.compass != nil, hud.toggle3D != nil, hud.tiltTrack != nil, !hud.presets.isEmpty)
            // Nothing that was there goes away as the room grows, and a control is only there if the ones before it in the order are.
            #expect(!last.compass || now.0); #expect(!last.toggle || now.1); #expect(!last.track || now.2); #expect(!last.strip || now.3)
            #expect(!now.1 || now.0); #expect(!now.2 || now.1); #expect(!now.3 || now.2)
            last = now
            if let track = hud.tiltTrack { lengths.append(track.height) }
            if let zoom = hud.zoomOut { #expect(zoom.maxY <= height - WorldHUD.margin + 0.001, "\(height)") }
        }
        #expect(last == (true, true, true, true), "a tall window has all of it")
        #expect(lengths.min() == 48 && lengths.max() == 96 && lengths.allSatisfy { WorldHUD.trackLength.contains($0) },
                "the handle is from 48 to 96 points, as the room allows (the strip takes room when it comes in): \(lengths.min() ?? 0)...\(lengths.max() ?? 0)")
        #expect(column(159).zoomIn != nil && column(159).compass == nil, "A1's least: the zoom buttons alone")
    }

    @Test func theTiltHandleMapsThePointerToATiltFromStraightDownAtTheTopToEdgeOnAtTheBottomAndTheThumbIsWhereTheTiltIs() throws {
        let track = CGRect(x: 800, y: 200, width: 30, height: 96)
        let top = WorldHUD.thumbCentre(tilt: 0, track: track), bottom = WorldHUD.thumbCentre(tilt: WorldPose.maxTilt, track: track)
        #expect(top.y == track.minY + 10 && bottom.y == track.maxY - 10 && top.x == track.midX)
        #expect(WorldHUD.tilt(atY: top.y, track: track) == 0 && abs(WorldHUD.tilt(atY: bottom.y, track: track) - .pi / 2) < 1e-6)
        #expect(WorldHUD.tilt(atY: -1000, track: track) == 0 && WorldHUD.tilt(atY: 1e6, track: track) == WorldPose.maxTilt)
        for tilt in stride(from: Float(0), through: .pi / 2, by: 0.1) {
            let at = WorldHUD.thumbCentre(tilt: tilt, track: track)
            #expect(abs(WorldHUD.tilt(atY: at.y, track: track) - tilt) < 1e-5, "the pointer on the thumb gives its tilt")
        }
        // Held off the thumb's centre by 5 points, the first move does not jump.
        let thumb = WorldHUD.thumbCentre(tilt: 0.6, track: track)
        #expect(abs(WorldHUD.tilt(atY: thumb.y + 5, track: track, grab: 5) - 0.6) < 1e-5)
        // The ticks of the Oblique and the Low views are between the two ends.
        let oblique = WorldHUD.thumbCentre(tilt: ViewPreset.oblique.tilt, track: track), low = WorldHUD.thumbCentre(tilt: ViewPreset.low.tilt, track: track)
        #expect(top.y < oblique.y && oblique.y < low.y && low.y < bottom.y)
    }

    @Test func aPressLandsOnTheNewControlsAndTheHandleIsEasyToGrab() throws {
        let hud = WorldHUD(free: free, minimap: true)
        for entry in hud.presets { #expect(hud.target(at: CGPoint(x: entry.rect.midX, y: entry.rect.midY)) == .preset(entry.preset)) }
        let toggle = try #require(hud.toggle3D), track = try #require(hud.tiltTrack)
        #expect(hud.target(at: CGPoint(x: toggle.midX, y: toggle.midY)) == .toggle3D)
        #expect(hud.target(at: CGPoint(x: track.midX, y: track.midY)) == .tilt)
        #expect(hud.target(at: CGPoint(x: track.minX - 3, y: track.midY)) == .tilt, "a few points off the track still take hold of it")
        #expect(hud.target(at: CGPoint(x: track.minX - 8, y: track.midY)) == .map)
        // The readout is not pressed.
        let readout = try #require(hud.readout)
        #expect(hud.target(at: CGPoint(x: readout.midX, y: readout.midY)) == .map)
        #expect(!hud.clearOfControls.contains(CGPoint(x: readout.midX, y: readout.midY)), "but the map's marks are kept off it")
    }

    @Test func everyControlIsInsideTheFreeRectangleAndNoTwoOverlapExceptTheHomeGlyphInTheMinimap() {
        for size in [CGSize(width: 780, height: 700), CGSize(width: 1472, height: 900), CGSize(width: 400, height: 300), CGSize(width: 1100, height: 360)] {
            let hud = WorldHUD(free: size, minimap: true)
            let all = hud.rects
            for (target, rect) in all {
                #expect(CGRect(origin: .zero, size: size).contains(rect), "\(target) is inside \(size)")
            }
            for (i, a) in all.enumerated() {
                for b in all[(i + 1)...] where !(Set([a.target, b.target]) == [.minimap, .home]) {
                    #expect(!a.rect.intersects(b.rect), "\(a.target) and \(b.target) overlap in \(size)")
                }
            }
            if let home = hud.home, let box = hud.minimap { #expect(box.contains(home)) }
        }
    }

    @Test func theMapsOverlayIsClippedToTheFreeRectangleWithTheControlsCutOut() {
        for size in [CGSize(width: 780, height: 700), CGSize(width: 1472, height: 900)] {
            let hud = WorldHUD(free: size, minimap: true)
            let clear = hud.clearOfControls
            #expect(clear.contains(CGPoint(x: size.width / 2, y: size.height / 2)) && clear.contains(CGPoint(x: 1, y: 1)))
            for (target, rect) in hud.rects {  // (the Home glyph, inside the minimap, is cut out with it: it is not a hole in a hole)
                #expect(!clear.contains(CGPoint(x: rect.midX, y: rect.midY)), "\(target) is not part of the map")
                #expect(!clear.contains(CGPoint(x: rect.minX + 1, y: rect.minY + 1)) && !clear.contains(CGPoint(x: rect.maxX - 1, y: rect.maxY - 1)))
            }
            let box = try! #require(hud.minimap)
            #expect(clear.contains(CGPoint(x: box.minX - 3, y: box.midY)) && clear.contains(CGPoint(x: box.midX, y: box.minY - 3)), "right next to it is the map")
        }
        let none = WorldHUD(free: CGSize(width: 200, height: 120), minimap: true)
        #expect(none.clearOfControls.contains(CGPoint(x: 100, y: 60)), "no controls, the whole rectangle")
    }

    @Test func aPressLandsOnTheControlUnderItAndTheHomeGlyphWinsInsideTheMinimap() {
        let hud = WorldHUD(free: free, minimap: true)
        for (target, rect) in hud.rects { #expect(hud.target(at: CGPoint(x: rect.midX, y: rect.midY)) == target) }
        let box = try! #require(hud.minimap)
        #expect(hud.target(at: CGPoint(x: box.maxX - 5, y: box.maxY - 5)) == .minimap)
        #expect(hud.target(at: CGPoint(x: box.minX - 1, y: box.midY)) == .map)
        #expect(hud.target(at: CGPoint(x: box.midX, y: box.minY - 1)) == .map)
        #expect(hud.target(at: CGPoint(x: 5, y: 5)) == .map)
        #expect(hud.target(at: CGPoint(x: free.width / 2, y: free.height / 2)) == .map)
    }

    @Test func aZoomEaseGoesFromOneDistanceToTheOtherInQuarterOfASecondAndEndsExactly() {
        var ease = ZoomEase(from: 100, to: 70.47)
        var last: Float = 100
        var done = false
        var elapsed: Float = 0
        while !done {
            let step = ease.step(dt: 1.0 / 60)
            elapsed += 1.0 / 60
            #expect(step.distance <= last + 1e-4, "monotone")
            last = step.distance
            done = step.done
            if elapsed > 1 { break }
        }
        #expect(done && last == 70.47)
        #expect(elapsed >= ZoomEase.duration - 0.02 && elapsed <= ZoomEase.duration + 0.03, "\(elapsed)")
        // Smooth in the logarithm: the middle is the geometric mean.
        var half = ZoomEase(from: 100, to: 25)
        let middle = half.step(dt: ZoomEase.duration / 2)
        #expect(abs(middle.distance - 50) < 0.01, "\(middle.distance)")
    }

    @Test func aPressOfAZoomButtonIsATurnOfE035() {
        #expect(abs(exp(WorldHUD.zoomStep) - 1.4190675) < 1e-5)
    }
}
