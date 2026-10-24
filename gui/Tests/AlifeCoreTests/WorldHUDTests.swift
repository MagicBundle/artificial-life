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
        #expect(WorldHUD(free: CGSize(width: 900, height: 160), minimap: true).minimap != nil)
        #expect(short.zoomIn != nil, "the zoom buttons need only the room of the buttons")
        let tiny = WorldHUD(free: CGSize(width: 200, height: 120), minimap: true)
        #expect(tiny.rects.isEmpty && tiny.target(at: CGPoint(x: 100, y: 60)) == .map)
    }

    @Test func theCompassAndTheZoomButtonsAreStackedAtTheTopRightAndATwoDWorldHasNoCompass() {
        let hud = WorldHUD(free: free, minimap: true)
        let compass = try! #require(hud.compass), zoomIn = try! #require(hud.zoomIn), zoomOut = try! #require(hud.zoomOut)
        #expect(compass.maxX == free.width - WorldHUD.margin && compass.minY == WorldHUD.margin)
        #expect(zoomIn.midX == compass.midX && zoomOut.midX == compass.midX)
        #expect(compass.maxY < zoomIn.minY && zoomIn.maxY < zoomOut.minY)
        let flat = WorldHUD(free: free, minimap: true, compass: false)
        #expect(flat.compass == nil)
        #expect(flat.zoomIn?.minY == WorldHUD.margin, "the buttons take the compass's place")
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
