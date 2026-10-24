import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeCore

/// The wrap of the world in y and the one-world patch (DESIGN §36.5; spec 2.2, 3.4 and 7.1 `SeamRuleTests`): the copies are drawn
/// where the picture reaches them, every layer of a copy uses that copy's image of a point, and nothing of the picture is ever outside
/// the patch and its copies.
@Suite struct SeamCopiesTests {
    private let L: Float = 320
    private let H: Float = 80

    @Test func theBaseCellIsTheOneTheCentreIsIn() {
        #expect(SeamCopies.base(y: 0, size: L) == 0 && SeamCopies.base(y: 100, size: L) == 0 && SeamCopies.base(y: 159.9, size: L) == 0)
        #expect(SeamCopies.base(y: 160.1, size: L) == 1 && SeamCopies.base(y: 300, size: L) == 1 && SeamCopies.base(y: 319.9, size: L) == 1)
        #expect(SeamCopies.base(y: 160, size: L) == 0, "ties to even, as the shaders' rint")
        let cell = SeamCopies.range(of: SeamCopies.Copy(k: 1, alpha: 1), size: L)
        #expect(cell == 160...480)
        #expect(SeamCopies.range(of: SeamCopies.Copy(k: 0, alpha: 1), size: L) == -160...160)
    }

    @Test func aViewFarFromTheWrapDrawsOnlyItsOwnCellAndOneThatReachesItDrawsTheNeighbourToo() {
        // Centred on the equator and reaching 60: the edges of the cell are 160 away.
        #expect(SeamCopies.plan(y: 0, extentY: 60, size: L) == [SeamCopies.Copy(k: 0, alpha: 1)])
        // Reaching within 0.1 L (32) of the north edge: 100 + 30 = 130 > 128.
        #expect(SeamCopies.plan(y: 100, extentY: 30, size: L) == [SeamCopies.Copy(k: 0, alpha: 1), SeamCopies.Copy(k: 1, alpha: 1)])
        #expect(SeamCopies.plan(y: 100, extentY: 27, size: L) == [SeamCopies.Copy(k: 0, alpha: 1)])
        // The south side: the cell k = 1 spans y 160...480, centre 320; a view at 175 reaching 20 is 145 + 20 from the centre of its cell.
        #expect(SeamCopies.plan(y: 175, extentY: 20, size: L).map(\.k) == [0, 1])
        #expect(SeamCopies.plan(y: 290, extentY: 40, size: L).map(\.k) == [1], "wholly inside the cell k = 1")
        #expect(SeamCopies.plan(y: 5, extentY: 140, size: L).map(\.k) == [-1, 0, 1], "a tall view draws both")
        #expect(SeamCopies.plan(y: 0, extentY: nil, size: L).map(\.k) == [-1, 0, 1], "the horizon in view: all three")
    }

    @Test func theCopiesFadeOverTheMorphAndNoneIsDrawnPastItButTheFlatMapIsOpaque() {
        #expect(SeamCopies.alpha(s: 0) == 1 && SeamCopies.alpha(s: 0.1) == 1 && SeamCopies.alpha(s: 0.3) == 0 && SeamCopies.alpha(s: 1) == 0)
        var last: Float = 1
        for i in 0...40 {
            let a = SeamCopies.alpha(s: Float(i) / 40 * 0.4)
            #expect(a <= last + 1e-6 && a >= 0 && a <= 1)
            last = a
        }
        #expect(SeamCopies.plan(y: 100, extentY: 60, size: L, s: 0.2).map(\.k) == [0, 1])
        #expect(SeamCopies.plan(y: 100, extentY: 60, size: L, s: 0.2).last!.alpha < 1)
        #expect(SeamCopies.plan(y: 100, extentY: 60, size: L, s: 0.5).map(\.k) == [0], "only the base cell once the morph is done with them")
    }

    // MARK: The patch covers the picture, whatever the view

    @Test func theCopiesAndTheMeshCoverEveryPointOfThePictureTheHazeHasNotTakenAtAnyHeadingTiltAndShape() {
        // (A1: "the clamps leave no horizon in view" and the patch is L wide. Now the view tilts to edge-on and the mesh is 3 L wide with
        // the fog, `WorldFog.meshHalfX`; what must be covered is every floor point the haze has not taken: its fog under 98%, which is
        // within 0.9 L of the point looked at, and so is within a cell of the plan and the mesh.)
        var g = SeededGenerator(seed: 21)
        var seen = 0, horizons = 0
        for trial in 0..<300 {
            let aspect = Float([0.5, 1, 1.38, 2.4, 4].randomElement(using: &g)!)
            var pose = WorldPose(x: .random(in: 0..<L, using: &g), y: trial % 3 == 0 ? L / 2 + .random(in: -20...20, using: &g) : .random(in: 0..<L, using: &g),
                                 z: .random(in: 0...H, using: &g), heading: .random(in: -.pi..<(.pi), using: &g),
                                 tilt: .random(in: 0...(.pi / 2), using: &g), distance: .random(in: 5...400, using: &g))
            pose.clamp(aspect: aspect, size: L, height: H)
            let extent = pose.footprint(aspect: aspect, size: L, height: H)
            if extent == nil { horizons += 1 }
            let plan = SeamCopies.plan(y: pose.y, extentY: extent?.y, size: L)
            let cells = plan.map { SeamCopies.range(of: $0, size: L) }
            for i in 0...10 {
                for j in 0...10 {
                    let ndc = SIMD2(Float(i) / 5 - 1, Float(j) / 5 - 1)
                    guard let offset = pose.offset(at: ndc, aspect: aspect, planeZ: 0), WorldFog.amount(c: WorldFog.chebyshev(offset), size: L) < 0.98
                    else { continue }
                    let at = SIMD2(pose.x, pose.y) + offset  // the floor point drawn there
                    seen += 1
                    #expect(abs(offset.x) <= WorldFog.meshHalfX * L, "x: the mesh is 3 L wide about the centre")
                    #expect(cells.contains { $0.contains(at.y) }, "trial \(trial): y = \(at.y) is in none of \(cells) (centre \(pose.y), reach \(extent?.y ?? -1))")
                }
            }
        }
        #expect(seen > 10000 && horizons > 20, "the cases are reached: \(seen) points, \(horizons) horizons")
    }

    // MARK: One seam rule for every layer

    @Test func everyLayerOfACopyUsesThatCopysImageOfAPointAndTheCopyThatCoversItIsADrawnOne() {
        // A point on either side of the wrap (y = L/2 = 160), a view centred 6 units from it.
        let center = SIMD2<Float>(100, 154)
        let plan = SeamCopies.plan(y: center.y, extentY: 40, size: L)
        #expect(plan.map(\.k) == [0, 1])
        for (point, copy) in [(SIMD2<Float>(100, 158), 0), (SIMD2(100, 163), 1), (SIMD2(100, 163 - L), 1), (SIMD2(100, 158 + L), 0),
                              (SIMD2(100 + L, 161), 1), (SIMD2(100 - L, 161.5), 1)] {
            let image = SeamCopies.image(of: point, center: center, size: L)
            #expect(abs(image.y - center.y) <= L / 2 && abs(image.x - center.x) <= L / 2, "the minimum image")
            let drawn = SeamCopies.copy(containing: point, center: center, plan: plan, size: L)
            #expect(drawn?.k == copy, "\(point): copy \(drawn?.k ?? -9)")
            if let drawn { #expect(SeamCopies.range(of: drawn, size: L).contains(image.y) && drawn.alpha == 1, "carries the copy's alpha") }
            // The same point in the other copy, shifted by L: not the image that is drawn.
            #expect(abs(image.y - L - center.y) > L / 2 || abs(image.y + L - center.y) > L / 2)
        }
    }

    @Test func aFeatureNextToTheWrapIsWhereTheMapPutsItWithinAPixelWhicheverCopyDrawsIt() throws {
        // The map's texel for a world point is u = x / L, v = y / L on a repeating sampler: the quad of copy k puts the point at
        // `(x, kL + v-offset)`. The projector places an organism, a bond end or a storm ring at the minimum image. Both are the same screen
        // point for every point near the wrap, in the cell the view is in and in its neighbour.
        let size = CGSize(width: 1100, height: 800)
        for centre in [Float(150), 160, 170, 310, 5] {
            var pose = WorldPose(x: 100, y: centre, distance: 40)
            pose.clamp(aspect: Float(size.width / size.height), size: L, height: H)
            let projector = WorldProjector(camera: pose.camera(size: L, height: H), size: L, view: size)
            let plan = SeamCopies.plan(y: pose.y, extentY: 30, size: L)
            for dy: Float in [-9, -1, -0.01, 0, 0.01, 1, 9] {
                for wrapped in [Float(0), 1, -1] {  // the point written in any of its images
                    let point = SIMD3<Float>(100 + 6 + wrapped * L, 160 + dy + wrapped * L, 0)
                    guard let layer = projector.screen(point) else { continue }
                    guard let copy = SeamCopies.copy(containing: SIMD2(point.x, point.y), center: SIMD2(pose.x, pose.y), plan: plan, size: L)
                    else { continue }  // (outside the 30 units this centre reaches)
                    // The map feature in that copy: the texel (u, v) of the point placed in the copy's cell.
                    let v = (160 + dy) - Float(copy.k) * L  // 160 + dy is the point's y in the base cell 0 or 1: shift into copy k's cell
                    let feature = SIMD3<Float>(106, Float(copy.k) * L + Torus.wrapDelta(v, size: L), 0)
                    let onMap = try #require(projector.screen(feature))
                    #expect(hypot(layer.x - onMap.x, layer.y - onMap.y) < 1, "centre \(centre) dy \(dy): \(layer) against \(onMap)")
                }
            }
        }
    }
}
