import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeCore

/// The words and numbers over the map (DESIGN §36.5): longitude, latitude, the hemisphere, the readout, the scale bar and the compass.
@Suite struct WorldReadoutTests {
    private let L: Float = 320

    @Test func latitudeIsTheGudermannianOfTheRowAndALongitudeIsADegreeOfTheTurn() {
        #expect(WorldReadout.latitude(ybar: 0, size: L) == 0)
        #expect(abs(WorldReadout.latitude(ybar: 36, size: L) - 37.5) < 0.1, "a warm pool at y = 36")
        #expect(abs(WorldReadout.latitude(ybar: 16, size: L) - 17.7) < 0.1, "the Home tilt of the globe, 0.05 L: 18 N")
        #expect(abs(WorldReadout.latitude(ybar: 160, size: L) - 85.05) < 0.01, "the pole row")
        for y in stride(from: Float(1), to: 160, by: 7) {
            #expect(WorldReadout.latitude(ybar: -y, size: L) == -WorldReadout.latitude(ybar: y, size: L), "odd")
            #expect(WorldReadout.latitude(ybar: y + 3, size: L) > WorldReadout.latitude(ybar: y, size: L), "increasing")
        }
        #expect(WorldReadout.longitude(x: 0, size: L) == 0 && WorldReadout.longitude(x: 160, size: L) == 180)
        #expect(abs(WorldReadout.longitude(x: -80, size: L) - 270) < 1e-4, "wrapped")
    }

    @Test func coordinatesReadLikeTheMinimapsReadout() {
        // 264 E: x = 264/360 L; 28 N: ybar = L/(2 pi) asinh(tan 28 degrees).
        let x = Float(264) / 360 * L
        let y = L / (2 * .pi) * Float(asinh(tan(28 * Double.pi / 180)))
        #expect(WorldReadout.coordinates(x: x, y: y, size: L) == "264 E  28 N")
        #expect(WorldReadout.coordinates(x: 208.9, y: 36, size: L) == "235 E  37 N")
        #expect(WorldReadout.coordinates(x: 208.9, y: 284, size: L) == "235 E  37 S", "y = 284 is 36 south of the equator")
        #expect(WorldReadout.coordinates(x: 0, y: 0, size: L) == "0 E  0")
        #expect(WorldReadout.coordinates(x: 319.9, y: 0, size: L) == "0 E  0", "360 is 0")
    }

    @Test func theHemisphereIsTheOneWithFPositiveNorthOfTheEquator() {
        #expect(WorldReadout.hemisphereName(y: 36, size: L) == "Northern hemisphere")
        #expect(WorldReadout.hemisphereName(y: 284, size: L) == "Southern hemisphere")
        #expect(WorldReadout.hemisphereName(y: 0.5, size: L) == "Equator" && WorldReadout.hemisphereName(y: 319.5, size: L) == "Equator")
        for y in stride(from: Float(5), to: 315, by: 9) {
            let north = WorldReadout.hemisphereName(y: y, size: L) == "Northern hemisphere"
            #expect(north == (WorldPose.hemisphere(y: y, size: L) == 0) && north == (y < 160), "y = \(y)")
        }
    }

    @Test func theHemispheresNameAgreesWithTheLatitudeEvenAtTheWrap() {
        // (review finding: at y = L/2 the name was by `wrap` (Southern) and the latitude by the tie-rounded `wrapDelta` (85 N))
        for y in [Float(159.99), 160, 160.01, 0.7, 319.3, 36, 284] {
            let ybar = Torus.wrapDelta(y, size: L)
            let name = WorldReadout.hemisphereName(y: y, size: L)
            let latitude = WorldReadout.latitude(ybar: ybar, size: L)
            if name == "Northern hemisphere" { #expect(latitude > 0, "y = \(y): \(name), \(latitude)") }
            if name == "Southern hemisphere" { #expect(latitude < 0, "y = \(y): \(name), \(latitude)") }
            let line = WorldReadout.line(pose: WorldPose(x: 10, y: y, distance: 50), size: L, height: 80, mapShown: true)
            #expect(line.contains(name) && (name == "Equator" || line.contains(latitude > 0 ? " N " : " S ")), "\(line)")
        }
    }

    @Test func theReadoutLineNamesTheHemisphereThePlaceAndTheEyesHeightInWorldUnits() {
        // Looking straight down at the floor from a distance of 108.8: the eye is 108.8 units up.
        let p = WorldPose(x: 208.9, y: 36, distance: 108.8)
        #expect(WorldReadout.line(pose: p, size: L, height: 80, mapShown: true) == "Northern hemisphere  235 E  37 N   109 units up")
        #expect(WorldReadout.line(pose: p, size: L, height: 80, mapShown: false) == "x 209  y 36   109 units up", "a world with no map has no hemispheres")
        #expect(!WorldReadout.line(pose: p, size: L, height: 80, mapShown: true).contains(" L "), "L is shown nowhere else")
    }

    @Test func theEyesHeightIsOverTheFloorNotTheDistanceFromThePointLookedAt() {
        // (review finding: an organism at z = 79 seen from 0.08 L = 25.6 units: the eye is at 104.6, above the 80-unit ceiling,
        // and the line said "0.08 L up"; tilted by 0.7 it said the same of an eye 101 over the floor)
        let straight = WorldPose(x: 100, y: 40, z: 79, distance: 25.6)
        #expect(abs(straight.eyeHeight(size: L, height: 80) - 104.6) < 1e-3)
        #expect(WorldReadout.line(pose: straight, size: L, height: 80, mapShown: true).hasSuffix("105 units up"))
        var tilted = straight
        tilted.tilt = 0.7
        #expect(abs(tilted.eyeHeight(size: L, height: 80) - (79 + 25.6 * cos(0.7))) < 1e-3)
        #expect(tilted.eyeHeight(size: L, height: 80) < straight.eyeHeight(size: L, height: 80))
        // At Home it is the distance, as z and the tilt are 0.
        let home = WorldPose(x: 160, y: 0, distance: 220)
        #expect(home.eyeHeight(size: L, height: 80) == 220)
        // A world seen from below has no up: the distance from its plane.
        let flat = WorldPose(x: 100, y: 100, tilt: 0.4, distance: 70, faceDown: true)
        #expect(flat.eyeHeight(size: L, height: 0) == 70)
        #expect(WorldReadout.line(pose: flat, size: L, height: 0, mapShown: false).hasSuffix("70 units away"))
        #expect(WorldReadout.eyeText(4.04) == "4.0 units up" && WorldReadout.eyeText(.nan) == "")
    }

    @Test func thePointerReadoutSaysWhetherItIsTheFloorOrAnOrganism() {
        #expect(WorldReadout.floorText(x: 208.9, y: 36, size: L) == "floor 235 E  37 N")
        #expect(WorldReadout.organismText(SIMD3(208.9, 36, 79.6), size: L) == "organism 235 E  37 N, height 80")
    }

    @Test func aScaleBarIsARoundNumberOfUnitsBetween56And140PointsLong() {
        var last: Float = .infinity
        for exponent in stride(from: Float(-3), through: 3, by: 0.05) {
            let ppu = pow(10, exponent)
            let bar = try! #require(ScaleBar.choose(pointsPerUnit: ppu))
            let mantissa = bar.units / pow(10, floor(log10(bar.units) + 1e-5))
            #expect([1, 2, 5].contains { abs(mantissa - Float($0)) < 1e-3 }, "\(bar.units) at \(ppu)")
            #expect(bar.points <= ScaleBar.maxPoints + 0.01 && bar.points >= 55, "\(bar.points) points for \(bar.units) at \(ppu)")
            #expect(abs(Float(bar.points) - bar.units * ppu) < 1e-3 * max(1, bar.units * ppu))
            #expect(bar.units <= last * 1.0001, "zooming in never makes the bar longer in units")
            last = bar.units
        }
        #expect(ScaleBar.choose(pointsPerUnit: 0) == nil && ScaleBar.choose(pointsPerUnit: .nan) == nil && ScaleBar.choose(pointsPerUnit: -3) == nil)
    }

    @Test func aScaleBarSaysItsUnits() {
        #expect(ScaleBar(units: 1, points: 10).label == "1 unit")
        #expect(ScaleBar(units: 20, points: 10).label == "20 units")
        #expect(ScaleBar(units: 0.5, points: 10).label == "0.5 units")
        #expect(ScaleBar.choose(pointsPerUnit: 4)?.label == "20 units", "140 / 4 = 35 units fit: 20")
    }

    @Test func theOrganismReferenceSaysItsSizeAndHowMuchItIsDrawnOff() {
        let physical = try! #require(ScaleBar.organism(radius: 1, pointsPerUnit: 8, drawnRadiusPoints: 8))
        #expect(physical.diameterUnits == 2 && physical.physicalPoints == 16 && abs(physical.drawnRatio - 1) < 1e-6)
        #expect(physical.label == "organism 2 units")
        #expect(ScaleBar.organism(radius: 1, pointsPerUnit: 2, drawnRadiusPoints: 4.5)?.label == "organism 2 units, drawn x2.2", "a symbol, larger than life")
        #expect(ScaleBar.organism(radius: 1, pointsPerUnit: 50, drawnRadiusPoints: 5)?.label == "organism 2 units, drawn x0.1", "capped, smaller")
        #expect(ScaleBar.organism(radius: 1, pointsPerUnit: 10, drawnRadiusPoints: 11)?.label == "organism 2 units", "within a quarter: no note")
        #expect(ScaleBar.organism(radius: 1, pointsPerUnit: 10, drawnRadiusPoints: .infinity)?.drawnRatio == 1)
        #expect(ScaleBar.organism(radius: 0, pointsPerUnit: 10, drawnRadiusPoints: 4) == nil)
        #expect(ScaleBar.organism(radius: 0.75, pointsPerUnit: 10, drawnRadiusPoints: 7.5)?.label == "organism 1.5 units")
    }

    @Test func aScaleBarSaysWhichPlaneItIsTrueOn() {
        // (review finding: at the region view a bar true on the floor is 2.6 times too long for an organism at the ceiling)
        #expect(ScaleBar.planeText(z: 0, height: 80) == "on the floor" && ScaleBar.planeText(z: 0.3, height: 80) == "on the floor")
        #expect(ScaleBar.planeText(z: 40.4, height: 80) == "at height 40")
        #expect(ScaleBar.planeText(z: 0, height: 0) == nil, "a 2D world has no planes")
        #expect(ScaleBar.planeText(z: .nan, height: 80) == nil)
        let bar = ScaleBar(units: 20, points: 100)
        #expect(bar.label(on: "on the floor") == "20 units on the floor" && bar.label(on: nil) == "20 units")
    }

    @Test func theOrganismIsASegmentOnTheBarsOwnScaleNeverClampedToARing() {
        // (review finding: the ring was clamped to a radius of 13 points, 0.44 of the bar the organism it names equalled)
        let bar = try! #require(ScaleBar.choose(pointsPerUnit: 29.3))  // 2 units = 58.6 points, a bar of 2 units
        #expect(abs(bar.units - 2) < 1e-6 && abs(bar.points - 58.6) < 0.1)
        let organism = try! #require(ScaleBar.organism(radius: 1, pointsPerUnit: 29.3, drawnRadiusPoints: 29.3))
        let segment = organism.segment(room: 200)
        #expect(abs(segment.points - bar.points) < 0.1 && !segment.clipped, "as long as the bar that is its own length: \(segment)")
        // Far away it is a mark of 2 points; very near it is wider than the plate and says so.
        let far = try! #require(ScaleBar.organism(radius: 1, pointsPerUnit: 0.3, drawnRadiusPoints: 1))
        #expect(far.segment(room: 200) == (2, false))
        let near = try! #require(ScaleBar.organism(radius: 1, pointsPerUnit: 400, drawnRadiusPoints: 400))
        #expect(near.segment(room: 200) == (200, true))
        #expect(ScaleBar.OrganismReference(diameterUnits: 2, physicalPoints: .infinity, drawnRatio: 1).segment(room: 120) == (120, true))
    }

    @Test func theCompassPointsNorthUpWhenTheMapIsNorthUpAndTurnsWithTheHeading() {
        var pose = WorldPose(x: 100, y: 100, distance: 100)
        #expect(abs(Compass.northAngle(camera: pose.camera(size: L, height: 80))) < 1e-5)
        pose.heading = 0.5
        let turned = Compass.northAngle(camera: pose.camera(size: L, height: 80))
        #expect(abs(abs(turned) - 0.5) < 1e-4)
        pose.heading = -0.5
        #expect(abs(Compass.northAngle(camera: pose.camera(size: L, height: 80)) + turned) < 1e-4, "the other way")
        pose.heading = .pi / 2
        #expect(abs(abs(Compass.northAngle(camera: pose.camera(size: L, height: 80))) - .pi / 2) < 1e-4)
        // Tilting the view foreshortens north's vertical component; it does not turn it the other way.
        pose.heading = 0.5
        pose.tilt = 0.8
        let expected = atan2(sin(Float(0.5)), cos(Float(0.5)) * cos(Float(0.8)))  // north is foreshortened along the screen's vertical
        #expect(abs(abs(Compass.northAngle(camera: pose.camera(size: L, height: 80))) - expected) < 1e-3)
    }
}
