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

    @Test func theReadoutLineNamesTheHemisphereThePlaceAndTheHeight() {
        let p = WorldPose(x: 208.9, y: 36, distance: 108.8)
        #expect(WorldReadout.line(pose: p, size: L, mapShown: true) == "Northern hemisphere  235 E  37 N   0.34 L up")
        #expect(WorldReadout.line(pose: p, size: L, mapShown: false) == "x 209  y 36   0.34 L up", "a world with no map has no hemispheres")
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
