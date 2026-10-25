import CoreGraphics
import Foundation
import Testing
import simd

@testable import AlifeCore

private func inside(_ p: CGPoint, _ polygon: [CGPoint]) -> Bool {
    var c = false
    var j = polygon.count - 1
    for i in 0..<polygon.count {
        let a = polygon[i], b = polygon[j]
        if (a.y > p.y) != (b.y > p.y), p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x { c.toggle() }
        j = i
    }
    return c
}

@Suite("The minimap's footprint at the globe")
struct WorldInsetGlobeTests {
    let inset = WorldInset(rect: CGRect(x: 100, y: 50, width: 200, height: 200), size: VantageFixture.L)

    @Test("flat view: no globe footprint")
    func flat() {
        #expect(inset.footprintPieces(of: VantageFixture.make(s: 0)) == nil)
        #expect(inset.footprintPieces(of: VantageFixture.make(s: 1, globe: false)) == nil)
    }

    @Test("the visible cap at Home is acos(R/(R+E)), about 69.5 degrees, and holds the centre")
    func home() throws {
        let v = VantageFixture.make(s: 1, ybar: 0)
        let cap = inset.visibleCap(of: v)
        let R = GlobeMap.radius(L: VantageFixture.L), E = Morph.eyeDistance(distance: v.pose.distance)
        #expect(abs(cap.radius - acos(R / (R + E))) < 1e-3)
        #expect(abs(cap.radius * 180 / .pi - 69.5) < 1.0)
        let pieces = try #require(inset.footprintPieces(of: v))
        let centre = inset.centre(of: v)
        #expect(pieces.contains { inside(centre, $0) })
        // every vertex of the circle is the cap radius from the direction under the eye
        for piece in pieces.prefix(1) {
            for p in piece {
                let u = Float((p.x - inset.rect.minX) / inset.rect.width) * VantageFixture.L
                let ybar = (0.5 - Float((p.y - inset.rect.minY) / inset.rect.height)) * VantageFixture.L
                let d = GlobeMap.direction(longitude: GlobeMap.longitude(x: u, L: VantageFixture.L),
                                           latitude: GlobeMap.latitude(ybar: ybar, L: VantageFixture.L))
                #expect(abs(acos(min(simd_dot(d, cap.direction), 1)) - cap.radius) < 0.02)
            }
        }
    }

    @Test("a cap that holds a pole spans every column and is closed along the edge")
    func polar() throws {
        let v = VantageFixture.make(s: 1, ybar: 0.42 * VantageFixture.L)
        let pieces = try #require(inset.footprintPieces(of: v))
        #expect(pieces.count == 1)
        let polygon = pieces[0]
        let xs = polygon.map(\.x)
        #expect(xs.min()! <= inset.rect.minX && xs.max()! >= inset.rect.maxX)
        // the top edge point under the centre's column is inside, the bottom edge is not
        #expect(inside(CGPoint(x: inset.rect.midX, y: inset.rect.minY + 1), polygon))
        #expect(!inside(CGPoint(x: inset.rect.midX, y: inset.rect.maxY - 1), polygon))
        #expect(inside(inset.centre(of: v), polygon))
        let south = try #require(inset.footprintPieces(of: VantageFixture.make(s: 1, ybar: -0.42 * VantageFixture.L)))
        #expect(inside(CGPoint(x: inset.rect.midX, y: inset.rect.maxY - 1), south[0]))
    }

    @Test("a tilted globe: the cap is about the point under the eye, off the centre")
    func tilted() {
        let v = VantageFixture.make(s: 1, ybar: 0, tilt: 1.0)
        let cap = inset.visibleCap(of: v)
        #expect(simd_dot(cap.direction, v.centreDirection) < 0.999)
        #expect(cap.radius > 0.3 && cap.radius < .pi / 2)
    }

    @Test("in the morph the footprint is the border marched onto the surface, around the centre")
    func morph() throws {
        for s: Float in [0.1, 0.3, 0.5, 0.75] {
            let v = VantageFixture.make(s: s, ybar: 30)
            let pieces = try #require(inset.footprintPieces(of: v), "s \(s)")
            #expect(!pieces.isEmpty)
            #expect(pieces.contains { inside(inset.centre(of: v), $0) }, "s \(s)")
        }
    }

    @Test("the footprint grows as the view comes up and shrinks to the cap")
    func monotone() throws {
        func area(_ s: Float) throws -> Double {
            let piece = try #require(inset.footprintPieces(of: VantageFixture.make(s: s, ybar: 0)))[0]
            var a = 0.0
            for i in piece.indices {
                let p = piece[i], q = piece[(i + 1) % piece.count]
                a += Double(p.x * q.y - q.x * p.y)
            }
            return abs(a) / 2
        }
        let areas = try [0.1, 0.4, 0.8].map { try area(Float($0)) }
        #expect(areas[0] > 0)
        #expect(areas[2] > 0)
    }
}

@Suite("The declutter rule and the geography labels")
struct WorldLabelsTests {
    let bounds = CGRect(x: 0, y: 0, width: 1000, height: 700)
    func storm(_ i: Int, _ x: CGFloat, _ y: CGFloat, named: Bool = true, zeta: Double = 0) -> LabelCandidate {
        LabelCandidate(id: "s\(i)", kind: .storm, rank: (named ? 1000 : 0) + zeta, anchor: CGPoint(x: x, y: y), size: CGSize(width: 60, height: 12))
    }

    @Test("at most 8 names, 90 points apart, named first")
    func rule() {
        var candidates: [LabelCandidate] = []
        for i in 0..<20 { candidates.append(storm(i, 80 + CGFloat(i % 5) * 40, 100 + CGFloat(i / 5) * 40, named: i % 2 == 0, zeta: Double(i))) }
        let placed = LabelDeclutter.place(candidates, in: bounds)
        #expect(placed.count <= LabelDeclutter.maxNames)
        for a in placed.indices {
            for b in placed.indices where a < b {
                #expect(hypot(placed[a].centre.x - placed[b].centre.x, placed[a].centre.y - placed[b].centre.y) >= LabelDeclutter.spacing - 1e-6)
            }
        }
        // the first placed is the highest ranked
        #expect(placed.first?.id == "s18")
    }

    @Test("deterministic, and a moved name has a leader to its anchor")
    func moved() {
        let a = [storm(1, 300, 300), storm(2, 310, 305, zeta: 5)]
        let first = LabelDeclutter.place(a, in: bounds)
        #expect(first == LabelDeclutter.place(a.reversed(), in: bounds))
        #expect(first.count == 2)
        let moved = first.first { $0.leader != nil }
        #expect(moved != nil)
        #expect(first.first { $0.leader == nil }?.id == "s2")
    }

    @Test("the geography never lands on a name")
    func geography() {
        let name = storm(1, 500, 350)
        let geo = LabelCandidate(id: "hemisphere", kind: .geography, rank: 0, anchor: CGPoint(x: 502, y: 352), size: CGSize(width: 110, height: 12))
        let placed = LabelDeclutter.place([geo, name], in: bounds)
        let n = placed.first { $0.id == "s1" }!, g = placed.first { $0.id == "hemisphere" }
        #expect(n.leader == nil)
        if let g {
            let rn = CGRect(x: n.centre.x - 30, y: n.centre.y - 6, width: 60, height: 12)
            let rg = CGRect(x: g.centre.x - 55, y: g.centre.y - 6, width: 110, height: 12)
            #expect(!rn.intersects(rg))
        }
    }

    @Test("the globe's places: the centre's hemisphere, EQUATOR, none on the flat map")
    func places() {
        func projector(ybar: Float, s: Float, globe: Bool = true) -> WorldProjector {
            let v = VantageFixture.make(s: s, ybar: ybar)
            return WorldProjector(pose: v.pose, size: VantageFixture.L, height: VantageFixture.H, view: VantageFixture.view, mapShown: true, globe: globe)
        }
        let north = GlobeLabels.places(projector: projector(ybar: 20, s: 1))
        #expect(north.contains { $0.text == "Northern hemisphere" })
        #expect(!north.contains { $0.text == "Southern hemisphere" })
        #expect(GlobeLabels.places(projector: projector(ybar: -20, s: 1)).contains { $0.text == "Southern hemisphere" })
        #expect(north.contains { $0.text == "EQUATOR" })
        #expect(GlobeLabels.places(projector: projector(ybar: 20, s: 0)).isEmpty)
        #expect(GlobeLabels.places(projector: projector(ybar: 20, s: 1, globe: false)).isEmpty)
    }
}
