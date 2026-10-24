import Foundation
import Testing

@testable import AlifeCore

/// The scroll law of the World window (DESIGN §36.5, decision D8; spec 7.2 "Scroll and zoom input"), as synthetic event streams.
@Suite struct WorldScrollTests {
    private func trackpad(_ dx: Float = 0, _ dy: Float, phase: ScrollGesturePhase = .changed, momentum: ScrollGesturePhase = .none, t: Double = 0) -> ScrollEvent {
        ScrollEvent(dx: dx, dy: dy, precise: true, phase: phase, momentum: momentum, time: t)
    }

    private func wheel(_ dy: Float, t: Double = 0) -> ScrollEvent { ScrollEvent(dy: dy, precise: false, time: t) }

    @Test func aTrackpadsTwoFingerScrollPansByTheDeltaAndAWheelZoomsAndNothingZoomsWhenThereIsNoDelta() {
        var law = ScrollLaw()
        #expect(law.handle(trackpad(3, -7), scrollZooms: false) == .pan(dx: 3, dy: -7))
        #expect(law.handle(trackpad(0, 0), scrollZooms: false) == .none)
        guard case .zoom(let factor) = law.handle(wheel(2), scrollZooms: false) else { Issue.record("a wheel zooms"); return }
        #expect(abs(factor - exp(-0.2)) < 1e-6, "0.1 per line, scrolling up comes closer")
        guard case .zoom(let out) = law.handle(wheel(-1, t: 0.1), scrollZooms: false) else { Issue.record("a wheel zooms"); return }
        #expect(abs(out - exp(0.1)) < 1e-6)
    }

    @Test func withScrollZoomsOnTheTrackpadZoomsAtSixThousandthsAPoint() {
        var law = ScrollLaw()
        guard case .zoom(let factor) = law.handle(trackpad(0, 50, phase: .began), scrollZooms: true) else { Issue.record("zooms"); return }
        #expect(abs(factor - exp(-0.3)) < 1e-6, "v1's coefficient was 0.01: this one is 0.006")
        #expect(ScrollLaw.preciseCoefficient == 0.006 && ScrollLaw.wheelCoefficient == 0.1)
    }

    @Test func momentumPansWithTheTrackpadButNeverZoomsWhateverTheSwitch() {
        var law = ScrollLaw()
        #expect(law.handle(trackpad(1, 9, phase: .none, momentum: .changed), scrollZooms: false) == .pan(dx: 1, dy: 9))
        #expect(law.handle(trackpad(1, 9, phase: .none, momentum: .changed), scrollZooms: true) == .none)
        #expect(law.handle(trackpad(0, 400, phase: .none, momentum: .began), scrollZooms: true) == .none)
        #expect(law.zoomed == 0)
    }

    @Test func oneAbsurdEventZoomsByTheGesturesWholeAllowanceAndNoMoreAndNaNIsNothing() {
        // (v1 capped an event at e^1.5; the gesture's cap, ln 4, is below it, so a fresh gesture's one huge event is a factor of 4.)
        #expect(ScrollLaw.eventCap == 1.5 && ScrollLaw.gestureCap < ScrollLaw.eventCap)
        var law = ScrollLaw()
        guard case .zoom(let factor) = law.handle(wheel(1000), scrollZooms: false) else { Issue.record("zooms"); return }
        #expect(abs(factor - 0.25) < 1e-6)
        var other = ScrollLaw()
        guard case .zoom(let back) = other.handle(wheel(-1000), scrollZooms: false) else { Issue.record("zooms"); return }
        #expect(abs(back - 4) < 1e-5)
        var nan = ScrollLaw()
        #expect(nan.handle(wheel(.nan), scrollZooms: false) == .none && nan.handle(wheel(.infinity), scrollZooms: false) == .none)
        #expect(nan.zoomed == 0)
    }

    @Test func aGestureCannotCrossTheRangeItsSummedLogIsCappedAtLn4() {
        var law = ScrollLaw()
        var total: Float = 0
        _ = law.handle(trackpad(0, 0, phase: .began, t: 0), scrollZooms: true)
        for i in 0..<300 {  // a hard flick of the fingers, decaying
            if case .zoom(let f) = law.handle(trackpad(0, 60 * exp(-Float(i) / 90), t: 0.008 * Double(i)), scrollZooms: true) { total += log(f) }
        }
        #expect(abs(total + log(4)) < 1e-4, "the factor of the whole flick is 1/4: \(exp(total))")
        #expect(abs(law.zoomed - ScrollLaw.gestureCap) < 1e-4)
        _ = law.handle(trackpad(0, 0, phase: .ended, t: 3), scrollZooms: true)
        var momentum = 0
        for i in 0..<400 { if law.handle(trackpad(0, 40, phase: .none, momentum: .changed, t: 3 + 0.008 * Double(i)), scrollZooms: true) != .none { momentum += 1 } }
        #expect(momentum == 0)
        // The next gesture has the allowance of its own.
        #expect(law.handle(trackpad(0, 20, phase: .began, t: 10), scrollZooms: true) != .none)
        #expect(law.zoomed < ScrollLaw.gestureCap)
    }

    @Test func aGestureCanZoomBackWithinItsAllowanceAndNotMoreThanItInTotal() {
        var law = ScrollLaw()
        _ = law.handle(trackpad(0, 0, phase: .began), scrollZooms: true)
        guard case .zoom(let a) = law.handle(trackpad(0, 100), scrollZooms: true) else { return }  // ln = -0.6
        guard case .zoom(let b) = law.handle(trackpad(0, -50), scrollZooms: true) else { return }  // +0.3
        #expect(abs(log(a) + 0.6) < 1e-5 && abs(log(b) - 0.3) < 1e-5)
        guard case .zoom(let c) = law.handle(trackpad(0, -1000), scrollZooms: true) else { Issue.record("room left"); return }
        #expect(abs(log(c) - (ScrollLaw.gestureCap - 0.9)) < 1e-4, "what is left of the allowance")
        #expect(law.handle(trackpad(0, -1000), scrollZooms: true) == .none, "and then nothing more")
    }

    @Test func anUnphasedWheelsGestureIsTheEventsWithNoPauseOfAQuarterOfASecondBetweenThem() {
        var law = ScrollLaw()
        var total: Float = 0
        for i in 0..<40 { if case .zoom(let f) = law.handle(wheel(1, t: Double(i) * 0.02), scrollZooms: false) { total += log(f) } }  // 40 notches in 0.8 s
        #expect(abs(total + log(4)) < 1e-4, "the notches of one spin sum to at most a factor of 4: \(exp(total))")
        if case .zoom(let f) = law.handle(wheel(1, t: 2.0), scrollZooms: false) { #expect(abs(f - exp(-0.1)) < 1e-6, "a second later it is a new gesture") } else { Issue.record("zoom") }
    }

    @Test func aWheelSpunForSecondsWithoutAPauseZoomsByAFactorOfFourInAllAndAPauseStartsANewGesture() {
        // (review finding: the gesture was a second from its first event, so three seconds of spin summed to three times ln 4)
        var law = ScrollLaw()
        var total: Float = 0
        for i in 0..<150 { if case .zoom(let f) = law.handle(wheel(1, t: Double(i) * 0.02), scrollZooms: false) { total += log(f) } }  // 3 s
        #expect(abs(total + log(4)) < 1e-4, "three seconds of one spin: \(exp(total))")
        // 0.2 s of quiet is still the same gesture; 0.3 s is a new one.
        #expect(law.handle(wheel(1, t: 2.98 + 0.2), scrollZooms: false) == .none)
        guard case .zoom(let f) = law.handle(wheel(1, t: 2.98 + 0.2 + 0.3), scrollZooms: false) else { Issue.record("a new gesture"); return }
        #expect(abs(f - exp(-0.1)) < 1e-6)
    }

    @Test func aScrollDeltaThatIsNotANumberPansNothing() {
        var law = ScrollLaw()
        #expect(law.handle(trackpad(.nan, 4), scrollZooms: false) == .none)
        #expect(law.handle(trackpad(4, .infinity), scrollZooms: false) == .none)
        #expect(law.handle(trackpad(4, -.infinity), scrollZooms: false) == .none)
        #expect(law.handle(trackpad(4, 3), scrollZooms: false) == .pan(dx: 4, dy: 3))
    }

    @Test func aTrackpadPanIsNotCappedAndEndingAGestureResetsTheAllowance() {
        var law = ScrollLaw()
        var summed: Float = 0
        for _ in 0..<1000 { if case .pan(_, let dy) = law.handle(trackpad(0, 5), scrollZooms: false) { summed += dy } }
        #expect(summed == 5000, "a pan is a grab: every point counts")
        var z = ScrollLaw()
        _ = z.handle(trackpad(0, 400, phase: .began), scrollZooms: true)
        #expect(z.zoomed > 0)
        _ = z.handle(trackpad(0, 0, phase: .ended), scrollZooms: true)
        #expect(z.zoomed == 0)
    }
}
