import Foundation

/// The phase of a scroll gesture or of its momentum (`NSEvent.Phase`, without AppKit).
package enum ScrollGesturePhase: Sendable {
    case none, mayBegin, began, changed, stationary, ended, cancelled
}

/// One scroll event as the law sees it: the deltas in points (a trackpad) or lines (a wheel), where it came from and when.
package struct ScrollEvent: Equatable, Sendable {
    package var dx: Float
    package var dy: Float
    /// `hasPreciseScrollingDeltas`: a trackpad or a Magic Mouse, whose deltas are points; a wheel's are lines.
    package var precise: Bool
    package var phase: ScrollGesturePhase
    package var momentum: ScrollGesturePhase
    /// Seconds from a monotonic origin.
    package var time: Double

    package init(dx: Float = 0, dy: Float, precise: Bool, phase: ScrollGesturePhase = .none, momentum: ScrollGesturePhase = .none, time: Double = 0) {
        self.dx = dx
        self.dy = dy
        self.precise = precise
        self.phase = phase
        self.momentum = momentum
        self.time = time
    }
}

/// What a scroll event does to the view.
package enum ScrollOutcome: Equatable, Sendable {
    case none
    /// A pan: the map follows the fingers by this many points (+y down).
    case pan(dx: Float, dy: Float)
    /// A zoom about the pointer: the factor on the distance (below 1 comes closer).
    case zoom(factor: Float)
}

/// The scroll law of the World window (DESIGN §36.5, decision D8), the Maps convention on a Mac:
///
/// * a **trackpad's** two-finger scroll **pans** (momentum included: the system's own momentum events are the glide), unless
///   View > "Scroll zooms" is on, which restores v1's scroll-is-zoom;
/// * the **mouse wheel** zooms about the pointer (0.1 per line), and so does the trackpad with Scroll zooms on (0.006 per point,
///   v1's was 0.01);
/// * **momentum never zooms**: an event with a momentum phase is ignored for zoom, so a flick does not carry on across the range;
/// * **a gesture cannot cross the range**: the summed `|ln factor|` of one gesture (from `began` to `ended`, and for an unphased
///   wheel the events with no pause of a quarter of a second between them) is capped at `ln 4`. v1 read the whole range, `ln(0.8 L / 4)` = 4.2, as 420 points of
///   summed delta at 0.01: one hard flick with its tail. An unphased wheel's gesture ends at a quiet gap of a quarter of a second
///   (not a second after its first event: a wheel spun without a pause is one gesture, however long, so it cannot zoom by `ln 4`
///   every second).
///
/// Pinch is not a scroll and is not capped: it is a direct manipulation of the scale.
package struct ScrollLaw: Equatable, Sendable {
    package static let preciseCoefficient: Float = 0.006
    package static let wheelCoefficient: Float = 0.1
    /// The most one event may zoom by, as `|ln factor|` (v1's cap).
    package static let eventCap: Float = 1.5
    /// The most one gesture may zoom by, as `|ln factor|`: a factor of 4 either way.
    package static let gestureCap: Float = log(4)
    /// The pause that ends an unphased wheel's gesture, in seconds: the events of a wheel with less than this between them are one
    /// gesture.
    package static let wheelGap: Double = 0.25

    /// The summed `|ln factor|` of the gesture in progress, and when it began.
    package private(set) var zoomed: Float = 0
    package private(set) var began: Double = -.infinity
    /// When the last unphased wheel event came.
    package private(set) var lastWheel: Double = -.infinity

    package init() {}

    /// What `event` does. `scrollZooms` is the View menu's switch.
    package mutating func handle(_ event: ScrollEvent, scrollZooms: Bool) -> ScrollOutcome {
        if event.phase == .began || event.phase == .mayBegin { reset(at: event.time) }
        defer { if event.phase == .ended || event.phase == .cancelled { zoomed = 0 } }
        if event.precise, !scrollZooms {
            // (AppKit gives no NaN; a delta that is not a number must not reach the grab, which would put the view at the origin)
            guard event.dx.isFinite, event.dy.isFinite else { return .none }
            return event.dx == 0 && event.dy == 0 ? .none : .pan(dx: event.dx, dy: event.dy)
        }
        // A zoom. Momentum never zooms.
        guard event.momentum == .none else { return .none }
        if event.phase == .none {
            if event.time - lastWheel > Self.wheelGap { reset(at: event.time) }
            lastWheel = event.time
        }
        let coefficient = event.precise ? Self.preciseCoefficient : Self.wheelCoefficient
        guard event.dy.isFinite else { return .none }
        var exponent = max(min(-event.dy * coefficient, Self.eventCap), -Self.eventCap)
        let allowed = max(Self.gestureCap - zoomed, 0)
        if abs(exponent) > allowed { exponent = exponent < 0 ? -allowed : allowed }
        guard exponent != 0 else { return .none }
        zoomed += abs(exponent)
        return .zoom(factor: exp(exponent))
    }

    private mutating func reset(at time: Double) {
        zoomed = 0
        began = time
    }
}
