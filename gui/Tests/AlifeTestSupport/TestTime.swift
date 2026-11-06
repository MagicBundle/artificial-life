import Foundation

/// The host's time scale, `ALIFE_TEST_TIME_SCALE` (a positive number; 1 when it is unset, not a number or not positive): **the one place a
/// slower host is allowed for**, shared by both test targets. A hosted CI runner is a few shared cores with no GPU, and the full run there
/// stalls the whole test process for tens of seconds at a time (DESIGN §22.0, I91). The waits and the wall-clock upper bounds of the tests
/// multiply by this number, through `eventually` and `patiently` (the polling waits), `TestTime.limit(_:)` and `TestTime.limit(seconds:)`.
///
/// At 1 (the default, and every run on a developer's Mac) nothing changes. The scale applies only to what a test *allows* a thing to take;
/// never to what the code under test does (a gate's own limit, a poll period, the grace given to a child process), and never to a lower bound
/// ("it waited at least this long"). The CI job sets 4.
public enum TestTime {
    public static let scale: Double = {
        guard let raw = ProcessInfo.processInfo.environment["ALIFE_TEST_TIME_SCALE"], let value = Double(raw), value.isFinite, value > 0 else { return 1 }
        return value
    }()

    /// A duration allowed for something to happen, times the scale.
    public static func limit(_ duration: Duration) -> Duration { scale == 1 ? duration : duration * scale }

    /// Seconds allowed for something to happen, times the scale.
    public static func limit(seconds: Double) -> Double { seconds * scale }
}
