import Combine
import Foundation

/// What a control that must not be rebuilt while it is open reads, held apart from the objects that change all the time.
///
/// `PlayController` publishes with every record that is not a frame (the clock four times a second, the storms, the environment, the
/// focus: `revision`), `StreamController` with every frame, `WorldModel` with a toast or a selection. A `Menu`, or a `Picker` drawn
/// as a pop-up, that observes one of them is evaluated again each time, and AppKit tears down the menu that is open and builds it
/// again: the menu flashes in and out of existence under the pointer. A control reads this object instead. It is woken by its
/// `sources`, computes the inputs of the control (`compute`) on the next turn of the main loop (a burst of publishes is one
/// wake) and publishes them only when they differ from the last ones: while what the control shows is the same, nothing it
/// observes changes and its body is not evaluated.
///
/// `refresh()` also works when no source publishes (a frame changes the colour mode and `PlayController` does not publish a frame;
/// a default of `UserDefaults` is not observable): the owner calls it when it changes such an input.
@MainActor
final class DistinctState<Value: Equatable>: ObservableObject {
    @Published private(set) var value: Value
    /// The inputs as they are now; nil when the owner is gone (the last value stays).
    private let compute: @MainActor () -> Value?
    private var subscriptions: [AnyCancellable] = []
    private var wakePending = false
    /// How many times the inputs changed (and so how many times a control that reads them was evaluated again), for the tests.
    private(set) var changes = 0

    init(initial: Value, sources: [ObservableObjectPublisher], compute: @escaping @MainActor () -> Value?) {
        value = initial
        self.compute = compute
        for source in sources {
            subscriptions.append(source.sink { [weak self] _ in
                // (an object announces a change before it makes it: the inputs are read on the next turn)
                MainActor.assumeIsolated { self?.wake() }
            })
        }
    }

    /// Reads the inputs now; publishes them if they changed.
    func refresh() {
        guard let new = compute(), new != value else { return }
        value = new
        changes += 1
    }

    private func wake() {
        guard !wakePending else { return }
        wakePending = true
        DispatchQueue.main.async { [weak self] in
            MainActor.assumeIsolated {
                self?.wakePending = false
                self?.refresh()
            }
        }
    }
}
