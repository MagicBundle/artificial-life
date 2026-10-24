import Foundation

/// A hook that the tests set to count how often a view's body is evaluated (nil in the app: a call costs a nil check). A menu, a
/// picker or a popover that is evaluated again while it is open is rebuilt under the player's hand, so the views that hold one
/// are pinned by a count (DESIGN §33.4.4).
@MainActor
enum ViewProbe {
    /// Called with the view's name and the object it was made for (the tests of one world count only its own views: the suites run
    /// in parallel and host views of their own).
    static var onEvaluate: ((String, ObjectIdentifier) -> Void)?

    static func note(_ name: String, for owner: AnyObject) { onEvaluate?(name, ObjectIdentifier(owner)) }
}
