import Foundation

@testable import AlifeCore

/// A preferences domain in memory, for the tests of `RepoPathStore`: nothing is written to the owner's Library.
final class MemoryDomain: TypedPreferenceDomain, @unchecked Sendable {
    private let lock = NSLock()
    private var values: [String: Any] = [:]

    func string(forKey key: String) -> String? {
        lock.withLock { values[key] as? String }
    }

    func store(_ value: String, forKey key: String) {
        lock.withLock { values[key] = value }
    }

    /// The typed side (booleans, numbers and strings), as `UserDefaults` has it.
    func object(forKey key: String) -> Any? { lock.withLock { values[key] } }

    func set(_ value: Any?, forKey key: String) {
        lock.withLock { values[key] = value }
    }

    /// Every key and value, for a test that checks nothing else was written.
    var all: [String: Any] { lock.withLock { values } }

    /// For setting up a domain that already holds a value, as the Lab's does before the shared one.
    func set(_ value: String, forKey key: String) { store(value, forKey: key) }
}
