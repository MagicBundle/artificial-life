import Foundation

@testable import AlifeCore

/// A preferences domain in memory, for the tests of `RepoPathStore`: nothing is written to the owner's Library.
final class MemoryDomain: PreferenceDomain, @unchecked Sendable {
    private let lock = NSLock()
    private var values: [String: String] = [:]

    func string(forKey key: String) -> String? {
        lock.withLock { values[key] }
    }

    func store(_ value: String, forKey key: String) {
        lock.withLock { values[key] = value }
    }

    /// For setting up a domain that already holds a value, as the Lab's does before the shared one.
    func set(_ value: String, forKey key: String) { store(value, forKey: key) }
}
