import Foundation

/// The launch argument that opens the World window on a world at start: `--world RUN` (or `--world=RUN`), RUN a directory
/// name under `runs/`.
package enum WorldLaunchArgument {
    /// The world named by the arguments (after the executable), nil when there is none or its name is not a directory name.
    package static func parse(_ arguments: [String]) -> String? {
        var value: String?
        var i = 0
        while i < arguments.count {
            let argument = arguments[i]
            if argument == "--world", i + 1 < arguments.count {
                value = arguments[i + 1]
                i += 1
            } else if argument.hasPrefix("--world=") {
                value = String(argument.dropFirst("--world=".count))
            }
            i += 1
        }
        guard let value, value.range(of: #"^[A-Za-z0-9][A-Za-z0-9._-]*$"#, options: .regularExpression) != nil else {
            return nil
        }
        return value
    }
}
