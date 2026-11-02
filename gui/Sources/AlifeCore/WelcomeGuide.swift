import Foundation

/// Welcome (docs/APP.md §3.1, PR-G): the one card a person without a working engine sees on the Worlds page, as a checklist of three things
/// (the alife folder, the engine, a check that it starts) with the one step that is next, what is wrong and what to do about it. Pure: the
/// page draws the answer, `WelcomeModel` reads the facts. The folder is an existing checkout (D2: the app neither clones nor runs an installer).
package enum WelcomeGuide {
    /// The engine's own answer to `alife --help`.
    package enum Health: Equatable, Sendable {
        case unchecked
        case checking
        case passed
        /// What went wrong: a sentence and the last lines the engine printed.
        case failed(Trouble)
    }

    /// Something ended badly (`uv sync`, `alife --help`): a sentence, and the last lines it printed (the card shows them and copies them).
    package struct Trouble: Equatable, Sendable {
        package var summary: String
        package var lines: [String]

        package init(summary: String, lines: [String] = []) {
            self.summary = summary
            self.lines = lines
        }
    }

    package struct Facts: Equatable, Sendable {
        /// The folder field as typed (trimmed); empty when nothing is chosen.
        package var folderPath: String
        package var folderExists: Bool
        /// The folder looks like a checkout of alife (`RepoFolder.isCheckout`): set-up is offered only in one.
        package var isCheckout: Bool
        package var engineReady: Bool
        package var uvFound: Bool
        /// `uv sync` runs now.
        package var setupRunning: Bool
        /// Another job (a run, a check) holds the engine's one job slot.
        package var otherJobRunning: Bool
        package var setupFailure: Trouble?
        package var health: Health
        /// Welcome was shown in this session: a check that takes a moment is then told, instead of happening behind a page that is
        /// already in order (a launch with a working folder never shows Welcome).
        package var announced: Bool

        package init(folderPath: String, folderExists: Bool, isCheckout: Bool = true, engineReady: Bool, uvFound: Bool = true, setupRunning: Bool = false,
                     otherJobRunning: Bool = false, setupFailure: Trouble? = nil, health: Health = .unchecked, announced: Bool = false) {
            self.folderPath = folderPath
            self.folderExists = folderExists
            self.isCheckout = isCheckout
            self.engineReady = engineReady
            self.uvFound = uvFound
            self.setupRunning = setupRunning
            self.otherJobRunning = otherJobRunning
            self.setupFailure = setupFailure
            self.health = health
            self.announced = announced
        }
    }

    /// What the card shows; nil when the engine is there and answers (the Worlds page goes on without it).
    package enum Step: Equatable, Sendable {
        case chooseFolder(chosen: String?)
        /// The folder exists and has no engine, and it does not look like alife: nothing is installed in it.
        case notAlife(chosen: String)
        case setUpEngine(uvFound: Bool)
        case settingUp
        case setupFailed(Trouble)
        case checking
        case checkFailed(Trouble)
    }

    package static func step(_ facts: Facts) -> Step? {
        if !facts.folderExists { return .chooseFolder(chosen: facts.folderPath.isEmpty ? nil : facts.folderPath) }
        // (a set-up that runs, or that failed, is told before the engine's answer: a repair of an engine that does not start shows its progress,
        // and the old failure is not shown beside it)
        if facts.setupRunning { return .settingUp }
        if let failure = facts.setupFailure { return .setupFailed(failure) }
        if !facts.engineReady {
            if !facts.isCheckout { return .notAlife(chosen: facts.folderPath) }
            return .setUpEngine(uvFound: facts.uvFound)
        }
        switch facts.health {
        case .failed(let why): return .checkFailed(why)
        case .passed: return nil
        case .unchecked, .checking: return facts.announced ? .checking : nil
        }
    }

    // MARK: The checklist

    package enum Mark: Equatable, Sendable { case done, current, waiting }

    package struct Item: Equatable, Sendable {
        package var title: String
        package var detail: String?
        package var mark: Mark
    }

    /// The three things, in order, each done, current or waiting for the one before.
    package static func checklist(_ facts: Facts) -> [Item] {
        let found = facts.folderExists && (facts.isCheckout || facts.engineReady)
        let folder = Item(title: "Find the alife folder", detail: found ? facts.folderPath : nil, mark: found ? .done : .current)
        let engine: Item = {
            if !found { return Item(title: "Set up the engine", detail: nil, mark: .waiting) }
            if facts.setupRunning { return Item(title: "Set up the engine", detail: "Installing now", mark: .current) }
            if facts.engineReady { return Item(title: "Set up the engine", detail: "Installed in the folder", mark: .done) }
            return Item(title: "Set up the engine", detail: facts.setupRunning ? "Installing now" : nil, mark: .current)
        }()
        let check: Item = {
            guard facts.engineReady, !facts.setupRunning else { return Item(title: "Check that it starts", detail: nil, mark: .waiting) }
            if facts.health == .passed { return Item(title: "Check that it starts", detail: "It answers", mark: .done) }
            return Item(title: "Check that it starts", detail: nil, mark: .current)
        }()
        return [folder, engine, check]
    }

    // MARK: Words

    package static func title(_ step: Step) -> String {
        switch step {
        case .chooseFolder: "Where is alife?"
        case .notAlife: "This is not the alife folder"
        case .setUpEngine(let uvFound): uvFound ? "Set up the engine" : "uv is needed first"
        case .settingUp: "Setting up the engine"
        case .setupFailed: "The engine could not be set up"
        case .checking: "Checking the engine"
        case .checkFailed: "The engine does not start"
        }
    }

    package static func explanation(_ step: Step) -> String {
        switch step {
        case .chooseFolder(let chosen):
            if chosen != nil {
                return "The folder you chose is not there (it may have been moved, or its drive may not be connected). "
                    + "Choose the folder that holds alife (the one you cloned), or fix the path."
            }
            return "alife did not find its folder in the usual places (\(RepoFolderPlaces.usual.joined(separator: ", "))). "
                + "Choose the folder that holds alife (the one you cloned)."
        case .notAlife(let chosen):
            return "\(chosen) does not look like alife (it has no pyproject.toml of alife and no src/alife), so nothing is installed in it. "
                + "Choose the folder you cloned."
        case .setUpEngine(let uvFound):
            return uvFound
                ? "This folder has no engine yet (.venv/bin/alife is missing). Setting it up installs the packages with uv. It takes a minute or two, "
                    + "longer the first time, and it removes nothing."
                : "alife installs its packages with uv, and uv was not found on this Mac. Install it, then press Check again."
        case .settingUp:
            return "uv is installing the packages. This takes a minute or two, longer the first time. You can leave this page; it carries on."
        case .setupFailed(let failure):
            return failure.summary
        case .checking:
            return "Asking the engine to start (alife --help). This takes a few seconds."
        case .checkFailed(let trouble):
            return trouble.summary
        }
    }

    /// The command that sets up an engine by hand: the same arguments as the button (a test pins this to `RunController.setupArguments`).
    package static let setupCommand = "uv sync --inexact --extra viewer --extra video"

    /// How to install uv with Homebrew (the uv page has the other ways).
    package static let uvInstallCommand = "brew install uv"

    /// The page that explains how to install uv.
    package static let uvPage = "https://docs.astral.sh/uv/getting-started/installation/"

    /// For someone who does not have alife yet (D2: the app does not fetch it). Copyable as one block.
    package static func fetchSteps(into folder: String = "~/artificial-life") -> String {
        """
        brew install git uv
        git clone https://github.com/MagicBundle/artificial-life \(folder)
        cd \(folder)
        \(setupCommand)
        """
    }

    package static let fetchExplanation =
        "alife is a folder you fetch once. In Terminal, run these four lines (skip what you already have), then choose the folder here. "
        + "If git asks for a login or says it cannot find the project, you need access to it first: ask whoever gave you alife."

    /// What the card's Copy button puts on the clipboard for a failure: the sentence and the last lines.
    package static func report(for failure: Trouble) -> String {
        ([failure.summary] + failure.lines).joined(separator: "\n")
    }

    /// The last non-empty lines of an output, for the card (a progress bar is many lines; the last few say where it is).
    package static func tail(_ lines: [String], count: Int = 8) -> [String] {
        let kept = lines.map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty && !$0.hasPrefix("$ ") }  // (the log's first line is the command with its long path)
        return Array(kept.suffix(count))
    }

    /// The sentence for `uv sync` that ended with `status` (`signalled`: by a signal), with what it printed last.
    package static func setupFailure(status: Int32, signalled: Bool, lines: [String]) -> Trouble {
        let summary: String
        if signalled {
            summary = "The set-up was stopped before it finished. Nothing was removed; press Set up again to carry on."
        } else if status == 130 {
            summary = "The set-up was cancelled. Nothing was removed; press Set up again to carry on."
        } else {
            summary = "uv stopped with exit code \(status), so the engine is not complete. The last lines it printed are below; "
                + "a network that is down is a common cause."
        }
        return Trouble(summary: summary, lines: tail(lines))
    }

    /// For a set-up that could not even start (uv is not executable, say).
    package static func setupFailure(summary: String, lines: [String]) -> Trouble {
        Trouble(summary: summary, lines: tail(lines))
    }

    /// The sentence for `alife --help` that did not answer well.
    package static func healthFailure(status: Int32?, timedOut: Bool, lines: [String]) -> Trouble {
        let head: String
        if timedOut {
            head = "alife --help did not answer within a minute, so the engine is not ready. Set it up again, or check the folder in Terminal."
        } else if let status {
            head = "alife --help ended with exit code \(status): the engine is in the folder but does not start. Setting it up again usually mends it."
        } else {
            head = "alife could not be started from this folder. Setting it up again usually mends it."
        }
        return Trouble(summary: head, lines: tail(lines, count: 6))
    }
}

/// The usual places the app looks for the alife folder, for the words (the places themselves are `RepoFolder.candidates`).
package enum RepoFolderPlaces {
    package static let usual = ["~/artificial-life", "~/alife"]
}
