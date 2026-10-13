import Foundation
import Testing

@testable import AlifeCore

/// The command lines of the 2D GUI before the 3D work (the 3D PR 5 package split and `World3DLevers`). The table was
/// captured from the sources as they were at origin/main e1f718e, before any change to `Levers.swift` or
/// `DirectorSettings.swift`, by running `argvCases()` against them. A 2D run, fork or resume must keep producing
/// exactly these arguments: that is what the 3D levers must not disturb. The cases named `real-preset-*` (a preset
/// chosen through its file in the repository, as the app does) and `engine-*` (a fork or resume of a run whose
/// config.json is what the engine writes, with `world.dims = 2` and the 3D keys at 0) were added after a review, and
/// captured from the same sources in the same way.
@Suite struct ArgvIdentityTests {
    static let expected: [String: [String]] = [
        "new/tiny": ["run", "--preset", "tiny", "--seed", "1", "--out", "runs/t"],
        "new/dev": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t"],
        "new/64k": ["run", "--preset", "64k", "--seed", "1", "--out", "runs/t"],
        "new/256k": ["run", "--preset", "256k", "--seed", "1", "--out", "runs/t"],
        "new/fast": ["run", "--preset", "tiny", "--overlay", "fast", "--seed", "1", "--out", "runs/t"],
        "new/blank-seed-steps": ["run", "--preset", "dev", "--steps", "2000", "--out", "runs/t"],
        "new/steps-padded": ["run", "--preset", "dev", "--seed", "1", "--steps", "750", "--out", "runs/spaced"],
        "new/rate_min": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t", "--set", "mutation.rate_min=0.01"],
        "new/sigma_min": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t", "--set", "mutation.sigma_min=0.01"],
        "new/patch_drift": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t", "--set", "fields.patch_drift=0.02"],
        "new/patch_drift-zero": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t", "--set", "fields.patch_drift=0.0"],
        "new/immigration_floor": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t", "--set", "population.immigration_floor=500"],
        "new/immigrants_per_step": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t", "--set", "population.immigrants_per_step=8"],
        "new/immigration_every+count": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t", "--set", "population.immigration_every=1000", "--set", "population.immigration_count=16"],
        "new/season": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t", "--set", "fields.season_period=20000", "--set", "fields.season_amplitude=0.6", "--set", "fields.season_mode=per_patch"],
        "new/all-levers": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t", "--set", "mutation.rate_min=0.002", "--set", "mutation.sigma_min=0.004", "--set", "fields.patch_drift=0.01", "--set", "population.immigration_floor=300", "--set", "population.immigrants_per_step=16", "--set", "population.immigration_every=2000", "--set", "population.immigration_count=4", "--set", "fields.season_period=10000", "--set", "fields.season_amplitude=1.0", "--set", "fields.season_mode=per_patch"],
        "new/levers-equal-to-defaults": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t"],
        "new/lever-spellings": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t", "--set", "mutation.rate_min=5e-05", "--set", "mutation.sigma_min=0.005", "--set", "fields.patch_drift=0.01"],
        "new/invalid-levers-skipped": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t"],
        "new/director-live": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t", "--director", "live"],
        "new/director-replay": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t", "--director", "replay"],
        "new/director-trickster-rule": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t", "--director", "live", "--persona", "trickster", "--set", "director.rule_every=2000"],
        "new/director-architect-goal": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t", "--director", "live", "--persona", "architect", "--goal=keep clades above 5"],
        "new/director-llm": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t", "--director", "live", "--llm-every", "50000", "--llm-url=http://127.0.0.1:8080/v1/chat/completions", "--llm-model=other-model"],
        "new/director-llm-defaults": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t", "--director", "live", "--llm-every", "25000"],
        "new/director-random": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t", "--director", "live", "--persona", "random", "--yoke-log=runs/arm-a/director.jsonl"],
        "new/director-with-levers-fast": ["run", "--preset", "64k", "--overlay", "fast", "--seed", "1", "--out", "runs/t", "--set", "fields.season_period=5000", "--set", "fields.season_amplitude=0.5", "--director", "live"],
        "name/taken-counter": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t-2"],
        "name/taken-padded": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/run008"],
        "name/taken-suffix": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/baseline-3"],
        "name/free": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/fresh"],
        "fork/live-untouched": ["run", "--init-from", "runs/src-live", "--out", "runs/child", "--director", "live"],
        "fork/live-seed-steps": ["run", "--init-from", "runs/src-live", "--seed", "7", "--steps", "300000", "--out", "runs/child", "--director", "live"],
        "fork/live-lever-changes": ["run", "--init-from", "runs/src-live", "--out", "runs/child", "--set", "mutation.rate_min=0.002", "--set", "fields.patch_drift=0.005", "--set", "population.immigration_floor=0", "--set", "fields.season_mode=global", "--director", "live"],
        "fork/live-director-off": ["run", "--init-from", "runs/src-live", "--out", "runs/child"],
        "fork/live-persona-llm": ["run", "--init-from", "runs/src-live", "--out", "runs/child", "--director", "live", "--persona", "architect", "--goal=diversify", "--llm-every", "0", "--set", "director.rule_every=1000"],
        "fork/plain-untouched": ["run", "--init-from", "runs/src-plain", "--out", "runs/child"],
        "fork/plain-director-live": ["run", "--init-from", "runs/src-plain", "--out", "runs/child", "--director", "live"],
        "fork/plain-levers": ["run", "--init-from", "runs/src-plain", "--out", "runs/child", "--set", "fields.season_period=100", "--set", "fields.season_amplitude=0.1"],
        "resume/live-untouched": ["resume", "runs/src-live"],
        "resume/live-steps": ["resume", "runs/src-live", "--steps", "50000"],
        "resume/live-off": ["resume", "runs/src-live", "--director", "off"],
        "resume/live-replay": ["resume", "runs/src-live", "--director", "replay"],
        "resume/live-llm-on": ["resume", "runs/src-live", "--llm-every", "10000"],
        "resume/plain": ["resume", "runs/src-plain"],
        "resume/plain-steps": ["resume", "runs/src-plain", "--steps", "1000"],
        "new/real-preset-tiny": ["run", "--preset", "tiny", "--seed", "1", "--out", "runs/t"],
        "new/real-preset-dev": ["run", "--preset", "dev", "--seed", "1", "--out", "runs/t"],
        "new/real-preset-64k": ["run", "--preset", "64k", "--seed", "1", "--out", "runs/t"],
        "new/real-preset-256k": ["run", "--preset", "256k", "--seed", "1", "--out", "runs/t"],
        "new/real-preset-levers-fast-director": ["run", "--preset", "tiny", "--overlay", "fast", "--seed", "1", "--out", "runs/t", "--set", "mutation.rate_min=0.01", "--set", "fields.season_period=5000", "--set", "fields.season_amplitude=0.5", "--director", "live"],
        "fork/engine-dev-untouched": ["run", "--init-from", "runs/eng-dev", "--out", "runs/child"],
        "fork/engine-dev-seed-steps": ["run", "--init-from", "runs/eng-dev", "--seed", "3", "--steps", "5000", "--out", "runs/child"],
        "fork/engine-dev-levers": ["run", "--init-from", "runs/eng-dev", "--out", "runs/child", "--set", "mutation.rate_min=0.002", "--set", "fields.patch_drift=0.01", "--set", "population.immigration_floor=50", "--set", "fields.season_period=100", "--set", "fields.season_amplitude=0.3"],
        "fork/engine-dev-levers-to-defaults": ["run", "--init-from", "runs/eng-dev", "--out", "runs/child", "--set", "mutation.sigma_min=0.003"],
        "fork/engine-dev-director-live": ["run", "--init-from", "runs/eng-dev", "--out", "runs/child", "--director", "live"],
        "fork/engine-tiny-untouched": ["run", "--init-from", "runs/eng-tiny", "--out", "runs/child"],
        "fork/engine-tiny-director-persona": ["run", "--init-from", "runs/eng-tiny", "--out", "runs/child", "--director", "live", "--persona", "architect", "--goal=diversify"],
        "fork/engine-live-untouched": ["run", "--init-from", "runs/eng-live", "--out", "runs/child", "--director", "live"],
        "fork/engine-live-levers-back": ["run", "--init-from", "runs/eng-live", "--out", "runs/child", "--set", "mutation.rate_min=0.001", "--set", "population.immigration_floor=0", "--set", "fields.season_mode=global", "--director", "live"],
        "fork/engine-live-director-off": ["run", "--init-from", "runs/eng-live", "--out", "runs/child"],
        "fork/engine-live-persona": ["run", "--init-from", "runs/eng-live", "--out", "runs/child", "--director", "live", "--persona", "gardener", "--set", "director.rule_every=2000"],
        "resume/engine-dev-untouched": ["resume", "runs/eng-dev"],
        "resume/engine-dev-steps": ["resume", "runs/eng-dev", "--steps", "2000"],
        "resume/engine-dev-director-live": ["resume", "runs/eng-dev"],
        "resume/engine-live-untouched": ["resume", "runs/eng-live"],
        "resume/engine-live-steps": ["resume", "runs/eng-live", "--steps", "30000"],
        "resume/engine-live-replay": ["resume", "runs/eng-live", "--director", "replay"],
        "resume/engine-live-off": ["resume", "runs/eng-live", "--director", "off"],
    ]

    @Test func twoDArgumentsAreUnchanged() throws {
        let cases = try argvCases()
        #expect(Set(cases.map(\.name)) == Set(Self.expected.keys), "the table and the cases name the same configurations")
        #expect(cases.count == Self.expected.count)
        for c in cases {
            #expect(c.args == Self.expected[c.name], "argument list of case \(c.name)")
        }
    }
}
