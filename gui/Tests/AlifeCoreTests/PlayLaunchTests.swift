import Foundation
import Testing

@testable import AlifeCore

/// The command lines of `alife play` that the launcher builds (DESIGN §33.2.8). `tests/test_gui_sync.py` checks that the
/// flags are the ones `alife play --help` lists.
@Suite struct PlayLaunchTests {
    @Test func aWorldIsResumedAtOneTimesByDefault() {
        #expect(PlayLaunch.arguments(resume: PlayResume(run: "planet_a")) == ["play", "runs/planet_a", "--rate", "1"])
    }

    @Test func aResumeCarriesEverythingItIsAskedFor() {
        let all = PlayResume(run: "p", checkpoint: 12_000, truncate: true, rate: .max, director: .rules, replay: true)
        #expect(PlayLaunch.arguments(resume: all) == [
            "play", "runs/p", "--checkpoint", "12000", "--truncate", "--replay", "--rate", "max", "--director", "rules",
        ])
        #expect(PlayLaunch.arguments(resume: PlayResume(run: "p", rate: nil)) == ["play", "runs/p"])
        #expect(PlayLaunch.arguments(resume: PlayResume(run: "p", rate: .multiplier(0.25)))
                == ["play", "runs/p", "--rate", "0.25"])
    }

    @Test func aPlanetIsAForkWithTheWeatherOnAndSpunUp() {
        let planet = PlanetSpec(source: "64k3d_s0", sourceStep: 500_000, name: "planet_a")
        #expect(PlayLaunch.arguments(planet: planet) == [
            "play", "--init-from", "runs/64k3d_s0:500000", "--overlay", "storm", "--set", "director.enabled=true",
            "--weather-spinup", "20000", "--houses", "auto", "--out", "runs/planet_a", "--rate", "1",
        ])
    }

    @Test func theOverlayIsChosenFromTheSizeOfTheSource() {
        // The 160 world has its own weather overlay; the 320 world, and a source of unknown size, `storm`. Explicit, no alias.
        for (size, overlay) in [(160.0, "storm160"), (320.0, "storm"), (64.0, "storm")] {
            let planet = PlanetSpec(source: "s", sourceStep: 450_000, name: "m", sourceWorldSize: size)
            #expect(PlayLaunch.arguments(planet: planet).starts(with: ["play", "--init-from", "runs/s:450000", "--overlay", overlay,
                                                                       "--set"]), "\(size)")
        }
        #expect(PlayLaunch.arguments(planet: PlanetSpec(source: "s", name: "m"))[3...4] == ["--overlay", "storm"][...])
        #expect(PlanetSpec.overlay(forWorldSize: 160) == "storm160" && PlanetSpec.overlay(forWorldSize: nil) == "storm")
        #expect(PlanetSpec.defaultSpinup(forWorldSize: 160) == 36_000)
        #expect(PlanetSpec.defaultSpinup(forWorldSize: 320) == 20_000 && PlanetSpec.defaultSpinup(forWorldSize: nil) == 20_000)
        // A weather-on source is forked as it is, whatever its size.
        let on = PlanetSpec(source: "p", name: "q", storm: false, weatherSpinup: 0, sourceWorldSize: 160)
        #expect(!PlayLaunch.arguments(planet: on).contains("--overlay"))
    }

    @Test func aWeatherOnSourceIsForkedWithoutTheOverlayAndTheSpinup() {
        let planet = PlanetSpec(source: "p", name: "q", storm: false, weatherSpinup: 0, directorEnabled: false,
                                houses: .clade, seed: 5, rate: nil, director: .off)
        #expect(PlayLaunch.arguments(planet: planet) == [
            "play", "--init-from", "runs/p", "--houses", "clade", "--seed", "5", "--out", "runs/q",
        ])
    }

    @Test func theNameIsTrimmedInTheOutArgument() {
        let args = PlayLaunch.arguments(planet: PlanetSpec(source: "p", name: "  x "))
        #expect(args.contains("runs/x"))
    }

    @Test func aPlanetThatCannotBeLaunchedIsExplained() {
        let ok = PlanetSpec(source: "s", name: "new")
        #expect(ok.validate(existing: ["s"]).isEmpty)
        #expect(PlanetSpec(source: "", name: "new").validate(existing: []) == ["Choose a source run."])
        #expect(PlanetSpec(source: "s", name: " ").validate(existing: []) == ["Enter a name for the planet."])
        #expect(PlanetSpec(source: "s", name: "a b").validate(existing: []).count == 1)
        #expect(PlanetSpec(source: "s", name: "../x").validate(existing: []).count == 1)
        #expect(PlanetSpec(source: "s", name: "s").validate(existing: ["s"]) == ["'runs/s' exists."])
        // The weather overlay needs a spin-up (the rest state has no storms for about 15,000 steps); none without it.
        #expect(PlanetSpec(source: "s", name: "n", storm: true, weatherSpinup: 0).validate(existing: []).count == 1)
        #expect(PlanetSpec(source: "s", name: "n", storm: false, weatherSpinup: 100).validate(existing: []).count == 1)
        #expect(PlanetSpec(source: "s", name: "n", directorEnabled: false, director: .rules).validate(existing: []).count == 1)
        #expect(PlanetSpec(source: "s", name: "n", seed: -1).validate(existing: []).count == 1)
    }

    @Test func everyFlagIsAFlagOfTheEngine() {
        // The names `tests/test_gui_sync.py` compares with `alife play --help`.
        let all = [
            PlayLaunch.arguments(resume: PlayResume(run: "p", checkpoint: 1, truncate: true, rate: .max,
                                                    director: .rules, replay: true)),
            PlayLaunch.arguments(planet: PlanetSpec(source: "s", sourceStep: 1, name: "n", seed: 1, director: .rules)),
        ].joined()
        let flags = Set(all.filter { $0.hasPrefix("--") })
        #expect(flags == [
            "--checkpoint", "--truncate", "--replay", "--rate", "--director", "--init-from", "--overlay", "--set",
            "--weather-spinup", "--houses", "--seed", "--out",
        ])
    }
}
