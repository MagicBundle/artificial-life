"""The app bundle of gui/ (alife.app) and the Swift that reads it agree.

Plain text checks on `gui/scripts/make-app.sh`, `check-apps.sh`, `make-icon.swift` and the Swift
sources: they need no Swift toolchain and no macOS. The one test that builds the bundle
(`gui/scripts/check-apps.sh`: a release build, ad-hoc signing, every Info.plist key,
`codesign --verify`) runs only on macOS with `ALIFE_APP_SMOKE=1`, and is `slow`.

There is one app since PR-H of the one-app program (docs/APP.md section 6): the Lab, Storm Worlds and
the preview are retired, and the script builds only alife.app.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

GUI = Path(__file__).resolve().parents[1] / "gui"
SCRIPTS = GUI / "scripts"
CORE = GUI / "Sources" / "AlifeCore"


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def swift_constant(source: str, name: str) -> str:
    m = re.search(rf'static let {name} = "([^"]*)"', source)
    assert m, f"{name} not found"
    return m.group(1)


def test_the_info_key_and_value_are_the_ones_the_script_writes() -> None:
    plan = text(CORE / "AppPlan.swift")
    script = text(SCRIPTS / "make-app.sh")
    key = swift_constant(plan, "infoKey")
    assert f"<key>{key}</key><string>$A_MODE</string>" in script
    # One value: the script's A_MODE is "alife" (there is no LaunchMode enum any more, and no lab or world).
    assert re.findall(r'A_MODE="(\w+)"', script) == ["alife"]
    assert swift_constant(plan, "name") == "alife"
    assert not (CORE / "LaunchMode.swift").exists()


def test_the_bundle_id_is_the_preferences_domain_the_old_apps_shared() -> None:
    script = text(SCRIPTS / "make-app.sh")
    plan = text(CORE / "AppPlan.swift")
    store = text(CORE / "RepoPathStore.swift")
    bundle_id = swift_constant(plan, "bundleID")
    assert bundle_id == "io.github.magicbundle.alife"
    assert f'DEFAULT_BUNDLE_ID="{bundle_id}"' in script
    assert 'ALIFE_BUNDLE_ID="${ALIFE_BUNDLE_ID:-$DEFAULT_BUNDLE_ID}"' in script
    # RepoPathStore.sharedSuite is that same id (AppPlan.bundleID); the old apps' ids are only read.
    assert "sharedSuite = AppPlan.bundleID" in store
    assert swift_constant(store, "labDomain") == "io.github.magicbundle.alifecontrol"
    assert swift_constant(store, "stormDomain") == "io.github.magicbundle.stormworlds"
    # The old ids are not bundle ids the script can build any more.
    assert "alifecontrol" not in script and "stormworlds" not in script
    assert 'A_ID="$ALIFE_BUNDLE_ID"' in script


def test_the_script_builds_one_app_named_alife_at_version_0_2_0() -> None:
    script = text(SCRIPTS / "make-app.sh")
    assert 'A_NAME="alife"' in script and 'A_EXEC="alife"' in script and 'A_DIR="alife.app"' in script
    assert 'VERSION="${VERSION:-0.2.0}"' in script
    assert 'A_CATEGORY="public.app-category.simulation-games"' in script
    # The retired options are refused, not ignored.
    assert re.search(r"--app\|--app=\*\|--preview\) die ", script)
    assert "Storm Worlds.app" in script  # (named only to print how to Trash it)
    # It signs ad-hoc only when told to: the smoke test must never reach the keychain.
    check = text(SCRIPTS / "check-apps.sh")
    assert "--sign -" in check
    assert "--install" not in check.replace("installs nothing", "").replace('"--install"', "")


def test_install_replaces_alife_and_touches_no_old_app() -> None:
    script = text(SCRIPTS / "make-app.sh")
    install = script[script.index("install_app()") : script.index("trash_note()")]
    # The install removes only temporary names (alife.app is moved aside first); old apps are listed only.
    assert sorted(set(re.findall(r'rm -rf "([^"]+)"', install))) == ["$old", "$tmp"]
    assert 'dest="$INSTALL_DIR/$A_DIR"' in script
    for name in ("AlifeControl.app", "Storm Worlds.app", "alife Preview.app"):
        assert name in script
    assert "were not touched" in script and "Trash" in script
    # the note is printed on a plain build as well as after an install
    assert script.rstrip().endswith("trash_note")


def test_install_is_atomic_and_refuses_a_trial_id() -> None:
    script = text(SCRIPTS / "make-app.sh")
    install = script[script.index("install_app()") : script.index("trash_note()")]
    # copied to a temporary name, the old app moved aside and put back if the move fails
    assert (
        'ditto "$bundle" "$tmp"' in install
        and 'mv "$tmp" "$dest"' in install
        and 'mv "$old" "$dest"' in install
    )
    assert install.index("ditto") < install.index('mv "$dest" "$old"')
    # --install with another ALIFE_BUNDLE_ID is refused before building
    assert 'DEFAULT_BUNDLE_ID="io.github.magicbundle.alife"' in script
    assert script.index("--install needs the default bundle id") < script.index("swift build -c release")


def test_the_icon_script_has_the_alife_variant_and_the_script_asks_for_it() -> None:
    icon = text(SCRIPTS / "make-icon.swift")
    assert 'variant == "alife"' in icon
    script = text(SCRIPTS / "make-app.sh")
    assert 'A_ICON="alife"' in script
    assert '"$SCRIPT_DIR/make-icon.swift" "$WORK/$A_ICON.iconset" "$A_ICON"' in script


def test_the_smoke_test_checks_the_one_bundle_and_the_refusals() -> None:
    check = text(SCRIPTS / "check-apps.sh")
    assert 'ALIFE="$OUT/alife.app"' in check
    assert "io.github.magicbundle.alife alife alife alife" in check
    assert "AlifeControl.app" not in check and "Storm Worlds.app" not in check
    assert '"--app lab" "--app both" "--preview"' in check


def test_the_readme_documents_the_one_app() -> None:
    readme = text(GUI / "README.md")
    assert "alife.app" in readme
    assert "check-apps.sh" in readme
    assert "--app both" not in readme


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("swift") is None or os.environ.get("ALIFE_APP_SMOKE") != "1",
    reason="builds the app bundles: macOS with Swift, and ALIFE_APP_SMOKE=1",
)
def test_the_bundle_builds_and_verifies(tmp_path: Path) -> None:
    out = tmp_path / "apps"
    done = subprocess.run(
        [str(SCRIPTS / "check-apps.sh"), "--out", str(out)], capture_output=True, text=True, timeout=1800
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert (out / "alife.app" / "Contents" / "Info.plist").is_file()
