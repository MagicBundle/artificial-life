"""The two app bundles of gui/ (AlifeControl.app, Storm Worlds.app) and the Swift that reads them agree.

Plain text checks on `gui/scripts/make-app.sh`, `check-apps.sh`, `make-icon.swift` and the Swift
sources: they need no Swift toolchain and no macOS. The one test that builds the bundles
(`gui/scripts/check-apps.sh`: a release build, ad-hoc signing, every Info.plist key,
`codesign --verify`) runs only on macOS with `ALIFE_APP_SMOKE=1`, and is `slow`.
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


def test_the_launch_mode_key_and_values_are_the_ones_the_script_writes() -> None:
    mode = text(CORE / "LaunchMode.swift")
    script = text(SCRIPTS / "make-app.sh")
    key = swift_constant(mode, "infoKey")
    assert f"<key>{key}</key><string>$A_MODE</string>" in script
    # The values: `case lab` / `case world` in LaunchMode, A_MODE="lab" / "world" in the script.
    assert re.findall(r"^    case (\w+)$", mode, re.M)[:2] == ["lab", "world"]
    assert re.findall(r'A_MODE="(\w+)"', script) == ["lab", "world"]


def test_the_lab_bundle_id_is_the_domain_the_repo_path_is_migrated_from() -> None:
    script = text(SCRIPTS / "make-app.sh")
    lab = swift_constant(text(CORE / "RepoPathStore.swift"), "labDomain")
    assert f'LAB_BUNDLE_ID="${{BUNDLE_ID:-{lab}}}"' in script
    # The shared suite is not a bundle ID of either app (a bundle cannot open its own domain as a suite).
    suite = swift_constant(text(CORE / "RepoPathStore.swift"), "sharedSuite")
    assert not re.search(rf"{re.escape(suite)}(?![\w.])", script)  # not merely a prefix of the Lab's ID


def test_the_storm_bundle_has_its_own_identity() -> None:
    script = text(SCRIPTS / "make-app.sh")
    assert 'STORM_BUNDLE_ID="${STORM_BUNDLE_ID:-io.github.magicbundle.stormworlds}"' in script
    assert 'A_NAME="Storm Worlds"' in script
    assert 'A_DIR="Storm Worlds.app"' in script
    assert 'A_NAME="AlifeControl"' in script and 'A_DIR="AlifeControl.app"' in script


def test_the_script_builds_the_lab_by_default_and_both_on_request() -> None:
    script = text(SCRIPTS / "make-app.sh")
    assert 'WHICH="lab"' in script
    assert 'both) APPS="lab storm"' in script
    # It signs ad-hoc only when told to: the smoke test must never reach the keychain.
    check = text(SCRIPTS / "check-apps.sh")
    assert "--sign -" in check
    assert "--install" not in check.replace("installs nothing", "")


def test_the_icon_script_has_a_storm_variant_and_the_script_asks_for_it() -> None:
    icon = text(SCRIPTS / "make-icon.swift")
    assert '["lab", "storm"]' in icon
    script = text(SCRIPTS / "make-app.sh")
    assert 'A_ICON="lab"' in script and 'A_ICON="storm"' in script
    assert '"$SCRIPT_DIR/make-icon.swift" "$WORK/$A_ICON.iconset" "$A_ICON"' in script


def test_the_readme_documents_both_apps() -> None:
    readme = text(GUI / "README.md")
    assert "--app both" in readme
    assert "Storm Worlds" in readme
    assert "check-apps.sh" in readme


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("swift") is None or os.environ.get("ALIFE_APP_SMOKE") != "1",
    reason="builds the app bundles: macOS with Swift, and ALIFE_APP_SMOKE=1",
)
def test_both_bundles_build_and_verify(tmp_path: Path) -> None:
    out = tmp_path / "apps"
    done = subprocess.run(
        [str(SCRIPTS / "check-apps.sh"), "--out", str(out)], capture_output=True, text=True, timeout=1800
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert (out / "Storm Worlds.app" / "Contents" / "Info.plist").is_file()
    assert (out / "AlifeControl.app" / "Contents" / "Info.plist").is_file()
