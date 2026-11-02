#!/usr/bin/env bash
# Build, bundle and sign alife.app (the one app: docs/APP.md), and optionally install it. Run from any directory. See `make-app.sh --help`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# The bundle id is also the preferences domain the Lab and Storm Worlds shared, so the folder they stored is already there. The override
# exists for a trial bundle that must have preferences of its own (its domain is then not the owner's).
DEFAULT_BUNDLE_ID="io.github.magicbundle.alife"
ALIFE_BUNDLE_ID="${ALIFE_BUNDLE_ID:-$DEFAULT_BUNDLE_ID}"
VERSION="${VERSION:-0.2.0}"

SIGN_ARG=""
INSTALL=0
INSTALL_DIR="/Applications"
OUT_DIR=""

usage() {
  cat <<USAGE
Usage: make-app.sh [--sign IDENTITY] [--out DIR] [--install] [--install-dir DIR] [--help]

Builds the release binary, generates the icon, assembles and signs
  alife.app   the app: one window with a sidebar (Worlds, Watch, Activity, New world); id $ALIFE_BUNDLE_ID
in DIR (default $GUI_DIR/.build).

Options:
  --sign IDENTITY     Signing identity (name or hash). Default: the first
                      "Developer ID Application" identity, else the first
                      "Apple Development" identity, else ad-hoc ("-"). "--sign -" is ad-hoc and
                      never touches the keychain.
  --out DIR           Where the bundle is written (default $GUI_DIR/.build).
  --install           Copy alife.app to the install directory (default /Applications), replacing a copy
                      there. Refuses, before building, if it is running. It touches no other app: AlifeControl.app,
                      Storm Worlds.app and alife Preview.app are left where they are, and the script prints how to Trash them.
  --install-dir DIR   Install somewhere else, for example ~/Applications.
  -h, --help          Show this help.

Environment:
  ALIFE_BUNDLE_ID     Bundle identifier (default io.github.magicbundle.alife). Another id has other preferences; for a trial.
  VERSION             CFBundleShortVersionString (default 0.2.0)

The Lab and Storm Worlds are retired (docs/APP.md section 6): --app and --preview no longer exist.
Never uses sudo. Not notarized: sharing an app needs a Developer ID identity plus notarization.
USAGE
}

die() { echo "error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sign) [[ $# -ge 2 ]] || die "--sign needs an argument"; SIGN_ARG="$2"; shift 2 ;;
    --out) [[ $# -ge 2 ]] || die "--out needs an argument"; OUT_DIR="$2"; shift 2 ;;
    --install) INSTALL=1; shift ;;
    --install-dir) [[ $# -ge 2 ]] || die "--install-dir needs an argument"; INSTALL_DIR="$2"; shift 2 ;;
    --app|--app=*|--preview) die "$1 is retired: this script builds one app, alife.app (the Lab, Storm Worlds and the preview are gone, docs/APP.md section 6)" ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done
INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"

BUILD_DIR="${OUT_DIR:-$GUI_DIR/.build}"
BUILD_DIR="${BUILD_DIR/#\~/$HOME}"
# Absolute, so that a later `cd` cannot change what it means.
mkdir -p "$BUILD_DIR"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd -P)"

A_NAME="alife"; A_EXEC="alife"; A_DIR="alife.app"; A_ID="$ALIFE_BUNDLE_ID"; A_ICON="alife"
A_CATEGORY="public.app-category.simulation-games"; A_MODE="alife"

is_running() {  # $1 = bundle ID, $2 = path of the installed executable
  if [[ -n "$(lsappinfo find bundleid="$1" 2>/dev/null || true)" ]]; then return 0; fi
  pgrep -f "$2" >/dev/null 2>&1
}

# Fail early, before building, if the install cannot proceed.
if [[ $INSTALL -eq 1 ]]; then
  # A trial id is for a bundle that is looked at and thrown away: installed, it would replace the real alife.app with another id.
  [[ "$ALIFE_BUNDLE_ID" == "$DEFAULT_BUNDLE_ID" ]] || die "--install needs the default bundle id ($DEFAULT_BUNDLE_ID); ALIFE_BUNDLE_ID=$ALIFE_BUNDLE_ID is for a trial bundle only."
  if [[ ! -d "$INSTALL_DIR" || ! -w "$INSTALL_DIR" ]]; then
    echo "error: $INSTALL_DIR is not writable (this script never uses sudo)." >&2
    echo "Try: $0 --install --install-dir ~/Applications" >&2
    exit 1
  fi
  # Installing replaces the copy in INSTALL_DIR with the bundle in BUILD_DIR; the same directory would delete the new one first.
  if [[ "$(cd "$INSTALL_DIR" && pwd -P)" == "$BUILD_DIR" ]]; then
    die "--out and --install-dir are the same directory ($BUILD_DIR): installing would delete the bundles just built. Use another --out."
  fi
  if is_running "$A_ID" "$INSTALL_DIR/$A_DIR/Contents/MacOS/$A_EXEC"; then
    die "$A_NAME ($A_ID) is running. Quit it first; not touching it in case a run or a world is active."
  fi
fi

# 1. Release build.
echo "==> swift build -c release"
(cd "$GUI_DIR" && swift build -c release)
BIN_DIR="$(cd "$GUI_DIR" && swift build -c release --show-bin-path)"
BINARY="$BIN_DIR/AlifeControl"
[[ -x "$BINARY" ]] || die "release binary not found at $BINARY"

WORK="$(mktemp -d "$BUILD_DIR/icon.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
BUILD_NUMBER="$(git -C "$GUI_DIR" rev-list --count HEAD 2>/dev/null || echo 1)"

# Signing identity order: --sign, Developer ID Application, Apple Development, ad-hoc.
IDENTITIES="$(security find-identity -v -p codesigning 2>/dev/null || true)"
pick_identity() {  # $1 = name prefix; prints "<hash>\t<name>" of the first match
  printf '%s\n' "$IDENTITIES" | sed -n "s/^ *[0-9]*) \([0-9A-F]\{40\}\) \"\($1[^\"]*\)\".*/\1\t\2/p" | head -n 1
}
SIGN_ID="-"; SIGN_NAME="ad-hoc"
if [[ -n "$SIGN_ARG" ]]; then
  SIGN_ID="$SIGN_ARG"; SIGN_NAME="$SIGN_ARG"
  [[ "$SIGN_ARG" != "-" ]] || SIGN_NAME="ad-hoc"
else
  PICK="$(pick_identity "Developer ID Application")"
  [[ -n "$PICK" ]] || PICK="$(pick_identity "Apple Development")"
  if [[ -n "$PICK" ]]; then
    SIGN_ID="${PICK%%$'\t'*}"; SIGN_NAME="${PICK#*$'\t'}"
  fi
fi

build_app() {
  local bundle="$BUILD_DIR/$A_DIR"

  # 2. Icon.
  echo "==> generating the $A_ICON icon"
  swift "$SCRIPT_DIR/make-icon.swift" "$WORK/$A_ICON.iconset" "$A_ICON"
  iconutil -c icns "$WORK/$A_ICON.iconset" -o "$WORK/$A_ICON.icns"

  # 3. Bundle. The executable is named for the app (`pkill -f alife` matches it too).
  echo "==> assembling $bundle"
  rm -rf "$bundle"
  mkdir -p "$bundle/Contents/MacOS" "$bundle/Contents/Resources"
  cp "$BINARY" "$bundle/Contents/MacOS/$A_EXEC"
  cp "$WORK/$A_ICON.icns" "$bundle/Contents/Resources/AppIcon.icns"

  # AlifeLaunchMode marks a bundle made by this script (AppPlan.swift): only such a process reads the old apps' preferences (RepoPathStore.swift).
  cat > "$bundle/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key><string>$A_ID</string>
  <key>CFBundleName</key><string>$A_NAME</string>
  <key>CFBundleDisplayName</key><string>$A_NAME</string>
  <key>CFBundleExecutable</key><string>$A_EXEC</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$BUILD_NUMBER</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSApplicationCategoryType</key><string>$A_CATEGORY</string>
  <key>NSPrincipalClass</key><string>NSApplication</string>
  <key>AlifeLaunchMode</key><string>$A_MODE</string>
</dict>
</plist>
PLIST
  plutil -lint "$bundle/Contents/Info.plist"

  # 4. Sign.
  echo "==> signing $A_NAME with: $SIGN_NAME"
  if [[ "$SIGN_ID" == "-" ]]; then
    codesign --force --sign - "$bundle"
  else
    # Hardened runtime, so notarization can be added later.
    codesign --force --options runtime --timestamp=none --sign "$SIGN_ID" "$bundle"
  fi
  codesign --verify --strict --verbose=2 "$bundle"
}

install_app() {
  local bundle="$BUILD_DIR/$A_DIR" dest="$INSTALL_DIR/$A_DIR" tmp="$INSTALL_DIR/.$A_DIR.new"
  echo "==> installing to $dest"
  # Re-check: it may have been launched during the build.
  if is_running "$A_ID" "$dest/Contents/MacOS/$A_EXEC"; then
    die "$dest is running. Quit it first."
  fi
  # Copied to a temporary name first and moved into place, so that a copy that fails leaves the installed app as it was.
  rm -rf "$tmp"
  ditto "$bundle" "$tmp" || { rm -rf "$tmp"; die "copying to $tmp failed; $dest was not touched."; }
  local old="$INSTALL_DIR/.$A_DIR.old"
  rm -rf "$old"
  if [[ -e "$dest" ]]; then mv "$dest" "$old" || die "could not move $dest aside; nothing was changed (the new copy is at $tmp)."; fi
  if ! mv "$tmp" "$dest"; then
    [[ -e "$old" ]] && mv "$old" "$dest"
    die "moving $tmp to $dest failed; the old app was put back."
  fi
  rm -rf "$old"
  echo "installed: $dest"
}

# The retired apps are never touched: if any is in the install directory their paths are printed with how to move them to the Trash.
trash_note() {
  local old=() name path
  for name in "AlifeControl.app" "Storm Worlds.app" "alife Preview.app"; do
    [[ -e "$INSTALL_DIR/$name" ]] && old+=("$INSTALL_DIR/$name")
  done
  [[ ${#old[@]} -gt 0 ]] || return 0
  echo
  echo "The retired apps are still installed, and were not touched (alife replaces them; its settings were read from theirs, never changed):"
  for path in "${old[@]}"; do echo "  $path"; done
  echo "To remove one, quit it, then drag it to the Trash in Finder, or run:"
  for path in "${old[@]}"; do echo "  osascript -e 'tell application \"Finder\" to delete POSIX file \"$path\"'"; done
}

build_app

# Notarization (not done): with a Developer ID identity and a timestamp (--timestamp instead of
# --timestamp=none), zip the bundle with `ditto -c -k --keepParent`, then
#   xcrun notarytool submit "alife.zip" --keychain-profile <profile> --wait
#   xcrun stapler staple "alife.app"

# 5. Install: only alife.app. The retired apps are never touched; their paths are printed with how to Trash them.
if [[ $INSTALL -eq 1 ]]; then
  if is_running "$A_ID" "$INSTALL_DIR/$A_DIR/Contents/MacOS/$A_EXEC"; then
    die "$A_NAME ($A_ID) is running. Quit it first; nothing was installed."
  fi
  install_app
else
  echo "built: $BUILD_DIR/$A_DIR (use --install to copy it to $INSTALL_DIR)"
fi
trash_note
