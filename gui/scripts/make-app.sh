#!/usr/bin/env bash
# Build, bundle and sign the AlifeControl apps (the research Lab and Storm Worlds), and optionally install them.
# Run from any directory. See `make-app.sh --help`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LAB_BUNDLE_ID="${BUNDLE_ID:-io.github.magicbundle.alifecontrol}"
STORM_BUNDLE_ID="${STORM_BUNDLE_ID:-io.github.magicbundle.stormworlds}"
VERSION="${VERSION:-0.1.0}"

SIGN_ARG=""
INSTALL=0
INSTALL_DIR="/Applications"
WHICH="lab"
OUT_DIR=""

usage() {
  cat <<USAGE
Usage: make-app.sh [--app lab|storm|both] [--sign IDENTITY] [--out DIR] [--install] [--install-dir DIR] [--help]

Builds the release binary once, generates the icons, assembles and signs
  AlifeControl.app   the research Lab (the run controls, the log, the director panel)
  Storm Worlds.app   the game: opens straight into the World window; the Lab is Window > Lab
in DIR (default $GUI_DIR/.build).

Options:
  --app WHICH         lab (default: what this script always built), storm, or both (recommended
                      with --install: the game and the Lab share the repository folder setting).
  --sign IDENTITY     Signing identity (name or hash). Default: the first
                      "Developer ID Application" identity, else the first
                      "Apple Development" identity, else ad-hoc ("-"). "--sign -" is ad-hoc and
                      never touches the keychain.
  --out DIR           Where the bundles are written (default $GUI_DIR/.build).
  --install           Copy the bundle(s) to the install directory (default /Applications), replacing a copy
                      there. Refuses, before building, if one of them is running.
  --install-dir DIR   Install somewhere else, for example ~/Applications.
  -h, --help          Show this help.

Environment:
  BUNDLE_ID           Bundle identifier of the Lab (default io.github.magicbundle.alifecontrol)
  STORM_BUNDLE_ID     Bundle identifier of Storm Worlds (default io.github.magicbundle.stormworlds)
  VERSION             CFBundleShortVersionString of both (default 0.1.0)

Never uses sudo. Not notarized: sharing an app needs a Developer ID identity plus notarization.
USAGE
}

die() { echo "error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app) [[ $# -ge 2 ]] || die "--app needs an argument"; WHICH="$2"; shift 2 ;;
    --sign) [[ $# -ge 2 ]] || die "--sign needs an argument"; SIGN_ARG="$2"; shift 2 ;;
    --out) [[ $# -ge 2 ]] || die "--out needs an argument"; OUT_DIR="$2"; shift 2 ;;
    --install) INSTALL=1; shift ;;
    --install-dir) [[ $# -ge 2 ]] || die "--install-dir needs an argument"; INSTALL_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done
INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"
case "$WHICH" in
  lab) APPS="lab" ;;
  storm) APPS="storm" ;;
  both) APPS="lab storm" ;;
  *) usage >&2; die "--app must be lab, storm or both (got: $WHICH)" ;;
esac

BUILD_DIR="${OUT_DIR:-$GUI_DIR/.build}"
BUILD_DIR="${BUILD_DIR/#\~/$HOME}"
# Absolute, so that a later `cd` cannot change what it means.
mkdir -p "$BUILD_DIR"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"

# What differs between the apps. $1 = lab|storm; sets A_* variables.
app_vars() {
  case "$1" in
    lab)
      A_MODE="lab"; A_NAME="AlifeControl"; A_EXEC="AlifeControl"; A_DIR="AlifeControl.app"
      A_ID="$LAB_BUNDLE_ID"; A_ICON="lab"; A_CATEGORY="public.app-category.developer-tools" ;;
    storm)
      A_MODE="world"; A_NAME="Storm Worlds"; A_EXEC="StormWorlds"; A_DIR="Storm Worlds.app"
      A_ID="$STORM_BUNDLE_ID"; A_ICON="storm"; A_CATEGORY="public.app-category.simulation-games" ;;
  esac
}

is_running() {  # $1 = bundle ID, $2 = path of the installed executable
  if [[ -n "$(lsappinfo find bundleid="$1" 2>/dev/null || true)" ]]; then return 0; fi
  pgrep -f "$2" >/dev/null 2>&1
}

# Fail early, before building, if the install cannot proceed.
if [[ $INSTALL -eq 1 ]]; then
  if [[ ! -d "$INSTALL_DIR" || ! -w "$INSTALL_DIR" ]]; then
    echo "error: $INSTALL_DIR is not writable (this script never uses sudo)." >&2
    echo "Try: $0 --app $WHICH --install --install-dir ~/Applications" >&2
    exit 1
  fi
  for app in $APPS; do
    app_vars "$app"
    if is_running "$A_ID" "$INSTALL_DIR/$A_DIR/Contents/MacOS/$A_EXEC"; then
      die "$A_NAME ($A_ID) is running. Quit it first; not touching it in case a run or a world is active."
    fi
  done
fi

# 1. Release build (once: both apps are this binary).
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

build_app() {  # $1 = lab|storm
  app_vars "$1"
  local bundle="$BUILD_DIR/$A_DIR"

  # 2. Icon.
  echo "==> generating the $A_ICON icon"
  swift "$SCRIPT_DIR/make-icon.swift" "$WORK/$A_ICON.iconset" "$A_ICON"
  iconutil -c icns "$WORK/$A_ICON.iconset" -o "$WORK/$A_ICON.icns"

  # 3. Bundle. The executable is named for the app so that the menu bar, Activity Monitor and `pgrep` tell the two apart.
  echo "==> assembling $bundle"
  rm -rf "$bundle"
  mkdir -p "$bundle/Contents/MacOS" "$bundle/Contents/Resources"
  cp "$BINARY" "$bundle/Contents/MacOS/$A_EXEC"
  cp "$WORK/$A_ICON.icns" "$bundle/Contents/Resources/AppIcon.icns"

  # AlifeLaunchMode tells the one executable which app it is (LaunchMode.swift); the same key turns on the shared
  # repository-folder setting (RepoPathStore.swift).
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

install_app() {  # $1 = lab|storm
  app_vars "$1"
  local bundle="$BUILD_DIR/$A_DIR" dest="$INSTALL_DIR/$A_DIR"
  echo "==> installing to $dest"
  # Re-check: it may have been launched during the build.
  if is_running "$A_ID" "$dest/Contents/MacOS/$A_EXEC"; then
    die "$dest is running. Quit it first."
  fi
  if [[ -e "$dest" ]]; then rm -rf "$dest"; fi
  ditto "$bundle" "$dest"
  echo "installed: $dest"
}

for app in $APPS; do build_app "$app"; done

# Notarization (not done): with a Developer ID identity and a timestamp (--timestamp instead of
# --timestamp=none), zip each bundle with `ditto -c -k --keepParent`, then
#   xcrun notarytool submit "Storm Worlds.zip" --keychain-profile <profile> --wait
#   xcrun stapler staple "Storm Worlds.app"

# 5. Install.
for app in $APPS; do
  app_vars "$app"
  if [[ $INSTALL -eq 1 ]]; then
    install_app "$app"
  else
    echo "built: $BUILD_DIR/$A_DIR (use --install to copy it to $INSTALL_DIR)"
  fi
done
