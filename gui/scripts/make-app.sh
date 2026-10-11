#!/usr/bin/env bash
# Build, bundle and sign AlifeControl.app, and optionally install it.
# Run from any directory. See `make-app.sh --help`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_NAME="AlifeControl"
BUNDLE_ID="${BUNDLE_ID:-io.github.magicbundle.alifecontrol}"
VERSION="${VERSION:-0.1.0}"

SIGN_ARG=""
INSTALL=0
INSTALL_DIR="/Applications"

usage() {
  cat <<USAGE
Usage: make-app.sh [--sign IDENTITY] [--install] [--install-dir DIR] [--help]

Builds the release binary, generates the icon, assembles and signs
  $GUI_DIR/.build/$APP_NAME.app

Options:
  --sign IDENTITY     Signing identity (name or hash). Default: the first
                      "Developer ID Application" identity, else the first
                      "Apple Development" identity, else ad-hoc ("-").
  --install           Copy the bundle to the install directory (default /Applications),
                      replacing an existing $APP_NAME.app there. Refuses if it is running.
  --install-dir DIR   Install somewhere else, for example ~/Applications.
  -h, --help          Show this help.

Environment:
  BUNDLE_ID           Bundle identifier (default io.github.magicbundle.alifecontrol)
  VERSION             CFBundleShortVersionString (default 0.1.0)

Never uses sudo. Not notarized: sharing the app needs a Developer ID identity plus notarization.
USAGE
}

die() { echo "error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sign) [[ $# -ge 2 ]] || die "--sign needs an argument"; SIGN_ARG="$2"; shift 2 ;;
    --install) INSTALL=1; shift ;;
    --install-dir) [[ $# -ge 2 ]] || die "--install-dir needs an argument"; INSTALL_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done
INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"

BUILD_DIR="$GUI_DIR/.build"
BUNDLE="$BUILD_DIR/$APP_NAME.app"
DEST="$INSTALL_DIR/$APP_NAME.app"

# Fail early, before building, if the install cannot proceed.
if [[ $INSTALL -eq 1 ]]; then
  if is_running_msg=$(osascript -e "application id \"$BUNDLE_ID\" is running" 2>/dev/null) && [[ "$is_running_msg" == "true" ]]; then
    die "$APP_NAME ($BUNDLE_ID) is running. Quit it first; not touching it in case a run is active."
  fi
  if pgrep -f "$DEST/Contents/MacOS/$APP_NAME" >/dev/null 2>&1; then
    die "$DEST is running. Quit it first; not touching it in case a run is active."
  fi
  if [[ ! -d "$INSTALL_DIR" || ! -w "$INSTALL_DIR" ]]; then
    echo "error: $INSTALL_DIR is not writable (this script never uses sudo)." >&2
    echo "Try: $0 --install --install-dir ~/Applications" >&2
    exit 1
  fi
fi

# 1. Release build.
echo "==> swift build -c release"
(cd "$GUI_DIR" && swift build -c release)
BIN_DIR="$(cd "$GUI_DIR" && swift build -c release --show-bin-path)"
BINARY="$BIN_DIR/$APP_NAME"
[[ -x "$BINARY" ]] || die "release binary not found at $BINARY"

# 2. Icon.
echo "==> generating icon"
mkdir -p "$BUILD_DIR"
WORK="$(mktemp -d "$BUILD_DIR/icon.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
swift "$SCRIPT_DIR/make-icon.swift" "$WORK/AppIcon.iconset"
iconutil -c icns "$WORK/AppIcon.iconset" -o "$WORK/AppIcon.icns"

# 3. Bundle.
echo "==> assembling $BUNDLE"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources"
cp "$BINARY" "$BUNDLE/Contents/MacOS/$APP_NAME"
cp "$WORK/AppIcon.icns" "$BUNDLE/Contents/Resources/AppIcon.icns"
BUILD_NUMBER="$(git -C "$GUI_DIR" rev-list --count HEAD 2>/dev/null || echo 1)"

cat > "$BUNDLE/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleName</key><string>$APP_NAME</string>
  <key>CFBundleDisplayName</key><string>$APP_NAME</string>
  <key>CFBundleExecutable</key><string>$APP_NAME</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$BUILD_NUMBER</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
  <key>NSPrincipalClass</key><string>NSApplication</string>
</dict>
</plist>
PLIST
plutil -lint "$BUNDLE/Contents/Info.plist"

# 4. Sign. Identity order: --sign, Developer ID Application, Apple Development, ad-hoc.
IDENTITIES="$(security find-identity -v -p codesigning 2>/dev/null || true)"
pick_identity() {  # $1 = name prefix; prints "<hash>\t<name>" of the first match
  printf '%s\n' "$IDENTITIES" | sed -n "s/^ *[0-9]*) \([0-9A-F]\{40\}\) \"\($1[^\"]*\)\".*/\1\t\2/p" | head -n 1
}
SIGN_ID="-"; SIGN_NAME="ad-hoc"
if [[ -n "$SIGN_ARG" ]]; then
  SIGN_ID="$SIGN_ARG"; SIGN_NAME="$SIGN_ARG"
else
  PICK="$(pick_identity "Developer ID Application")"
  [[ -n "$PICK" ]] || PICK="$(pick_identity "Apple Development")"
  if [[ -n "$PICK" ]]; then
    SIGN_ID="${PICK%%$'\t'*}"; SIGN_NAME="${PICK#*$'\t'}"
  fi
fi
echo "==> signing with: $SIGN_NAME"
if [[ "$SIGN_ID" == "-" ]]; then
  codesign --force --sign - "$BUNDLE"
else
  # Hardened runtime, so notarization can be added later.
  codesign --force --options runtime --timestamp=none --sign "$SIGN_ID" "$BUNDLE"
fi
codesign --verify --strict --verbose=2 "$BUNDLE"

# Notarization (not done): with a Developer ID identity and a timestamp (--timestamp instead of
# --timestamp=none), zip the bundle with `ditto -c -k --keepParent`, then
#   xcrun notarytool submit AlifeControl.zip --keychain-profile <profile> --wait
#   xcrun stapler staple "$BUNDLE"

# 5. Install.
if [[ $INSTALL -eq 1 ]]; then
  echo "==> installing to $DEST"
  # Re-check: it may have been launched during the build.
  if pgrep -f "$DEST/Contents/MacOS/$APP_NAME" >/dev/null 2>&1; then
    die "$DEST is running. Quit it first."
  fi
  if [[ -e "$DEST" ]]; then rm -rf "$DEST"; fi
  ditto "$BUNDLE" "$DEST"
  echo "installed: $DEST"
else
  echo "built: $BUNDLE (use --install to copy it to $INSTALL_DIR)"
fi
