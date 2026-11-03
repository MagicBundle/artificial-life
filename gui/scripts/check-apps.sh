#!/usr/bin/env bash
# Smoke test of the app bundle: builds alife.app with make-app.sh, signed ad-hoc only (no keychain, no prompt), and checks the Info.plist
# keys, the executable, the icon and `codesign --verify`; also that the retired options of the script are refused.
#
#   gui/scripts/check-apps.sh [--out DIR]
#
# macOS with a Swift toolchain only, and a release build takes minutes, so this is a manual step (also run by
# tests/test_app_bundles.py when ALIFE_APP_SMOKE=1). It installs nothing and launches nothing. DIR (default: a new
# directory under $TMPDIR) must not exist or must be empty: it is the one thing this script writes into.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) [[ $# -ge 2 ]] || { echo "error: --out needs an argument" >&2; exit 2; }; OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,9p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$OUT" ]]; then OUT="$(mktemp -d "${TMPDIR:-/tmp}/alife-apps.XXXXXX")"; fi
mkdir -p "$OUT"
if [[ -n "$(ls -A "$OUT")" ]]; then echo "error: $OUT is not empty" >&2; exit 2; fi

FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }
pass() { echo "ok:   $*"; }

VERSION_EXPECTED="9.8.7"
echo "==> make-app.sh --sign - --out $OUT"
VERSION="$VERSION_EXPECTED" "$SCRIPT_DIR/make-app.sh" --sign - --out "$OUT"

plist() {  # $1 = bundle, $2 = key; prints the value, or nothing when the key is missing
  /usr/libexec/PlistBuddy -c "Print :$2" "$1/Contents/Info.plist" 2>/dev/null || true
}

expect() {  # $1 = bundle, $2 = key, $3 = expected value
  local got
  got="$(plist "$1" "$2")"
  if [[ "$got" == "$3" ]]; then pass "$(basename "$1") $2 = $3"; else fail "$(basename "$1") $2: expected '$3', got '$got'"; fi
}

check_bundle() {  # $1 = bundle, $2 = id, $3 = name, $4 = executable, $5 = mode, $6 = category
  local b="$1"
  [[ -d "$b" ]] || { fail "$b was not built"; return; }
  plutil -lint "$b/Contents/Info.plist" >/dev/null && pass "$(basename "$b") Info.plist is valid" || fail "$b Info.plist"
  expect "$b" CFBundleIdentifier "$2"
  expect "$b" CFBundleName "$3"
  expect "$b" CFBundleDisplayName "$3"
  expect "$b" CFBundleExecutable "$4"
  expect "$b" CFBundlePackageType APPL
  expect "$b" CFBundleIconFile AppIcon
  expect "$b" CFBundleShortVersionString "$VERSION_EXPECTED"
  expect "$b" LSMinimumSystemVersion 14.0
  expect "$b" LSApplicationCategoryType "$6"
  expect "$b" AlifeLaunchMode "$5"
  local build
  build="$(plist "$b" CFBundleVersion)"
  if [[ "$build" =~ ^[0-9]+$ ]]; then pass "$(basename "$b") CFBundleVersion = $build"; else fail "$(basename "$b") CFBundleVersion '$build' is not a number"; fi
  if [[ -x "$b/Contents/MacOS/$4" ]]; then pass "$(basename "$b") executable $4"; else fail "$b/Contents/MacOS/$4 missing or not executable"; fi
  if [[ "$(head -c 4 "$b/Contents/Resources/AppIcon.icns" 2>/dev/null)" == "icns" ]]; then pass "$(basename "$b") icon is an icns"; else fail "$b icon"; fi
  if codesign --verify --strict "$b" 2>/dev/null; then pass "$(basename "$b") codesign --verify"; else fail "$(basename "$b") codesign --verify"; fi
  local details
  details="$(codesign -dv "$b" 2>&1 || true)"  # a variable, not a pipe: grep -q closing early would fail the pipeline
  if [[ "$details" == *"Signature=adhoc"* ]]; then pass "$(basename "$b") is signed ad-hoc"; else fail "$(basename "$b") is not ad-hoc signed"; fi
}

ALIFE="$OUT/alife.app"
check_bundle "$ALIFE" io.github.magicbundle.alife alife alife alife public.app-category.simulation-games

# One app: nothing else was built, and the retired options are refused (before any build).
BUILT="$(ls "$OUT" | grep -c '\.app$' || true)"
[[ "$BUILT" == "1" ]] && pass "alife.app is the only bundle built" || fail "$BUILT bundles were built"
for retired in "--app lab" "--app both" "--preview"; do
  # shellcheck disable=SC2086
  if "$SCRIPT_DIR/make-app.sh" $retired --sign - --out "$OUT/refused" >/dev/null 2>&1; then fail "make-app.sh $retired was accepted"; else pass "make-app.sh $retired is refused"; fi
done
[[ ! -e "$OUT/refused" ]] && pass "a refused option builds nothing" || fail "a refused option left $OUT/refused"

if [[ $FAILED -ne 0 ]]; then echo "FAILED (bundles are in $OUT)" >&2; exit 1; fi
echo "all checks passed (bundles are in $OUT)"
