# alife: one app. Design specification, Revision 1

> **Status: approved by the owner, 2026-11-01.** Decisions of section 8: D0 (c) build the default path to newcomer standard, keep Welcome minimal, decide the rest after the preview; D1 B, grow then planet, with A tried by the owner; D2 A, an existing checkout with instructions; D3 C, ask on close, with Dock progress and a bounce, falling back to B if spike S3 fails; D4 A, pause on leaving Watch within the app, with the setting; D5 B, a parallel track after a play-through of the preview.

Status: PR-A, PR-B, PR-C, PR-D and PR-E built (§5.4 as built, §7.1, §7.2, §7.3, §7.4); the rest design only. Base: worktree `preset-wt` (= main, `4d98ba3`). Owner request, 2026-11-01: "merge both apps into one.
Simply call the new app 'alife'. Emphasize user-friendly interfaces and intuitive workflows."
Revision 1 answers the independent review (`app_review/review.md`) and the coordinator's resolutions; Revision 0 is `spec_rev0.md`.
Section 12 lists every finding and what was done, including the ones I checked and where I disagree (none were wrong; three needed correction of the review's own
wording, §12).

Evidence. Paths are relative to `gui/Sources/AlifeControl/` (AC), `gui/Sources/AlifeCore/` (CORE), or the repo root. **[V]** read in code, **[I]** inferred (reason given),
**[U]** not verifiable here (needs a spike, the engine, or the owner's hands). Statements with a `file:line` and no tag are [V]. The Lab form was rendered headlessly
(`app_spec/shots/lab-light.png`, `lab-dark.png`); nothing else was drawn or run. The owner is "the owner" or "they" throughout.

---

## 0. Summary

- The code merge is small (one executable already). The work is the experience: today someone who wants to *play* and someone who wants to *experiment* use two
  apps that disagree on what a "world" is, which windows exist, when to quit, and where settings live.
- **Recommendation: one window, worlds first.** A sidebar lists your worlds. A world opens to **Watch** (the live picture) or **Details** (facts, settings it was made
  with, branches). **Create** is one set-up flow with recommended starting points and an *Advanced* disclosure holding every Lab lever. Names: "Alife Control" and
  "Storm Worlds" are retired; a world's *kind* is **Played** or **Research** and its *shape* is Flat, 3D box or Planet (§2.2).
- **Two honest facts the first draft blurred:** there are two kinds of world with two sets of Watch controls (§3.4), and a planet takes two processes (grow, then
  make the planet), which the app presents as one named pipeline with parking points (§3.2).
- **Safety in one sentence:** one played world at a time, a world is saved before anything stops it, an app-level guard allows one writer per directory, leaving the Watch
  page pauses (owner decision D4), and quitting goes through a pre-confirmation, then `TerminationGate` (one method added), with a panel that does not depend on a window.
- **Build order:** safety fixes that help today's apps first, then a shell you can install beside the old apps (a spike first), the library, the set-up page, safety
  rules, Watch and Details, Welcome, retirement of the old modes. Eight PRs, §7. No engine change; no golden test touched before the last PR.
- **Six owner decisions** (D0 to D5), §8. Out of this program: Trash, in-app comparison, viewer legibility work (unless D5 says otherwise), the cuts in §9.2.

---

## 1. What a user does today

### 1.1 Inventory

| Thing | Today | Evidence |
|---|---|---|
| Executable | one binary; `@main enum AlifeEntry` picks `StormWorldsApp` or `AlifeControlApp` from `LaunchMode` | AC/AlifeControlApp.swift:41-45 |
| Bundles | `AlifeControl.app` (mode `lab`) and `Storm Worlds.app` (mode `world`, executable `StormWorlds`) | gui/scripts/make-app.sh:8-9,77-86,175 |
| Lab app windows | `WindowGroup("Alife Control")`, `Window("World")`, `Window("3D Viewer")`; no `Settings`, no `.commands` | AC/AlifeControlApp.swift:54-80; `.commands` only at AC/StormWorldsApp.swift:23 |
| Storm app windows | `Window("Storm Worlds")` first, `Window("Alife Control")` as a tool window, 3D Viewer, `Settings` (folder only), Window-menu World/Lab Command-1/2 | AC/StormWorldsApp.swift:15-36,43-55 |
| At launch | Lab: the form. Storm: the World window, launcher sheet over a blank map, or `RepoGateView`. Never the last world | AC/WorldView.swift:59-62; docs/DESIGN.md:10924 |
| Quit after last window | `true` in both | CORE/LaunchMode.swift:101; AC/AlifeControlApp.swift:20 |
| Closing windows | Lab main: closes viewer and World (Lab app) or only the viewer (Storm app). World window: `play.stop()` (saves). 3D Viewer: `stream.stop()` | CORE/LaunchMode.swift:93-96; AC/ContentView.swift:62-64; AC/WorldView.swift:55; AC/Viewer3DView.swift:69-70 |
| Processes (at most) | one `RunController.job` (`run`/`inspect`/`uv sync`), one pygame viewer, one `alife stream`, one `alife play` plus retired ones still saving | AC/RunController.swift:111,135-136; AC/StreamController.swift:64; AC/PlayController.swift:111-119 |
| Run and world at once | allowed: `canStart` ignores the play session | AC/RunController.swift:207 |
| Locks | `play.lock` is taken only by `alife play`; `alife resume` refuses a played directory; **`alife play` does not check for a running `run`/`resume`** | src/alife/cli.py:652-657,1213-1227,1342-1347 |
| Quit | the gate waits (40 s) only for `alife play`; a run gets one SIGINT and the app does not wait; `shutdown()` calls `stop()`, which is guarded by `!stopRequested` and `job == .run`, so it never kills a run and does not touch `uv sync`/`inspect` | AC/AlifeControlApp.swift:24-33; AC/AppTermination.swift:14; AC/RunController.swift:446,570-575 |
| Persisted | `repoPath` (shared domain `io.github.magicbundle.alife`); per bundle: `world.dockSide/dockExtent/dockExtentBottom/dockCollapsed` (`@AppStorage`) and `world.autopause` (a **JSON string**), `exposure`, `scrollZooms`, `minimap`, `globe`, `spin`, `keepNorthUp`, `heightColour`, `skipConfirmCostOne`, `hintShown`. Nothing of the Lab form | CORE/RepoPathStore.swift:28-30; AC/WorldView.swift:11-14; AC/WorldModel.swift:196-206,288-299; AC/WorldModel+Decide.swift:129 |
| Sandbox / bookmarks / keychain | none (no entitlements, no bookmark API, no `SecItem`) | make-app.sh:184-188; grep of gui/Sources |
| Restoration, undo, charts | Lab main window restorable, others not; no `UndoManager`; no charts | AC/ContentView.swift:62; AC/WorldView.swift:55; AC/Viewer3DView.swift:69 |
| CI | no GUI job; Swift tests run by hand | .github/workflows/ci.yml |

### 1.2 The real paths

**P1 First launch, Lab.** A two-column form: a fixed 400 pt scroll column of boxes, then Command, Start, Status, Director, tool buttons, Log (AC/ContentView.swift:25-47,40). The
Repository box is first, pre-filled with the owner's own `~/artificial-life` (AC/RunController.swift:47). Missing engine: "Set up environment" and a sentence about `uv sync`
(ContentView.swift:76-109). Start sits in the other column and is disabled by `canStart`, which includes `repoReady` (RunController.swift:207) while `validationErrors` does not
mention the repository (:189-197): the Command box shows nothing. A newcomer sees a grey Start with no reason beside it (and the render, `shots/lab-light.png`, shows a mostly
empty right column: Status "No metrics yet", Director "off", an empty Log).

**P2 First launch, Storm Worlds.** `RepoGateView` while the folder is unusable (StormWorldsApp.swift:61-91,122-150); its only way to set up the environment is a button that opens
the Lab (:137). With a folder, the launcher sheet opens over a blank map (WorldView.swift:59-62).

**P3 Start a world in the game.** The planet form's Source is a *research run with a checkpoint* (PlayWorlds.swift:172-186); on a fresh repository: "No world yet: make a planet
below." (WorldLauncherView.swift:77), empty Source, "Choose a source run." (PlayLaunch.swift:188). The way out is another window: start a research run in the Lab and wait
(`mini3d` to 500,000 steps took about 1 h 10 on the M5 Pro: HANDOVER.md:726). The engine has `alife play --preset` (cli.py:1965; DESIGN.md:10671); "a launcher entry that makes a
world from nothing" is listed as not done (HANDOVER.md:735).

**P4 Start a run in the Lab.** Mode New run / Resume / Fork changes the form (ContentView.swift:116-123,176-201); levers named by engine keys; validation text quotes keys
(CORE/Levers.swift:56-90) and shows in the Command box only (ContentView.swift:393-397). The Steps field blank means the preset default, **500 steps for `tiny3s`**
(ContentView.swift:195-201; src/alife/presets/tiny3s.toml:52).

**P5 Reopen.** Game: Worlds... > Resume = `alife play runs/NAME` (WorldLauncherView.swift:101; PlayLaunch.swift:213-220), always at 1x (WorldLauncherModel.swift:65-67). Lab: Mode
Resume = `alife resume` (headless). The clock strip's "Resume" and the chapter card's "Continue" both un-pause (AC/ControlCenterView.swift:180-181; AC/WorldReportViews.swift:162,183).

**P6 Watch and intervene.** World window: map, control center docked left/right/bottom (WorldView.swift:34-52), tabs Cast, Chronicle, Report, Powers (ControlCenterView.swift:78-97), clock
strip, storage strip, alerts, inspector, chapter card. Orientation aids exist (CORE/WorldReadout.swift:1-12). The View menu is inside the dock header and disabled until live
(ViewMenu.swift:77,163); keys are a table (CORE/WorldKeys.swift:46-56).

**P7 Reports.** Only inside a live world; a closed world's chronicle returns by reopening it (PlayController.swift:84-86).

**P8 Experiment, fork, compare, inspect.** Fork pre-fills from the source's *latest* checkpoint (CORE/Levers.swift:11); Resume refuses levers (ContentView.swift:217-220); Inspect prints to the
Log (:533-535). Comparing = run A, wait (one job: RunController.swift:111), run B, compare outside. An expert today can watch run A in the 3D Viewer window while composing run B and
tailing the Log, because they are separate windows (AlifeControlApp.swift:60; Viewer3DView.swift:48-70).

**P9 Storage.** Only in the World dock (ControlCenterView.swift:39,64; StorageViews.swift:127-133).

### 1.3 Friction list

| ID | Friction | Evidence |
|---|---|---|
| F1 | A game that cannot make a world from nothing: empty Source, "Choose a source run" | WorldLauncherView.swift:77; PlayLaunch.swift:188; PlayWorlds.swift:172-186 |
| F2 | Start disabled with no reason beside it (folder/engine missing) | RunController.swift:189-207; ContentView.swift:76-109,418-420 |
| F3 | Default folder is the owner's path; nothing else is tried | RunController.swift:47 |
| F4 | Engine setup from the game goes through the Lab; a missing `uv` is a Log line | StormWorldsApp.swift:137; RunController.swift:436-439 |
| F5 | One word, several meanings: **Resume** x3 (launcher = play live, Lab mode = headless, clock strip = un-pause), **Continue** (chapter card = un-pause), **Home** (camera button), and run/world/planet/fork/branch for overlapping things | WorldLauncherView.swift:101; ContentView.swift:116; ControlCenterView.swift:180; WorldReportViews.swift:162; WorldView.swift:70; WorldLauncherModel.swift:76 |
| F6 | Jargon: "Keep inventing levers", "--overlay fast", "Compile the director into the world (director.enabled)", Houses exact/clade/auto, "Ledger error", "Clades", keys as placeholders | ContentView.swift:215,172,273,454; WorldLauncherView.swift:221-225 |
| F7 | A world made without `director.enabled` "takes no powers", behind a checkbox | WorldLauncherView.swift:221; WorldPowersView.swift:20; PlayLaunch.swift:86-87 |
| F8 | Resume buttons and "Make planet and play" grey with no text while a world is live | WorldLauncherView.swift:102,137; WorldLauncherModel.swift:58-63 |
| F9 | Validation quotes engine keys, far from the field | CORE/Levers.swift:56-90,424-445; ContentView.swift:393-397 |
| F10 | Hidden features: View menu in the dock header, disabled until live; keys known from a hint shown once; Pauses/"Skip the second click"/"Raw kin events" in a *View* menu; no macOS menu | ViewMenu.swift:77,143-163; WorldModel+HUD.swift:219-234 |
| F11 | Lab app has no menu bar, no Settings scene, one key equivalent (Cmd-.) | AlifeControlApp.swift:52-81; ContentView.swift:409 |
| F12 | Two apps, two behaviours; `--world` handled in two places | LaunchMode.swift:76-109; ContentView.swift:66-70; StormWorldsApp.swift:84-91 |
| F13 | Settings per bundle (dock, display, hint, window frames) | HANDOVER.md:757; DESIGN.md:10928 |
| F14 | The Lab form is forgotten at every launch | RunController.swift:110 |
| F15 | Quitting stops a long run (one SIGINT, no confirmation, no wait); the gate shows nothing while waiting [I: `.terminateLater` has no UI] | AlifeControlApp.swift:24-33; AppTermination.swift:14-36 |
| F16 | Closing the last window quits, ending an overnight run | CORE/LaunchMode.swift:101; ContentView.swift:62-64 |
| F17 | Three ways to look at a run in three places; the 3D Viewer window opens empty from the Window menu | ContentView.swift:512-531; StormWorldsApp.swift:20-22 |
| F18 | The log is global but drawn only in the Lab page | RunController.swift:161-166 |
| F19 | Lab right column mostly empty before a run | `shots/lab-light.png` |
| F20 | No library: run names in a picker vs play worlds in a sheet; two scanners | ContentView.swift:136-141; RunController.swift:252-262; PlayWorlds.swift:146-196 |
| F21 | The repository form exists three times | ContentView.swift:76-109; StormWorldsApp.swift:93-119,153-170 |
| F22 | No comparison, metric history or queue (out of this program, §9.2) | ContentView.swift:443-471 |
| F23 | Start has no keyboard shortcut | ContentView.swift:418 |

Kept, not friction: aim-preview-confirm for powers (WorldPowersView.swift:201-230), the obstacle text that exists (WorldLauncherView.swift:98-100,208-211), the exact command preview, the
Esc ladder, the chapter pause, the storage strip's confirmation (StorageViews.swift:127-133), the preset card.

---

## 2. Target design

### 2.1 Principles
1. The world is the unit. 2. Remove a concept before explaining it. 3. Defaults first, every lever one click away (an *Advanced* disclosure, remembered). 4. Say why: a disabled
control has its reason beside it; a setting has an effect line. 5. Nothing starts or is lost without being asked (opening the app starts nothing: DESIGN.md:10924).
6. Observe, decide, watch (owner, 2026-10-19): rare, meaningful decisions; the map gets the screen. 7. macOS conventions. 8. **One word per action** (§2.5), checked by a test.

### 2.2 The model
- **alife folder** (today "repository"): engine (`.venv/bin/alife`) plus `runs/`.
- **World**: a directory under `runs/` with at least one save point. Two **kinds**, named by what the user did and can do:
  **Played** (has `play.json`: made by *Make planet* or by an old game; live play, powers, chronicle, reports) and **Research** (made by *Create*, *Branch*; grown headlessly;
  numbers, settings, an autopilot, an observer picture). Kind is a fact shown as a badge; it is never a mode switch.
- **Shape** is separate: Flat, 3D box, Planet (the preset guide's groups: CORE/PresetGuide.swift:19-41). A `tiny3s` run is a **Research** world of **Planet** shape; the recipe "Small
  planet" (§3.2) makes a Research world and then a Played one. "Planet" never names a kind.
- **Activity** (a process): playing, growing, watching (stream), setting up, checking.
- **States** (§5.2): Saved, Live, Paused, Growing, Saving, In use elsewhere, Unfinished.

### 2.3 Alternatives
| | A. Two modes in one window | B. One window per world | **C. Worlds library, one window (recommended)** | D. Game only, Lab hidden |
|---|---|---|---|---|
| Newcomer clarity | poor | good | **good** | best for a player |
| Experts keep everything | yes | yes | **yes (Advanced, Command, Run panel, §3.7)** | no |
| Removes | nothing | some | **launcher, Lab, 3D Viewer window, two scanners** | the Lab |
| Cost | S | XL (singletons; one `alife play` at a time: PlayController.swift:55,123-125) | **L** | S, loses research |
Why not A: renames two apps, keeps F5/F13/F20. Why not B: several windows invite several played worlds (one 686 MB save to write on stop: HANDOVER.md:701). Why not D: the Lab is
the source of every playable world (P3).

### 2.4 Structure

**One main window**, `Window("alife", id: "main")`. **One minimum for every page: 1,100 x 700** (today's World minimum: WorldView.swift:53; LauncherLayout.windowMinHeight 700), so
entering Watch never resizes the window abruptly (the owner dislikes abrupt changes). The Lab's 960 minimum is dropped.

```
sidebar (collapsible; Ctrl-Cmd-S, the system toggle)     detail
 Worlds (landing)          rows nested: a played world's seed run under it     Worlds | World: Watch / Details | Create / Branch / Keep growing /
 <world rows + state>                                                          Make planet | Activity | Welcome
 Unfinished (n)
 Activity [badge]
```
- **Worlds** (landing): *Play "<last played world>"* (Research last-used: *Open*), *New world*, cards (search, filters All / Played / Research / Unfinished). "Last saved" is the directory
  mtime (PlayWorlds.swift:84,154); nothing about "a report waiting" is shown, because it is not in `play.json`.
- **Watch** and **Details**: §3.4 and §3.7. Played world: Watch is the live world. Research world: Watch is the observer picture (§3.4) with a **Run panel** docked beside it.
- **Create / Branch / Keep growing / Make planet**: one set-up page (§3.2), also openable in its own window (`Window("Set up", id: "setup")`, one instance) so an expert can compose
  beside a Watch (§3.7). This second window is a form, not a world: R1 (one played world) is untouched.
- **Activity**: jobs (setup, run, check, watch) with Stop/Force stop, and the global log (Clear, Copy).
- **Welcome**: first run (§3.1).

**Keeping the Watch page and its Metal view alive in one window (spike S1 decides; do not assume).** Facts: `WorldModel` is a `@StateObject` created in `WorldView.init`
(WorldView.swift:10,21-28); `WorldMTKView` and `SphereMTKView` each install an `NSEvent` local key monitor that handles any key of their window unless an `NSText` is the first
responder (WorldMetalView.swift:53-83; SphereMetalView.swift:44); the toolbar, `navigationTitle` and `.frame(minWidth:1100,minHeight:700)` are modifiers of `WorldView` (WorldView.swift:53-81);
`PlayController` has **single** delivery closures (`onDelivery/onEvents/onAlerts/onReports/onPlans/onReset`) assigned in `WorldModel.init` (WorldModel.swift:303-308), so a model that
does not exist when a header or backlog arrives loses the toasts, alerts and reports (the reason `WorldWindowRoot` exists: StormWorldsApp.swift:71-76). Design:
1. **Hoist.** `WorldSession` (app level, `@MainActor`, created at launch) owns the `WorldModel`; the closures are assigned once, so any `PlayController.start` (a Play click, `--world`) is
   delivered whether or not a Watch page exists. `WorldWindowRoot`'s ordering trick disappears. Test: start a world through `openWorldAtLaunch` with no page; the backlog reaches the chronicle.
2. **`watchActive`** is a state of `WorldModel` (set by the page appearing/disappearing and the window being key). `handleKey` returns false when `!keysEnabled` (= `watchActive && windowIsKey`), so
   the key falls through to SwiftUI (Space/Return activate a focused button on every other page). Test: with `watchActive == false`, a key event sends no `PlayCommand` (stand-in
   process as in `PlayControllerTests`). The attention predicate (`windowIsKey`, WorldModel+Decide.swift:218) gains `&& watchActive`: today a one-window app would count "selected another page"
   as "the user is looking" and suppress the Dock badge and bounce.
3. **Chrome by page.** Toolbar items, title and dock belong to `WatchPage`, which exists only while selected; the canvas host (`WatchCanvasHost`, below the pages in a `ZStack`) carries no toolbar,
   title or size. A hidden canvas has `opacity 0`, `allowsHitTesting(false)`, `accessibilityHidden`, and its draw loop paused [U].
4. **Spike S1 (first step of PR-B), pass criteria:** (a) selecting other pages and back keeps the same `MTKView` instance and does not call `makeNSView` again; (b) Space, `.`, `g`, `k`, `1`
   on a button/list/toggle of another page send nothing and still activate the control; (c) no Watch toolbar item or title on other pages; (d) GPU frame time of the hidden canvas is ~0; (e) Cmd-Q,
   close and `--world` still reach the model. **Fallback F1:** hoisted model, but the Metal view is created on entering Watch and removed on leaving it (the model keeps camera, selection,
   store; the renderer rebuilds: measure time to first frame). **Fallback F2:** a full-window Watch with no sidebar, entered by Play/Open and left by an explicit "Worlds" button (the
   sidebar and Watch never share a window). F1 and F2 need the same hoist and the same `watchActive` gate, so only step 3 differs.
**Spike S1, done 2026-11-01 (PR-B): the primary design passes; neither fallback is needed.** *Spike.* A throwaway package outside the repository
(a lone `Window`, a `NavigationSplitView` with four pages, a `WatchCanvasHost` holding a real `MTKView` that installs a local key monitor and takes
first responder as `WorldMTKView` does, toolbar and title only on the Watch page, a key monitor gated by `watchActive`), driven by its own timer and by
key events posted to the application's queue. (a) Over 9 page changes, a window resize and two sidebar toggles: `makeNSView` once, `dismantleNSView`
never, one view identity. (b) Space, `.`, `g`, `k`, `1` on the Worlds page: the monitor saw 5, the world acted on 0, and the page's own focusable
view took 5; the **ungated control** (same code, gate off) had the world act on 5 and the page take 0, which is the leak the gate closes. On Watch the
world acted on all 5. (c) Toolbar items `overview` and `close` and the title "World mini3s_a" exist only while Watch is selected; the other pages have the
sidebar toggle and their own title. (d) Hidden for 3 s: 0 draws (the model's 90 deliveries asked for none); shown for 3 s: 90. *Real app, same session*
(the debug binary with the shell, a private repository and preferences suite, the real engine; actions injected in the process by a dylib kept outside
the repository, mouse and keys posted to the event queue): a planet made from the Worlds page (a click on *Make planet*) went live, the window moved to
Watch by itself, and the one `SphereMTKView` (same address, 0x783e24b200) was **not hidden** on Watch, **hidden, not first responder, not accepting
first responder** on Worlds (and Activity, Lab), and back on Watch it was first responder again; with `--world planet_a` the world started through the
same path and was delivered to a model that no page had made (criterion (e) for `--world`; the unit test `aWorldStartedByTheLaunchArgumentIsDeliveredWithNoPageEverShown`
does it without a view). On Watch, Space, `.`, Space reached the engine as `pause`, `step`, `run`; on Worlds, Activity and Lab, 13 key events (Space `.` `g` `k` `1`
and Space `.` `g`) sent **nothing**; back on Watch, Space paused it. Cmd-Q and a window close go through PR-A's gate as before (exercised in the quit run
below). *Not shown:* Space or Return **activating a focused button** on another page (Tab did not move focus to a button in the scripted runs, with or
without full keyboard access: the evidence is that the keys reach the focused list, text field editor and page and none reaches the world), the GPU frame
time of the hidden canvas in a real world (the draw count is 0 in the spike and the real view defers its draws and stops its display link while hidden:
`WorldMTKView.drawDeferred`, `canvasVisibilityChanged`), and an owner's hand on the trackpad.

5. Spike S2 (quit panel) is in §5.4, spike S3 (Dock reopen of a lone `Window`, `@SceneStorage`) in §6.

**Menus.** Bare letters are never menu key equivalents: they can fire while typing in a text field, and `WorldKeys` already handles them (CORE/WorldKeys.swift:61 ignores Command chords, so
chords cannot clash). The World menu lists actions with the key shown as text in the title where a single-letter key exists; a test compares the shown hints with the active form's key table (§3.4).
Real equivalents are chords only:
```
alife   About / Settings... Cmd-, / Quit Cmd-Q
File    New World... Cmd-N / Branch From This World... Shift-Cmd-B / Make Planet From This World... / Close World (no key; Cmd-W closes the window) / Reveal in Finder Shift-Cmd-R
Edit    standard (text undo comes with SwiftUI text fields) / Copy Command Shift-Cmd-C
World   (items follow the active Watch form) Play-Pause (Space), Step (.), Slower ([), Faster (]), Continue (G), Overview (H), Follow (F), North Up (N), Powers (P), Next Alert (A),
        Acknowledge (K), Next Colour (C); Run Cmd-R / Stop Cmd-. ; Check World... ; Open in Classic 2D Viewer
View    Show/Hide Sidebar (system) / Control Center > Show-Hide / Full Screen (system). Display options stay in the dock's View menu, unchanged (§3.8)
Window  system, Worlds Cmd-1, Watch Cmd-2, Activity Cmd-3, Lab Cmd-4 (until PR-D; the sidebar's order)        Help  alife Help / World Controls (the hint) / Show Welcome
```
Menu items that a SwiftUI `Menu`/`CommandMenu` rebuilds with live state are stabilised as the dock View menu is (`DistinctState`: ViewMenu.swift:4-8; `MenuStabilityTests`).
**Settings (Cmd-,)**: General (alife folder, engine status and *Set up engine*, *Show advanced settings by default*, Show Welcome) and Autopilot (local model URL and name, *Test connection*; today per run).
Display preferences, Pauses and confirmations stay in the dock's View menu, so there is one home each.

### 2.5 Names: one word per action

| Meaning (action) | Word | Replaces | Where |
|---|---|---|---|
| Open a Played world live | **Play** | launcher "Resume" | card, Details, menu File |
| Continue a Research world headlessly (`alife resume`) | **Keep growing** | Lab "Resume" | Details, set-up page |
| Make a new Research world | **Create** | "New run" + Start | New world page button |
| Copy a world from its latest save point and change settings (`alife run --init-from`) | **Branch** | "Fork" | Details, set-up page |
| Make a Played world from a save point (`alife play --init-from`) | **Make planet** | "New planet" | Details, set-up page |
| See the live picture (Played: live; Research: observer) | **Watch** | World / 3D Viewer / Open viewer | page |
| Facts about a world | **Details** | Lab right column | page |
| Un-pause the world at the rate it holds, from the chapter card or `G` | **Continue** | (kept) | chapter card, World menu |
| Pause or run the clock | **Pause** / **Run** (one toggle) | clock-strip "Resume" | clock strip, Space |
| Camera to the overview | **Overview** | "Home" | toolbar button, `H` |
| Dock tab of storms and lineages | **Cast** | (kept; the owner's word: DESIGN 33.4.8) | dock |
| End a background job at a save point | **Stop** (Cmd-.) / **Force stop** | (kept) | Details, Activity |
| End a live world, saving | **Close World** | "Close world" | toolbar, menu |
| Verify a world (`alife inspect --check`) | **Check world** | Inspect | Details |
| Pick up the world where it was left (Settings) | *Pick up where I left off* | "Resume when I come back" | Settings (D4) |
| Repository | **alife folder**; Set up environment > **Set up engine** | | |
| Lever / "Keep inventing levers" | **Setting** / groups Mutation, Immigration, Seasons | | Advanced |
| Checkpoint / Source | **Save point** / **Start from** | | |
| Fast overlay / Overlay / Spin-up / Houses | Lighter brains / Add-on / Weather warm-up / "How family lines are rebuilt" (Advanced) | | |
| Director, personas | **Autopilot**, Style (Gardener, Trickster, Architect, Random control). The engine's `director.*` names stay in the command line and docs. Note: planets also have `PlayDirector.rules` with no UI today (CORE/PlayLaunch.swift:17-20) | | |
| `director.enabled` toggle | not asked: a planet is always made with powers; Advanced "Make a world that takes no powers" | | |
| Preset, Powers, Chronicle, Report, Chapter, Alerts | kept | | |
| Storm Worlds | retired as a name; **Planet** (shape) and **Weather** (a Make planet option) | | |

**Audit (a test, `NameAuditTests`):** a table of every label that triggers an action, mapped to the action id; the test fails if one label maps to two actions or two labels to one action.
Resume now appears nowhere; Continue means only the chapter-card un-pause; Overview only the camera; Play only live play; Keep growing only `alife resume`; "Watch" only the picture. Recount of
F5: three Resume + one Continue + one Home became zero shared words.

---

## 3. Workflows

### 3.1 First launch of a newcomer
```
open alife -> folder known and engine ok ----------------------------------------------> Worlds page (empty: "Start your first world")
           -> no folder: try the stored path, ~/artificial-life, ~/alife (never ~/Documents, ~/Desktop: privacy prompts [I])
                found -> engine check;  not found -> Welcome 1 "Where is alife?" [Choose folder...] (+ "I don't have it": copyable steps)
           -> folder ok, no .venv/bin/alife -> Welcome 2 "Set up the engine" [Set up engine]: `uv sync --inexact --extra viewer --extra video`
                uv missing: plain text + [Open the uv page] [Check again]; failure: last log lines + [Copy log] [Try again]
           -> engine present -> health check `alife --help` exits 0 [U] -> Worlds page
```
A fresh Mac still needs a terminal (or Homebrew) for `uv` and `git`: the first draft's claim of "launch without the terminal" does not hold for that person (D2). With D0 = "owner and
collaborators" this flow is a one-card checklist; with D0 = "real newcomers" it is the full three-step Welcome (PR-G).

### 3.2 What Create does (and how a planet takes two processes)

**The page.** *New world*: Name (suggested), **Starting point** (recipe or preset, with the existing preset card and Compare sheet), **Goal** (visible, two choices: *A world to play* / *A research
world*; Flat shapes allow only the second), **Grow for** (a number of steps, pre-filled, with an estimate line "about 1.5 min at this preset's measured speed" from the manifest), Advanced
(seed, settings, add-ons, autopilot, lighter brains, stop-after timer), *Show command*, **Create**. There is no "until I stop": `--steps` is always a number (the preset default is
shown as a number, with a note "very short: the world will be mostly unevolved" under 5,000); **Stop** at any time ends at a save point. Weather is not on this page: it belongs to
*Make planet*, because a research run has no weather (PlayLaunch.swift:223-237).

**Recipes (data, owner may edit; `Recipes` in `AlifeCore/Recipe.swift`).** **Measured** on the M5 Pro with the real engine (PR-D, 2026-11-01; the estimates of Revision 1 are replaced). "Start" is the engine's own start (imports, compile) before the first step; the first run after an install compiles for longer and was not measured.
| Recipe | Preset | Grow | Measured, stage 1 | Stage 2 (cold weather cache) | Then (Goal = play) |
|---|---|---|---|---|---|
| Small planet | `tiny3s` | 30,000 steps | **106 s** = 16 s start + 334 st/s | **11 s** (10.5 s of the 10,000-step warm-up) | `--overlay stormsph16 --weather-spinup 10000 --set director.enabled=true --houses auto --rate 1` |
| Bigger planet | `mini3s` | 100,000 steps [U: size picked by the owner, D1] | **482 s** (8 min) = 24 s start + 218 st/s | **13 s** (8,000 steps) | `--overlay stormsph22 --weather-spinup 8000` ... (PlayLaunch.swift:108-112,143-147) |
| Flat world | `dev` | 20,000 steps | **60 s** = 3 s start + 352 st/s | none: research only (weather refuses 2D: `weatherObstacle`); no powers, no chronicle | |
The estimate line of the page ("About 1 min 50 s, then about 10 s to make the planet") is these numbers (`Recipe.Measured`); a preset that is not a recipe uses the preset guide's snapshot speed and says nothing about the start. **Through the app, Small planet** (the final bundle, a private folder, the real engine, a cold weather cache): *Create* pressed at 17:29:19, the planet's `play.json` written at 17:31:12, **1 min 53 s to the first moving picture**, 102 s of it stage 1 and 11 s stage 2; the words on the page ("about 1 min 50 s", "about two minutes") hold. The weather warm-up, spec [U], is 10.5 s here and is reused for a repeated world.

**What Create produces:**
| Goal / page | Commands (existing builders, untouched: `RunSettings.arguments`, `PlayLaunch.arguments`) | Worlds created | Lands | Stop / failure leaves |
|---|---|---|---|---|
| Research (any preset) | `alife run --preset P --seed S --steps N --out runs/NAME [--set ...]` | 1 Research world NAME | Watch (observer, Follow) with the Run panel | Stop: SIGINT, final save point, world exists. Failure: world listed "Failed (exit n)" with the log; no save point: Unfinished |
| Play (recipe) | stage 1 as above with `--out runs/NAME-grow`; stage 2 `alife play --init-from runs/NAME-grow --overlay O --weather-spinup W --set director.enabled=true --houses auto --out runs/NAME --rate 1` (newest save point, no `:STEP`: PlayLaunch.swift:224-226, so stage 2 is fully known before stage 1 ends) | 2 worlds: **NAME** (Played, what the user plays) and **NAME-grow** (Research, its seed) | Watch (observer) during stage 1, then Watch (live) | see the pipeline table |
| Make planet (from a world, any save point) | stage 2 only; the **save-point picker is kept** (WorldLauncherView.swift:197-204) | 1 Played world | Watch (live) | Stop: SIGINT now; **Unfinished** NAME (nothing is deleted) |
| Branch (from a world's latest save point) | `alife run --init-from runs/SRC --out runs/NAME --set <only changed>` | 1 Research world | Watch (observer) | as Research |
| Keep growing (Research world) | `alife resume runs/W [--steps N]` | none (same world) | Watch (observer) | as Research |
"Show command" shows two labelled blocks (Stage 1, Stage 2) and Copy gives a two-line script joined by `&&`. In the library NAME shows "Grown from NAME-grow" and NAME-grow is
nested under it in the sidebar, so one click is one world to the user and nothing is hidden. The set-up page's first name suggestion names the *played* world (`planet_c`).

**Pipeline "Grow, then play"** (new pure state machine `PipelinePlan` in AlifeCore; the glue is new code, the command builders are not touched):
| State | Shown | Stop / Cancel | Failure | Window closed / app quit |
|---|---|---|---|---|
| 1 Growing (a run job) | banner "Growing the world: step n of N. Next: make it a planet." in Watch (observer); Activity row; Dock progress if D3 = C | **Stop**: SIGINT; pipeline **parks**: "Growing stopped at step n. [Make planet from here] [Keep growing] [Done]". Never auto-continues | exit != 0: "Growing failed: <first error line>", [Show log] [Try again with a new name]; NAME-grow kept; pipeline ends | Window closed: the run continues (D3); at the end, with no visible Watch, the pipeline **parks** at "Grown. [Make planet]" and the Dock icon bounces once. Quit: confirmation, SIGINT, wait (§5.4); on relaunch nothing starts: the Research world shows "Make planet" |
| 1 to 2 | "Grown. Making your planet..." | | | Stage 2 starts **only if** this pipeline's Watch is the visible page, the window is key and the app is active; else park. **Nothing starts invisibly** |
| 2 Making planet (play process `starting`) | `play.progress` stderr lines (fork, houses, warm-up) + elapsed | **Cancel**: SIGINT now | `phase = .failed(msg)`: banner with message; **Unfinished** NAME with [Reveal] [Make again with a new name] (spec kept in memory for the session) | close: `play.stop()` interrupts (as today: PlayController.swift:173-184); quit: the gate interrupts a `starting` world |
| 3 Live | normal Watch | Close World (saves) | | as §5 |

### 3.3 Come back to a world
Launch: Worlds page, **Play "mini3s_a"** (the most recently played Played world) as the primary button, "step 450,000, saved yesterday". Nothing starts by itself. Play opens at **1x** (each
Play passes `.multiplier(1)`: WorldLauncherModel.swift:65-67; no rate is remembered, and none is promised). The chapter card that was waiting is shown first (the backlog exists:
PlayController.swift:84-86); the user reads, then **Continue** (`G`). Another world: select it; Details shows; **Play** switches (§5.2 R5).

### 3.4 Watch and decide: two kinds, two sets of controls
| | Played world (live Watch) | Research world (observer Watch) |
|---|---|---|
| Renderer | `WorldMetalView` / `SphereMetalView` fed by `alife play` | `ViewerMetalView` / `SphereLabMetalView` fed by `alife stream` |
| Forms | live only | **Follow** (while a run grows it: newest save point, so it moves at the preset's `checkpoint_every`: 50 steps for `tiny3s` (tiny3s.toml:54), 10,000 by default (config.py:305), i.e. jumpy for big presets) and **Re-simulate** (steps the latest save point in a separate process, without the director) (Viewer3DView.swift:80-91) |
| Can | pause, step, rates, powers, chronicle, reports, alerts, chapter pauses, storage strip, follow organisms, minimap | pause/run, step (Re-simulate) or *Newest* (Follow), steps per frame, colour, agents, bonds, frame, field slice, camera orbit/fly, reset, **Snapshot** PNG (Viewer3DView.swift:95-153) |
| Cannot | | any power, chronicle, report, weather map, chapter pause, storage strip, being a "world" the game can pause for you |
| Docked panel | control center (Cast, Chronicle, Report, Powers, clock, storage, alerts) | **Run panel**: Numbers, Autopilot journal, Log (tabs); same dock side and divider as the control center |
| Keys | `WorldKeys` (CORE/WorldKeys.swift:46-56): `f` follow organism, `s` side view, `o` oblique, `r`/`h` overview | the viewer table (`f` field slice, `s` snapshot, `o` orbit/fly, `r` reset view; Viewer3DView.swift:98-153) |
The two tables conflict on `f`, `s`, `o`, `r` and are **not merged**: each form has its own, the World menu follows the active form, and Help > World Controls shows the active one. When a Research
world is *Branched into a Played one*, the user changes form and key meanings with the picture, which the toolbar states ("Observer" chip vs "Live"). A research world's Watch page says in one line
what it cannot do ("This is an observer view. To play it, make a planet from it.").

Anatomy and decisions are today's (W5). Leaving the Watch page pauses a live world (D4); a **Run job's** observer stream is stopped when the Watch page is left (`onDisappear { stream.stop() }` is kept:
Viewer3DView.swift:69-70), and the numbers and log remain on the Run panel only while Watch is shown; the Activity page and the Set-up window keep the log.

### 3.5 Change a setting, with the consequences explained
A world's settings are part of its history, so a change is a **Branch** (today's Fork; Resume refuses levers: ContentView.swift:217-220), said in one sentence at the top. Each row: plain name,
value with unit, effect line (today's `help`), chip *default* / *changed from source* / ***remembered from last time*** (lever edits are remembered, and shown as a chip on the row and a
line "Remembered from last time: 2 settings [Reset all]" at the top, so a hidden edit never travels into a new world), reset arrow, error under the field in plain words with the key in
a tooltip. **Plain-language messages are written for the default path only** (mutation, immigration, seasons, 3D basics, name/seed/steps, folder and engine): the keys a first-time user can reach
(CORE/Levers.swift:56-90); others show the engine's message. A **Changes** list at the top: "3 settings changed: Gravity 0.03 to 0.05; ...". Branch keeps *latest save point* only (as today); *Make planet* keeps the picker.

### 3.6 Run a comparison experiment (advanced user)
Branch with one setting changed, Keep growing or Create, repeat; each branch is a world "branched from A at step N" with its changed settings in Details; compare outside the app, as today. In-app
comparison, a run queue and "Vary a setting" are **not** in this program (§9.2); noted as a possible later project.

### 3.7 What stays one click for experts, and the live picture beside numbers and log (H7)
Today's expert layout is picture + form + log at once (separate windows). The lightest way to keep it, with **no second played world** (R1 intact):
(a) a Research world's Watch has the **Run panel** docked (Numbers, Autopilot, Log), the same dock mechanics as the control center, so picture, numbers and log are one screen;
(b) the **set-up page can open in its own window** (toolbar "Open in window") so a form can be composed beside a Watch; both read the same singletons (`RunController.shared`, `StreamController.shared`),
so no model duplication; (c) the global log is also the Activity page. Not offered: a second observer or a second Watch window (one `alife stream`).
| Expert need | New home |
|---|---|
| Every lever | Create / Branch page, Advanced (remembered; also Settings > General "Show advanced settings by default") |
| Exact command | Show command (footer, with its Copy command button); remembered |
| Start / Stop keys | Cmd-R / Cmd-. |
| Timer | set-up page "Stop automatically after" |
| Force stop | Details, Activity, World menu |
| Autopilot (personas, goal, LLM tier, yoke, replay, test) | Advanced > Autopilot; Settings > Autopilot (endpoint) |
| Add-ons (arms, structures, bonds, comm, farming) | Advanced > Add-ons |
| 3D / planet levers | Advanced (planet shape fixed in a branch: ContentView.swift:326) |
| pygame viewer | World > Open in Classic 2D Viewer |
| Inspect, Reveal, Log | Details; File; Run panel and Activity |
| `--world RUN` | kept (§6) |
Slower than today: starting a run is Worlds > New world > Advanced > Create (one more click); a Branch is select > Branch (one more); `--world` and Cmd-N shorten both.

### 3.8 Every current feature, mapped (one home each)
| # | Today | New home |
|---|---|---|
| 1-3 | Repository field/Choose/status; Set up environment; Re-check on activation | Settings > General; Welcome; banner on Worlds when unusable; automatic on activation |
| 4 | Mode New / Resume / Fork | Create / Keep growing / Branch (Make planet for the game) |
| 5-6 | Source picker, Refresh; name + New name | the selected world; auto-refresh (§5.3); name field with suggestion |
| 7 | Preset picker, card, Compare | set-up page (card reused as is) |
| 8-9 | Fast overlay; Seed, Steps | Advanced: Lighter brains; Seed; "Grow for" |
| 10-13 | Mutation/immigration/season levers; 3D levers; planet levers; feature layers and overlays | Advanced groups |
| 14 | Director box, live panel, test connection | Advanced > Autopilot; Run panel > Autopilot; Settings > Autopilot |
| 15-16 | Timer; Command box | footer "Stop automatically after"; Show command |
| 17 | Start / Stop / Force stop / Cancel | **Create** (Cmd-R), Stop (Cmd-.), Force stop (Details, Activity) |
| 18 | Status: Step, Population, Steps/s, Clades, Generation max, Ledger error, Height mean, Grounded, At surface, and the feature-metric groups (arms, structures, bonds/roles, comm, farming) (ContentView.swift:443-471,478-494) | Run panel > Numbers: all of them, with plain labels and the engine name in the tooltip |
| 19 | Open viewer (pygame), Open 3D viewer, Open World | Classic 2D Viewer (menu); Watch; Play |
| 20-21 | Inspect, Reveal; Log Clear/Copy | Check world, Reveal in Finder; Run panel and Activity |
| 22 | Resume list with obstacles | cards with state chips and reasons (§5.2 matrix) |
| 23 | New planet form incl. **Checkpoint picker** | Make planet page, **Start from + Save point** picker kept |
| 24 | Unfinished builds (read-only list, "nothing is deleted") | sidebar group Unfinished (n): each row has the reason (which stage, from the log if this session), **Reveal in Finder**, **Make again with a new name** (this session only); no deletion |
| 25 | Starting box with progress and Cancel | Watch banner; Activity row |
| 26-31 | Dock side picker/collapse/divider, Overview button, Close world, clock strip, storage strip, alerts, chapter card, tabs, inspector | **unchanged in place** (one home each); transport is the clock strip only; dock side is the toolbar picker only; Close World is the toolbar button (+ menu entry) |
| 32 | Dock View menu: Colour (`c`), Intent labels (stream), Agents, Agent points, Bonds, Bond flow, Structures, Soil, Territory, Organism size, Weather map, Brightness, Graticule/Frame, Minimap, Globe, Spin, Keep north up, Scroll zooms, View angle, Tilt, Pauses, "Skip the second click", Raw kin events, Controls... (ViewMenu.swift:77-160) | **stays as the dock's View menu, unchanged** (its stabilisation is kept); the menu bar only adds Next Colour (C) and Help > World Controls |
| 33 | 3D viewer: Source Follow/Re-simulate, Pause/Step, steps/frame, colour, Agents/Bonds/Frame/Field slice, Camera Orbit/Fly, Reset view, Snapshot (Viewer3DView.swift:78-153) | observer Watch toolbar, unchanged; the Snapshot button stays (the World view has none, so no Cmd-S) |
| 34-37 | `--world`, `--launch-mode`; Settings (Storm); Window menu items; display preferences | kept / ignored after retirement; Settings; Window menu; migrated (§6) |

---

## 4. Wireframes

W1 Welcome (engine missing).
```
+------------------------------------------------------------------------------------+
|  Welcome to alife                                                                  |
|  (1) Find alife        done      ~/artificial-life                       [Change...]|
|  (2) Set up the engine current   It is not set up in this folder yet.              |
|        [ Set up engine ]    installs packages with uv   > Details                 |
|  (3) Make your first world  waiting        uv missing?  [Open the uv page][Check again]|
+------------------------------------------------------------------------------------+
```
W2 Worlds.
```
+--------------+---------------------------------------------------------------------+
| Worlds       |  Worlds                                       [search]   [+ New world]|
| > mini3s_a   |  [ Play "mini3s_a" ]  Played - Planet - step 450,000 - saved yesterday|
|    mini3s_a- |  All  Played  Research  Unfinished (2)                                |
|    grow      |  +-------------+ +-------------+ +-------------+                      |
|   dev_s1     |  | [snapshot]  | | [snapshot]  | | [snapshot]  |                      |
|   Unfinished |  | mini3s_a    | | grow_t3s    | | dev_s1      |                      |
| Activity (1) |  | Played      | | Research    | | Research    |                      |
|              |  | Paused      | | Growing 41% | | Saved       |                      |
|              |  | [Play]      | | [Watch]     | | [Branch...] |                      |
+--------------+---------------------------------------------------------------------+
```
W3 Create (goal "A world to play").
```
|  New world                                                       [Cancel] [Create]   |
|  Name [planet_c]   Starting point [Small planet v] (card: snapshot, what, cost, speed) |
|  Goal (x) A world to play  ( ) A research world        Grow for [30,000] steps (~1.4 min)|
|  > Advanced settings   Remembered from last time: 2 settings [Reset all]               |
|  > Show command   Stage 1: alife run ...   Stage 2: alife play --init-from ...         |
|  This makes 2 worlds: planet_c (you play it) and planet_c-grow (its seed).             |
```
W4 Watch, Played world (control center right, owner's layout).
```
| [=] mini3s_a  Live 1x   [L|R|B] [Overview]                       [Close World]   (!)  |
+------------------------------------------------------+--------------------------------+
|  the planet / map: most of the window                | name, chapter bar, rate, step  |
|  readout, scale bar, compass, minimap (DESIGN 36.5)  | storage, alerts                |
|  [ chapter card: Continue | 2x | Powers | Show ]     | [Cast|Chronicle|Report|Powers] |
+------------------------------------------------------+--------------------------------+
```
W5 Watch, Research world (observer; Run panel docked).
```
| [=] grow_t3s  Observer - Following run 41%   [L|R|B]  [Pause][Newest][colour][Snapshot] |
+------------------------------------------------------+--------------------------------+
|  the 3D picture (observer; not playable)             | [Numbers | Autopilot | Log]     |
|  "To play it, make a planet from it."                | Population 988  Speed 361 st/s |
|                                                      | step 12,300 / 30,000  [Stop]   |
+------------------------------------------------------+--------------------------------+
```
W6 Quit dialogs (§5.4).
```
 +-----------------------------------------+   +--------------------------------------+
 | Stop "grow_t3s" and quit?               |   | Saving "mini3s_a"...                 |
 | It writes a save point first (seconds). |   | [=======>      ]  you may lose up to |
 | Continue it later with Keep growing.    |   | [Quit now without saving] after 10 s |
 |        [Cancel]   [Stop and Quit]       |   | about 7 min of play at 1x.           |
 +-----------------------------------------+   +--------------------------------------+
```

---

## 5. Safety of the running world

### 5.1 Today
See §1.1 (Closing windows, Processes, Locks, Quit). One more fact: `PlayController.start` replaces a live world through its ladder (PlayController.swift:123-127), so *Make planet* while a
world is live replaces it, and the launcher disables its buttons instead of saying so (F8).

### 5.2 Rules and the state x action matrix

- **R1 One played world, one run job, one observer stream.** A run and a played world may coexist; Watch/Details then say "A run is using the GPU; the world may be slower" [I: ContentView.swift:542].
  A second window exists only for the set-up form (§2.4); no second Watch.
- **R2 A world is saved before anything stops it.** Stop, close, switch, quit go through the ladder (§5.4 for the gate's change).
- **R3 Closing the window** stops what only the window needed (a live world, an observer stream: today's behaviour) and **not a run job**; whether the app then stays alive is **D3**.
- **R4 Leaving the Watch page** pauses a live world: **D4**. Returning resumes only a world that was running when left (not one paused by the user, a chapter or an alert).
- **R5 Switching.** Play on world B while A is live: one sheet "Stop A to play B? A is saved first (about N s)." Make planet while any world is live: the same sheet (it replaces the live world:
  PlayController.swift:123-127). A world still saving shows "saving..." (WorldLauncherModel.swift:59).
- **R6 One writer per directory.** `DirectoryGuard` (pure, AlifeCore) records, at click time and again at start, every directory this app has a *writing* process on (Play, Keep growing, Create/Branch/Make
  planet target, retired savers). An action that would add a second writer is disabled with the reason. Foreign writers: `play.lock` (`PlayLockProbe`, covers a played world in another app) is probed;
  a foreign `alife run`/`resume` has no lock, so the app can only go by what the library sees (`In use elsewhere`, from fresh output): as built, that **refuses** the action with the reason, like a foreign played world (no warn-only path). This closes "Keep growing, then Play" (`alife play` checks no run lock: cli.py:1342-1347).
- **R7 Opening the app starts nothing**; restoration restores the selected page only.
- **R8 Keep-awake** for a run stays (RunController.swift:475); Activity shows it.

State x action (Played = has `play.json`; Research otherwise). "ok" = enabled; text = disabled with that reason; `sheet` = R5 sheet.
| Action \ State | Saved | Live / Paused (this app) | Growing (this app) | Saving (retired) | In use elsewhere | Unfinished |
|---|---|---|---|---|---|---|
| **Play** (Played only) | ok (`sheet` if another world is live) | already open: go to Watch | n/a | "saving..." | "being played in another window" | n/a |
| **Keep growing** (Research only) | ok (if no run job) | n/a | "already growing" | n/a | "in use by another process" [I] | "not a finished world" |
| **Branch** (reads latest save point; starts a run job) | ok (if no run job: "a run is going") | "close it first" (it is being written) | ok, from its newest save point (note shown) | "saving..." | warn | n/a |
| **Make planet** (reads a save point; **replaces a live world**) | ok (`sheet` if a world is live) | "close it first" | ok, newest save point (note) | "saving..." | warn | n/a |
| **Watch** | Played: Play; Research: Re-simulate (starts a stream) | live | Follow | "saving..." | Research: Re-simulate; Played: n/a | n/a |
| **Check world** | ok | "wait until it is saved" [I: conservative] | "wait until it is saved" | "saving..." | warn | ok |
| **Stop** / **Close World** | n/a | Close World | Stop | n/a | n/a | n/a |
| **Reveal in Finder** | ok | ok | ok | ok | ok | ok |
Correction to Revision 0: *Branch* (a headless run) does not disturb a live world; *Make planet* does (R5).

### 5.3 Library scan, off the main thread
`PlayCatalogScanner.scan` lists every directory and reads `config.json`, `meta.json`, `play.json` and the `checkpoints`/`events` directories (PlayWorlds.swift:146-196); today it runs on the main
actor when the launcher opens (WorldLauncherModel.refresh), and `RunController.refreshRuns` lists names and probes locks every 2 s (RunController.swift:252-276). `WorldLibrary` runs the scan in a detached task,
caches per directory by modification date, refreshes on app activation, on the Worlds page appearing and on a change of `runs/` (a `DispatchSource` on the directory), and keeps only the cheap lock probe on a
timer while a world page is visible. Dangling entries on an unmounted drive are skipped (PlayWorlds.swift:152) and counted: "N worlds are on a drive that is not connected" [I].

### 5.4 Quit, step by step
1. `applicationShouldTerminate` first runs a **pre-confirmation** (NSAlert, before the gate): if a run job exists: "Stop 'X' and quit? It writes a save point first; continue it later with Keep growing." [Cancel]
   [Stop and Quit] (Cancel returns `.terminateCancel`). If only a world is live: no question (as today) and no data loss.
2. The gate: `shouldTerminate(hasProcess:stop:reply:)` keeps its signature and guarantee (AppTermination.swift:4-8). Two things are added: `hasProcess = play.hasProcess || run.isRunning`, and
   one method **`abandonWait()`** (cancels the wait and replies now) plus an `onWaiting` callback (elapsed, which world). `AppTerminationTests` stay untouched; new tests cover the two additions.
3. **The panel** is an independent `NSPanel` owned by `AppDelegate` (not SwiftUI window content), so it exists when the quit came from closing the last window and no window is open. It shows "Saving
   'X'...", the elapsed time and, after 10 s, **Quit now without saving**. Honest wording: "You may lose up to about N minutes of play at 1x" with N from `play.json`'s `base_rate` and the 6,000-step
   save cadence (cli.py:1072-1078; play.py:119,124,127), about 7 minutes at 14.4 steps/s, including any powers cast since. SIGTERM itself is safe: saves are written to a temp file and renamed (runio.py:253-265).
4. **Orphans.** `RunController.shutdown()` today only calls `stop()` (RunController.swift:570-574), which is guarded by `!stopRequested` and `job == .run` (:446), so after the gate's 40 s a run
   (and an unfinished `uv sync`/`inspect`) is **left running**. New rule: after the user chose Stop and Quit, `shutdown()` escalates: SIGTERM to the run job and to any setup/check job (the same SIGTERM
   `forceStop` uses), losing at most the steps since the last save point, which the confirmation text states. `StreamController.shutdown()` already terminates its processes (StreamController.swift:133-140).
5. **Spike S2, done 2026-11-01 (PR-A): it passes; the fallback is not needed.** In the running app (the debug binary a `swift run` starts, a private
   preferences suite, a real run going, Cmd-Q then Stop and Quit, the engine made slow to save): `applicationShouldTerminate` returned `.terminateLater`
   and the run loop then ran in `NSModalPanelRunLoopMode` (49 of 50 samples; the one other was taken before the quit); main-queue blocks, `Task.sleep` and
   so the gate's polling kept running in it (50 of 50); the `NSPanel` (`QuitPanel`, `worksWhenModal`, level `.floating`) was on screen for 40 s (window-server
   list: on screen, layer 3; `occlusionState` visible); its text changed every second ("Elapsed 0 s" to "Elapsed 39 s", 40 distinct values); 47 captures of the
   panel as the window server composes it were all different (the elapsed text and the spinner), and one read by eye shows the title, the spinner, the
   elapsed time and, after 10 s, the warning line and the **Quit now without saving** button; a mouse-down and mouse-up pair posted to the application's
   event queue at the button's centre ran its action (`abandonWait()`: the app replied at once, `applicationWillTerminate` sent SIGTERM, and no process of the build was left).
   **Limits of the evidence.** The actions were injected in the process (a dylib loaded into the debug binary, kept outside the repository): the click was a
   posted event, not a hardware one, and Cmd-Q was `NSApp.terminate(nil)`, which is what the menu item calls. A system logout was not exercised (a quit Apple event with the log-out reason was sent to the process itself in the later runs, §5.4 as built (b)); a real Dock Quit with another app in front was not;
   the display session was awake for those runs but twice went to sleep during the day (windows then report themselves occluded and draw nothing), which is
   why the evidence comes from the runs where it was awake. The fallback (panel during the confirmation phase, a `terminate(nil)` re-entry, the Dock badge) is
   not built; it stays written here in case a later macOS changes this.

**As built in PR-A** (differences from the steps above, and what the independent review of PR-A changed; each small):
(a) the question says "continue it later from the Lab with Mode: Resume" because *Keep growing* does not exist before PR-D (the wording changes with the rename); it is not asked when the run is already stopping (the user pressed Stop), while a wait is in progress or while the question is already up (a second quit then is the first one's business), and Cancel is the default button, Stop and Quit the destructive one.
(b) **Why the app is quitting is read** (`kAEQuitReason` of the quit Apple event, `QuitReason`): for a log out, restart or shut down nothing is asked, because nobody is there to answer and an unanswered alert would hold the machine; the run is stopped, and the wait and panel follow. A sent event with reason "log out" was exercised for real (a self-sent Apple event; no logout occurred).
(c) **The wait is not 40 s for everything.** A run waits at most 120 s (a large state or an LLM call in flight can take longer than 40 s, and SIGTERM drops the save in flight); a played world, a setup and a check keep 40 s (the world's own ladder is 30 s + 10 s). The panel says so ("Elapsed N s of at most M s", and what is lost at the limit, from the start), and the question says 120 s. The gate has a `limit` closure for it; its `init` timeout and `AppTerminationTests` are unchanged.
(d) **A setup or a check at quit** gets the same SIGINT, the same gate and panel, then SIGTERM at the limit; no question (uv and `alife inspect` handle SIGINT by finishing cleanly). Before PR-A a setup was left running after the app quit; since the first PR-A commit it was killed with SIGTERM and no SIGINT first; now it is asked to stop first. LLMCheck's connection test (a short HTTP probe) is not tracked.
(e) `onWaiting` carries the elapsed time only (the delegate knows which worlds and run are saving); the panel appears 0.4 s into the wait; `play.json` is read once per world; "Quit now without saving" has a keyboard route (the panel becomes key, Space, Option-Command-Q); `shutdown` has the parameter `escalating`, set from `LaunchPlan.quitWaitsForRun`.
(f) **Sheets.** With a sheet attached to a window AppKit ignores `terminate` and never asks the delegate (reproduced in plain AppKit and on `main`; AppKit's behaviour, not this code's). PR-A adds a Quit command (`QuitCommands`, in place of SwiftUI's) and a quit Apple-event handler that ask the question first (so that Cancel leaves the sheets), dismiss the sheets through their own cancel action (Escape; AppKit's `endSheet` after half a second for a sheet that has none: it leaves a SwiftUI sheet's state out of step) and then call `terminate`, from a run-loop block (calling it from a task or a dispatch block deadlocks the gate's polling; found by running the app). Exercised for real on the Storm Worlds launcher sheet with the Quit command and with a quit Apple event; the Lab's preset comparison is the same mechanism and was not opened in the scripted runs.
(g) **A stored folder is never replaced** (a drive that is not connected, a deleted worktree): it stays chosen in memory and on disk, the status says the folder cannot be found, and a button offers the detected folder (`~/artificial-life`, then `~/alife`) or "Choose..."; detection only fills an empty choice. In both modes.
(h) Loss wording (L7): "7 minutes" is play at 1x counted in steps (6,000 steps at 14.4 steps/s; at 0.25x the same steps take four times as long); the periodic save is a background write, so the true worst case is a little more. A run's loss carries no number (its checkpoint cadence is its own).
(i) **Known and left as they are.** Closing the last Lab window while a run goes and answering Cancel leaves the app with no window (the Dock icon brings it back); a run started while the quit waits is not stopped by the gate (unconfirmed that it can be reached: the main window gets no events in that loop, but Command-R is a keystroke).

---

## 6. Migration

**Names.** `CFBundleName`/`CFBundleDisplayName` `alife`; bundle `alife.app`; `CFBundleExecutable` `alife`; `AlifeLaunchMode` `alife`; id **`io.github.magicbundle.alife`** (decision M1), which is
today's shared preferences domain, so `repoPath` carries over; a bundle cannot open its own id as a suite (`UserDefaults(suiteName:)` of the own id returns nil, so `RepoPathStore.forApp`
already falls back to `standard`: RepoPathStore.swift:81-83), so after retirement the store reads `standard` plus the two old domains by name. Category simulation-games; version 0.2.0 at PR-H.
**Preview build (PR-B onward):** `make-app.sh --app alife --preview --install` makes **`alife Preview.app`**, id `io.github.magicbundle.alife.preview`, own preferences; it installs beside the old apps and
changes neither: it **reads** the shared domain's `repoPath` and the old apps' preferences and **never writes them** (the preview's own choice of folder is stored in its own domain, `RepoPathStore`'s read-only mode; as built, §7.1). Its id is followed by `.`, so `tests/test_app_bundles.py:49-51` (`suite(?![\w.])`) stays green; the final id replaces the preview in PR-H.

**Settings.** `PreferenceMigration` is idempotent, never deletes, sets `migrated.v1`. `PreferenceDomain` is string-only (RepoPathStore.swift:5-8), but the `world.*` keys are Bool, Double and a JSON
**string** (`world.autopause`), so PR-B adds a **typed domain protocol** (`object(forKey:)`, `set(_:forKey:)`) next to it, with `MemoryDomain` for tests. Order: `repoPath` from the shared domain, else
Storm's, else the Lab's; the 14 `world.*` keys (§1.1) from Storm's domain, else the Lab's, only where unset (including `world.hintShown`). Window frames and saved state are **not** migrated (names derive from scene ids [I]).
**TCC [I]:** a new bundle id and signature re-ask folder access (`~/artificial-life`), **Removable Volumes** (an archived `runs/`), and **Local Network** if the director's model is on the LAN
(the engine child is attributed to the responsible app); Full Disk Access given to `AlifeControl.app` does not carry; ad-hoc signatures change per build (the script already prefers Developer ID). No keychain
items, entitlements or sandbox exist, so nothing else is affected.

**`--world RUN`** is parsed once (CORE/WorldLaunchArgument.swift), starts RUN through the same path as *Play* (the model exists before any page: §2.4 item 1), works only on a fresh launch via
`open -a` (gui/README.md:82). `--launch-mode X` is accepted and ignored after PR-H.

**The old apps** stay installed and working; they share `repoPath` through the same domain; `play.lock` stops two players of one world (cli.py:1130-1135), but not a run plus a play (R6 warns).
`make-app.sh --install` prints their paths and removes nothing (M2). Three apps in Launch Services until the owner removes them.

**Icon:** a third variant `alife` in `make-icon.swift` (the storm icon's world without the bolt; the owner's eye is needed); the "icons differ" check (check-apps.sh:75) becomes "icon is an icns".
**Names of targets:** keep the SwiftPM target and directory `AlifeControl` (46 test files `@testable import AlifeControl`; `tests/test_gui_sync.py:819` reads
`gui/Sources/AlifeControl/StorageViews.swift`). `alife.app` and `Alife.app` are one name on a case-insensitive filesystem, and `Contents/MacOS/alife` is alone in its directory; the app starts the engine by
absolute path (RunController.swift:187). Hazard: `pkill -f alife` now also matches the app; documented.
**`make-app.sh`** `--app alife|lab|storm|both` (default `alife` at PR-H; `lab|storm|both` print a deprecation line until then), `--preview`.
**`LaunchMode` and its pinned tests (corrected).** Adding `case alife` is not additive: `LaunchModeTests.swift:44` (`allCases == ["lab","world"]`) fails and `LaunchPlan.windowsAtLaunch` is an exhaustive `switch`
that stops compiling (CORE/LaunchMode.swift:81-86). **PR-B edits exactly these**: the test at :44 and the switch; `tests/test_app_bundles.py:42` (`A_MODE="(\w+)"` == ["lab","world"]) gets "alife"; the
bundle-id test at :45-51 is extended with the preview id. The remaining 12 `LaunchModeTests` tests, the 15 `RepoPathStoreTests` and the 3 `StormWorldsSharedRepoTests` stay until PR-H, which rewrites them
(`LaunchModeTests` into `AppPlanTests`) together with `check-apps.sh:68-75` and the rest of `tests/test_app_bundles.py`.
**Docs (counts corrected).** DESIGN.md §33.4.10 is rewritten as "The alife app" with a Review-log row (id from the coordinator) in PR-H; DESIGN.md has 60 lines naming an old app (20 say "Storm Worlds")
and most are history; `gui/README.md` has 20 lines (its "Install as an app" is rewritten); `docs/HANDOVER.md` has 10 lines (749-757 and the click-through checklist); `CLAUDE.md:17` names a *path*
(`gui/Sources/AlifeControl/...`), which does not change; top-level `README.md` gains one paragraph.
**Spike S3, done 2026-11-01 (PR-B): reopening works with an explicit handler; `@SceneStorage` could not be shown to work here, so the page is a preference.**
The same throwaway app as S1, as an ad-hoc bundle (`open -a` on a running app sends the same reopen Apple event as a Dock click; a real Dock click was not
made). With `applicationShouldTerminateAfterLastWindowClosed` false and the lone `Window` closed: the app lives on with no window (visible windows 0 for 14
s). *SwiftUI's default does not reopen it:* `applicationShouldHandleReopen(hasVisibleWindows: false)` was called twice and no window came, with the default
`return true`. *With the handler calling a captured `openWindow(id:)` and returning false,* the window came back at once, **the same hierarchy**: one window,
`makeNSView` still 1, `dismantleNSView` 0, the selected page unchanged (a closed `Window`'s content is kept alive, so the Metal view and the model survive a
close). The captured `openWindow` (stored when the content first appeared) still worked after the window closed. *`@SceneStorage`:* in five attempts (direct
launch, `open`, `-NSQuitAlwaysKeepsWindows`, the app's own default, `applicationSupportsSecureRestorableState`) the page was **not** restored after a normal
quit, and macOS 27.2 wrote no saved-state files at all for the spike, so the property is unproven here; the window reports itself restorable. **Outcomes for
the plan: D3 = C stays feasible** (it needs `applicationShouldHandleReopen` + a captured `openWindow`, as planned, PR-E); the selected page is kept in the
app's own preferences (`shell.page`, never Watch), not in `@SceneStorage`, so PR-F must not depend on the latter. The question of D3 for the owner is
unchanged: the preview built in PR-B still quits with its last window (as the old apps do), because R3 is PR-E.

---

## 7. Build plan

Principles: every PR leaves a working app and green tests; no engine change, config key or trajectory change (the Python suite is affected only by `tests/test_app_bundles.py`); each PR names the pinned
tests it edits; the launcher and the Lab form are removed from the new app only after the set-up page replaces them (no gap in which a planet cannot be made). Sizes are changed lines, code plus tests, as
ranges; my earlier point estimates were low (the review judged 1.5x to 2x: I accept it for the set-up page, the shell and Watch/Details).

| PR | Content | Size | Mode / gating | Fixes |
|---|---|---|---|---|
| **A. Safety and quick fixes for today's apps** | R2 and §5.4: pre-confirmation, gate extended to the run, `abandonWait()`, independent panel (spike S2 first, fallback as written), `shutdown()` escalation; F2: a plain reason next to Start; F3: tries stored, `~/artificial-life`, `~/alife` and no default text; F23: Cmd-R. New behaviours are `LaunchPlan` flags (`confirmsRunOnQuit`, `quitWaitsForRun`), **on for both existing modes** (they are fixes); R3/R4 are not here | M 500-900 | all modes; `AppTerminationTests`, `LaunchModeTests` untouched (flags are new properties) | F2, F3, F15, F23 |
| **B. The shell, with a preview you can install** | **Spike S1 first** (pass criteria §2.4; choose primary or F1 or F2), S3; `case alife` (edits in §6); hoist `WorldSession`, `watchActive`, key gate; `AlifeApp` single `Window`, `NavigationSplitView`: *Worlds* (the launcher's content, **including the New-planet form**, as a page), *Lab* (today's `ContentView` as a page), *Activity*; menu bar and Settings > General; typed preference domain and `PreferenceMigration`; `--world`; `make-app.sh --preview`, `check-apps.sh` row; icon | L 1,600-2,600 | `.alife` only; old modes unchanged | F11-F13, F21; **owner can install `alife Preview.app` from here** |
| **C. The library** | `WorldLibrary` (off-main scan, §5.3), sidebar rows with states and nesting, Worlds page with *Play*, search, filters, Unfinished with reasons and Reveal; the New-planet form still opens from the card ("Make planet...") | L 900-1,600 | `.alife` | F8, F20 |
| **D. The set-up page and pipeline** | Create / Branch / Keep growing / Make planet page, recipes, `PipelinePlan` + glue, `DirectoryGuard` (R6), Show command (two blocks), Goal, Changes list, remembered values with chips, plain messages for the default path, name audit (`NameAuditTests`), "Open in window"; then the Lab and launcher pages are removed **from the preview** | XL 2,000-3,400 | `.alife` | F1, F5, F6, F7, F9, F14 |
| **E. Safety rules in the shell** | R1, R5 sheet, R3 per D3, R4 per D4 (attention predicate with `watchActive`), reopen handling, Dock progress if D3 = C | M 400-800 | `.alife` | F15, F16 |
| **F. Watch and Details** | Research Watch form with the Run panel (Numbers incl. Height/Grounded/Surface and feature groups, Autopilot, Log), Details page **decoupled from `targetRun`/`trackedRun`** (a per-world metrics/director reader), World menu by form, Help, Settings > Autopilot, Cast kept, `SceneStorage` | L 1,200-2,200 | `.alife` | F10, F17-F19 |
| **G. Welcome** (size depends on D0) | detect, choose, set up with progress, `uv` guidance, health check, `alife play --preset` recipe (A) behind D1 via an additive `PlayLaunch.arguments(preset:)` | S-M 300-1,200 | `.alife` | F1, F3, F4 |
| **H. Retire the old modes** | remove `StormWorldsApp`, `WorldWindowRoot`, `RepoGateView`, `LaunchPlan` fields; flip the default key; final bundle id replaces the preview; rewrite the pinned tests (§6); docs | M -1,800..-1,000 deleted, +300..+600 added | all | F12 |
Dependencies: A independent (ship first); B before C-H; C before D, E, F; D before G; H last. After PR-B the owner can dogfood while C-G land.

### 7.1 As built in PR-B (the shell; what differs from the row above)
- **Bundle:** `alife Preview.app`, id `io.github.magicbundle.alife.preview`, executable `alife`, `AlifeLaunchMode` `alife`, a third icon (the storm icon's world, a few stars, no cloud or bolt; the owner's eye is needed). `make-app.sh --app alife` requires `--preview` (the final `alife.app` is PR-H); `check-apps.sh` builds and checks it too. The Python `tests/test_app_bundles.py` edits are the two §6 names (the mode list, the preview id) plus one new test.
- **Window:** one `Window("alife Preview", id: "main")`, minimum 1,100 x 700 for every page, sidebar **Worlds, Watch, Activity, Lab** (Cmd-1 to Cmd-4 in that order) with a second line (the open world and its state; what runs; nothing for the others) and an engine-status line at the foot. Window menu: Worlds Cmd-1, Watch Cmd-2, Activity Cmd-3, Lab Cmd-4, the order of the sidebar (the spec had Activity on Cmd-2, before Watch and the Lab were pages; the Lab goes in PR-D). File: Make Planet..., Close World, Reveal in Finder (Shift-Cmd-R). Settings (Cmd-,) is General only: the alife folder and the engine (set up). The *World* and *Help* menus, "Show advanced settings by default" and "Show Welcome" of §2.4 are PR-F/G/D; a toggle that does nothing is not shown.
- **Hoist:** `WorldSession.shared` (made with the shell, before any page) owns the `WorldModel`; `ShellModel` owns the page, remembers it in `shell.page` (never Watch), sets `watchActive` from it, brings Watch forward when a world goes **live and the player started it** on the Worlds page (Play, Make planet) **or `--world` did** (during `starting` the Worlds page shows the progress and Cancel, as the launcher's did; a world is never pulled in front of someone who went on to another page), and handles `--world` once. **A world that does not open is told on the Worlds page** (`StartFailure`: a sentence in plain words from `WorldsGuide.explainFailure`, the engine's message under it, *Show Activity*, *Dismiss*), whether it failed at once or the engine stopped while it started; a world that was live and then died is told by its Watch page. A `--world` that cannot start says why on the Worlds page. `RunController.openWorldAtLaunch` gained a defaulted `play:` parameter (tests).
- **Gate:** `WorldModel.watchActive` (true in the two old modes, always), `keysEnabled` (= `watchActive`: a key event reaches the monitor only for the key window, so the spec's `&& windowIsKey` adds nothing and would change the old modes' tests), `viewerIsLooking` (= `watchActive && windowIsKey()`, read by the attention rules: the dock badge and bounce), `setWatchActive`. The app becoming active clears the badge only with Watch shown. **The in-app cue:** `NSApp.requestUserAttention` does nothing for an active app (Apple's documentation: it returns 0 and has no effect while the app is active), and the app is active whenever the player is on another page, so a chapter or alert stop is shown by the sidebar's Watch row (`WatchCue`: "chapter 2 is waiting", "stopped for an alert", "2 unread alerts", with an orange dot); the Dock bounce stays for an app that is not active. The Metal views (`WorldMTKView`, `SphereMTKView`) take no focus, defer draws, stop their display link and are `isHidden` while `!watchActive`; both are used unchanged by the old modes (`watchActive` stays true).
- **Pages:** *Worlds* has the next step first (`WorldsGuide`: choose the folder, set up the engine, "start with a research run" in the Lab when there is nothing to make a planet from, make the first planet, Play the latest world, a world is open) and then, once there is a world or a run to start from (with neither, the card is the whole page: *Open the Lab* sets the Lab's form to a new `tiny3s` run of 30,000 steps (the §3.2 "Small planet" recipe, about two minutes) and the page offers the planet as soon as the run has a save point, looking again every 3 s while a run grows), *Play* (rows with a reason beside every grey Play), *Make a planet* (the launcher's form with §2.5's words: Start from, Save point, Weather warm-up, "How family lines are rebuilt" and "This world takes powers (director.enabled)" under a remembered *Advanced*; the command is the launcher's, `PlanetSpec` untouched) and *Unfinished*. **A planet can always be made from the app, but still only from an existing research run** (F1 is PR-D/G): with none the card says so and opens the Lab ready; PR-D's pipeline is not built. *Watch* is the World window's content (`WorldDockLayout`, shared with `WorldView`), toolbar Dock / **Overview** / **Close World** (the old "Home" and "Close world" are unchanged in the old modes); with no world open it is a card with *Go to Worlds* and **no control center**; after a world ended or failed a *Back to Worlds* button. The on-map hint still says "H home" (PR-F renames it with the World menu). `NameAuditTests` (new) checks the words of the new pages: the table of labels and actions, and the retired words (Resume, Checkpoint, Fork, Home, spin-up, director.enabled, "another window", branch) in their strings; PR-D extends it to the set-up page. *Activity* lists what runs with Stop / Force stop / Cancel / Close World (`ActivityGuide`) and the global log, or says what to do. *Lab* is today's `ContentView` (its words, Resume, Fork, Repository, are unchanged until PR-D; the note at its top says what the page is for); "Open World" reads "Go to Worlds" there; "Open 3D viewer" still opens the 3D Viewer window (a second window, until PR-F) which closes with the main window. The Lab's grey buttons keep their tooltips as reasons.
- **Reopening and closing:** `AppDelegate.applicationShouldHandleReopen` calls the `openWindow` captured by the shell (spike S3) and returns false when the main window is not showing, so a Dock click brings it back whatever kept the app alive. It looks at the main window itself, not at AppKit's `hasVisibleWindows`: found by running the final bundle, the quit question's alert leaves a glass-effect window on this macOS that counts as visible after Cancel, and the reopen event then did nothing. **Chosen for the Settings window left open:** closing the main window is closing the app's last window: `mainWindowClosed` closes the other windows (Settings, the 3D Viewer), AppKit then ends the app through the question and the wait of PR-A. D3's ask-on-close itself is PR-E.
- **Preferences:** `TypedPreferenceDomain` (`object(forKey:)`, `set(_:forKey:)`; `UserDefaults` and the tests' `MemoryDomain`), `AppPreferenceKeys` (the 14 `world.*` keys), `PreferenceMigration` as in §6, run once at launch **only for a bundle** (an unbundled `swift run` keeps its own preferences, as it always did) and **not at all under `ALIFE_PREFS_SUITE`** (a trial would mark the real domain as migrated with the trial's values). It writes only the app's own domain. **The folder** (`RepoPathStore`): the preview reads its own choice, then the shared domain, then the Lab's, and **writes only its own domain**; the shared domain `io.github.magicbundle.alife` is never written by the preview (the old apps still share and write it). A folder chosen in the preview is not seen by the old apps.
- **Tests:** `LaunchModeTests` line 44 and the `windowsAtLaunch` switch are the only edits to pinned tests; `AppWindow` is unchanged (the main window id is `AppShell.mainWindowID`). New: `AlifeModeTests`, `ShellPageTests`, `PreferenceMigrationTests`, `WorldsGuideTests`, `ActivityGuideTests`, `ShellTests` (delivery before any page, the key gate, the canvas, attention, the page following the world), `ShellViewTests` (every page, light and dark, five folder states).

### 7.2 As built in PR-C (the library; what differs from the row above)
- **`WorldLibrary`** (app level, made in `AlifeApp` beside `ShellModel`, read by the sidebar, the Worlds page and the File menu) merges the catalog of played worlds and the run list into one list of rows with a kind (Played or Research), a shape, a state, the last-saved time and the nesting (§5.3). The scan is `LibraryScanner` in `AlifeCore`: it runs on a serial utility queue, caches each directory by six modification stamps (so a repeat scan stats and reads nothing; 4.8 ms for 200 directories against 45 ms cold), counts links to an unmounted drive instead of listing them ("N worlds are on the drive "X", which is not connected"), and leaves the held `play.lock` out of the cache (a probe every 4 s while the app is active). `LibraryState.catalog` is the launcher's `PlayCatalog` of the same folder (a test compares the two on real directories; they differ in exactly the four things listed in DESIGN §33.4.12: a link to a folder that exists is listed, `modified` is the last-saved date, a world written by `alife resume` counts as locked, and unreadable entries' names are taken); the launcher sheet of the old modes still scans for itself, and the command-line builders are untouched. **Not as §5.3 words it:** the watch on `runs/` sees worlds made or removed, not save points written inside a world, so a 4 s probe and a 20 s rescan (only while this app grows a run) cover those; and `RunController.refreshRuns` (a main-thread listing with a lock probe per directory every 2 s, which the Lab's pickers read) now runs only while the Lab page is shown (`pollsRunList`; the old modes keep it on).
- **States** are §5.2's seven plus **Starting** (a world being opened): Saved, Live, Paused, Starting, Growing, Saving, In use elsewhere, Unfinished, each with a sentence (tooltip and screen reader). "In use elsewhere" for a world another process writes (research or played: `alife resume` runs on a played directory and takes no `play.lock`, seen with the real engine) is the spec's heuristic made safe: `status` `running` and a sign of life (`metrics.jsonl`, `checkpoints/`, `meta.json`, the directory) within `max(20 s, 0.75 x metrics_every)`, so a slow run, a long save or a resume still loading is not shown as Saved; "cut off" needs ten windows of silence. "Last saved" is the newest of the directory's date and `checkpoints/`'s date (the directory's alone misses a new save point). A research world with no save point is not a world and is counted in one line under the list, except the run this app grows, which is listed from its first moment.
- **Sidebar:** under the four pages a section *Your worlds*: the first eight worlds (the open one, what grows, what is in use, then the newest), a research world nested under the played world made from it (its `-grow` seed first, else the newest maker; opened when it needs the eye), then *All N worlds...* and *Unfinished (n)*. Rows show the state under the name and a dot when it needs the eye. **Worlds page:** the next-step card (the most recent playable world first), a search, the filters All, Played, Research, Unfinished with counts, and cards (state chip, what it is, when saved, Play or Watch, *Make planet...*, Reveal in Finder, the reason under the card for every grey button). *Make planet...* opens the planet form in a sheet on that world's newest save point. Unfinished builds show why they stopped and where, the world they were made from, Reveal and **Make again with a new name**, which fills the form (this session's own request, else the source and save point of `meta.json`) and starts nothing.
- **Review fixes (§33.4.12):** a played world is not playable while the Lab or another process writes it, and the Lab refuses to continue it (R6 held from this app); every look at an entry has a deadline (a hung volume costs the deadline once); the watch on `runs/` follows a replaced or late `runs/`; a node of the sidebar is ranked by its best row (the order rule is stated); a world that finished saving is read again; File > Reveal follows the page in view; the row chosen last in the sidebar stays chosen (arrow keys through the whole list, seen in the running app); names taken include unreadable entries and capitals; the planet sheet speaks in the page's words.
- **Not built (later PRs):** a Details page (selecting a world highlights its card; Watch for a research world is PR-F), Keep growing and Branch (PR-D), a thumbnail on the card (no snapshot file exists), "Play on world B while A is live" as a sheet (PR-E: Play stays grey with "Close A first").
- **Tests:** `LibraryScannerTests`, `LibraryStateTests`, `LibraryBenchTests`, `WorldLibraryTests`, `WorldLauncherLibraryTests`, new tests in `ShellTests` and `ShellViewTests`; `NameAuditTests` and `WorldsGuideTests` extended (the Play reason for a world in use is now "In use elsewhere"). No pinned test is edited.

### 7.3 As built in PR-D (the set-up page and the pipeline; what differs from the row above)
- **The page** (`SetupPage`, `SetupModel`) replaces the Lab and the launcher's planet sheet in the `alife` window (the old modes keep both, unchanged). It is the sidebar's fourth row, **New world** (Cmd-4; File > New World... Cmd-N; the Worlds page's *New world* button), and it opens on an intent: *Create* (Name, Starting point with the recipes then the other presets, Goal, Grow for, one button), *Branch* and *Keep growing* (from a card's buttons; Branch from a Played world too), *Make planet* (the old form's Start from, Save point, weather and warm-up, Name; from a card, File > Make Planet..., an Unfinished build's *Make again with a new name*, or a parked pipeline). The button's words are the intent's (§2.5): **Create, Branch, Keep growing, Make planet**; the cards' buttons that open the page end in an ellipsis (**Branch..., Keep growing..., Make planet...**), as *Make planet...* did. Cmd-R presses it; the Stop of the bar answers Cmd-period. **"Open in window"** (toolbar) opens the same page in `Window("Set up", id: "setup")`, one instance, over the same models; closing the main window closes it (§6). The form **is** `RunController.settings`, as the Lab's was, and the second stage's form is the launcher's `PlanetSpec`: `RunSettings.arguments`, `PlayLaunch.arguments` and the golden tests are untouched, and a test pins that the command shown is the command run.
- **Default path.** Name (a free one is suggested: `planet-1`, `planet-2`; a taken one is told with *Use planet-2*), Starting point (a sentence; *About this starting point* holds the preset card and the Compare sheet), Goal, Grow for (a number of steps and "About 1 min 50 s, then about 10 s to make the planet"; under 5,000 steps "Very short: the world will be mostly unevolved"), and **Advanced**, remembered: Seed, Lighter brains, *Make a world that takes no powers*, how family lines are rebuilt, Settings (the ten dynamics levers, in plain names, the engine's key as the tooltip), World shape and gravity (3D and planet levers; a branch keeps its source's shape), Add-ons (the feature layers' boxes as they are), Autopilot (`DirectorSettingsBox`, titled Autopilot here), *Stop automatically after*. A flat preset allows only the research goal and says why; a planet whose shape the weather is not calibrated for is told **before** anything grows. **Show command** has one block per stage ("Stage 1", "Stage 2") and *Copy command* joins them with `&&`.
- **Changes and remembered values (§3.5).** A *Changes* box under the Goal lists exactly the `--set` keys the command will pass ("1 setting changed: Gravity 0.03 to 0.09"), by plain name, with what they were, plus the other choices (seed, lighter brains, add-ons, autopilot, stop-after, no powers). Lever edits of the last Create or Branch are **remembered** (`RememberedLevers`, preferences `setup.remembered.v1`) and come back in the next form: an orange line "Remembered from last time: 1 setting (Gravity) [Reset all]", a *remembered from last time* chip on each row, the Changes box, and the command; a key that does not fit the world (a planet radius in a flat world, any shape in a branch) is **not applied and is said** ("Not used here..."), and stays remembered. A form started with no edit forgets them. Plain messages (`SetupMessages`) cover mutation, immigration, seasons, the 3D basics, name, seed and steps, with the key left for the tooltip; any other key keeps the engine's text.
- **Pipeline** (`PipelinePlan`, a pure state machine in `AlifeCore`; `PipelineCoordinator`, its glue). The table of §3.2 as built, tested state by state with real processes and a stand-in engine (`PipelineTests`): *Growing* (Stop is one SIGINT; the exit parks the pipeline, never continues; a Stop in the engine's first seconds ends on the signal itself and is a Stop only if a save point exists), *1 to 2* (stage 2 starts by itself only when `windowIsLooking` (the main or set-up window shown, not minimised, not occluded, the app active) and the app is not quitting, no other world is open, the name is **still** free on disk, case-insensitively, and the engine is there; else it parks with the reason and bounces once), *Making* (the planet's own progress lines in words, a fraction, the time left; Cancel is one interrupt now), *Live* (Watch comes forward only if the person was on the set-up or Worlds page, as for Play; a person who went to Activity gets the bar's *Watch*), failures (the run's last error line, or a sentence; the grown world is kept) and the parked states, each with buttons, always with *Done*. **The bar** (`PipelineStrip`) is on every page but the set-up page, which shows the same thing in full (`PipelineCard`: title, what is happening, a bar, the clock, the three steps with the current one, what comes next, the buttons). Nothing is resumed at a launch: a pipeline lives in memory, and **what the folder shows is honest by itself**: a seed (`NAME-grow`, planet-shaped, with a save point) whose `NAME` does not exist says "Planet not made yet" in the sidebar and on its card, with *Make planet...* offered (`LibraryRow.unmadePlanet`), and a run stopped before its target says "Stopped at step 12,000 of 30,000." (`run.steps` of `config.json`, new in the scanner).
- **One click is one world** (`LiveFacts.pending`): while the seed grows or the planet is made, the library lists the planet `NAME` (Played, Planet, *Growing* then *Starting*, "Growing first, then it becomes a planet.") with the seed nested under it, so the sidebar and the Worlds page show one world; the placeholder gives way to the real world with no gap and hides the half-built folder from *Unfinished* while it is being made.
- **`DirectoryGuard` (R6)** (`AlifeCore`, pure; fed by the coordinator at click time and again at start): the writers of this app (the open world, worlds saving, the pipeline's directories and the promised planet) and what other processes hold (`In use elsewhere`), compared as the file system compares (capitals and composed or decomposed letters do not matter); a foreign run **is refused** like a foreign played world (as built; there is no warn-only path). `SetupRules` is §5.2's matrix for Keep growing, Branch and Make planet, with the reason for each grey button and a note when the source still grows.
- **The Lab and the launcher are gone from the window.** `ShellPage.lab` became `.setup`; `LabPage`, `PlanetSheet` and `PlanetForm` are deleted (the Lab's `ContentView` and the launcher's view stay for the two old modes). **§3.8 line by line, what moved and what has no home yet:** rows 1-3, 4-13, 15-17, 22-24, 26-37 have their homes as the table says. Row 14: the Director box is Advanced > Autopilot; its *live* panel (interventions, journal) is on **Activity while a run grows** until the Run panel (PR-F). Row 18: the Status numbers, every one of them (step, alive, speed, family lines, deepest generation, energy check, height, on the floor, at the surface, the feature groups), are `RunNumbersView` on Activity **while a run grows**. Row 19: *Open viewer* is the card's menu item **Open Classic 2D Viewer** (flat worlds), *Open 3D viewer* is **Watch in 3D viewer** (it opens the 3D Viewer window, Follow while the world grows, else Re-simulate), *Open World* is Play. Row 20: *Inspect* is **Check world** (the card's menu; the result is in the Activity log), *Reveal* is on every card, the log is Activity. Row 25: the starting box is the bar and the page's card. **Two capabilities have no home until PR-F**: the numbers and the autopilot journal of a run that is *not* growing (the Lab showed the last row of the chosen run when idle; the Details page and the Run panel are PR-F), and Settings > Autopilot (the model's address is set per run, in Advanced). A research world has **no observer Watch** (PR-F): after Create, Branch or Keep growing the person lands on the page's progress card, and a finished research world is a card with *Make planet...*, *Keep growing...*, *Branch...* and the viewers.
- **Other edits:** `RunController` records how a run ended (`lastRunExit`: status, signalled, stop requested, forced, the last error line) and gained by-name viewer and check methods; `QuitWording.confirmDetailShell` says "continue it later with Keep growing" in the `alife` window (the old modes keep their text); `DirectorSettingsBox` takes a title; `ShellModel` brings Watch forward for a world the person started from the set-up page too; `WorldLauncherModel.start(spec:)`; `LibraryScanner` reads `run.steps`.
- **Tests never put anything on a screen.** The real quit panel is placed at -20000 in the tests (`QuitPanel(place:keepsOnScreen:)`), a sheet's host in the quit tests is a window that reports a sheet without AppKit attaching one (a real sheet flew its window to the bottom-left corner of the display), and no titled window is ordered in (AppKit moves it back onto a screen). Proved with `CGWindowListCopyWindowInfo` polled every 100 ms through a full `swift test`: no window of the test process had a visible part inside a display (before the change: the panel and four test windows did).
- **Seen with the real engine (a Branch, final bundle, scripted).** A Branch of a flat research world with *Mutation rate floor* 0.001 to 0.0005 ran `alife run --init-from runs/planet-1 --out runs/planet-1-branch-1 --set mutation.rate_min=0.0005`, grew 20,000 more steps (parent `runs/planet-1:20000`, final 40,000: a blank length is the source's own run length beyond its save point, so the bar knows its target) and ended *planet-1-branch-1 has finished growing*, saved at step 40,000 (the end step is the newest save point, not the last numbers the bar read). The same Branch with 0.004 was **refused by the engine** ("the state violates the new config", exit 4, nothing written): the bar said *Growing failed*, with the engine's reason in plain words, that it stopped before it made a save point, and *Try again with a new name*; no folder was left, and the value stays remembered with a chip and *Reset all*.
- **Review fixes (after the independent review of PR-D).** (H1) A run that ends with the engine's status `extinct` (exit 0, before its target) is read from `meta.json` at its exit: the bar says *The population died out at step N* (a state of its own, `diedOut`), never "finished" and never at the target step, stage 2 is never started from it, and the library row refuses Make planet and Keep growing ("Everything died out in this world"). `alife play` that ends by itself on an extinct world, or for any other reason with exit 0 before its header (it reads as a cut stream), is told the same way or as "The planet's engine ended by itself before the planet was ready", with its last line; only a Cancel is called cancelled. Seen with the real engine: Small planet with *Buoyancy upkeep* 5000 died out at step 28 and the app said so. (M1) *Make again with a new name* ends a parked plan and opens the form; every card button is grey while a world is being made or waits for an answer (`SetupRules.pipelineReason`). (M2) the stop timer is clamped as the Lab's was (0-99 h, 0-59 min) and `timerDuration` clamps too; `TimeWords` and the sums of lengths cannot trap. (M3) "seed" means only the random seed; the run under a planet is "the grown world"; no title, step or bar line shows `NAME-grow` on the default path; exception class names are stripped from messages. (M4) Estimates say "on a recent Mac (measured on an M5 Pro)", the first run after an install is slower (one line), the stage-2 bar has a fraction only for the weather warm-up. (M5) A preset without `run.steps` grows the engine's 100,000, said so. (M6) Branch, Keep growing and Make planet forms survive the page appearing again, in either window; only File > New World, the Worlds page's button or the form's *New world instead* start a Create. (M7) the two safety checks of the coordinator are tested and a run that ends under another name fails the plan in the open. (M8) a foreign run is refused (as the doc now says). Low: remembered edits only after a successful start and only from a Create; houses is shown in Changes; a failed stage 1 that made progress offers Make planet and Keep growing; the bar shows the stop timer's countdown; the bulk "put back" buttons are back; the name message is shown once; `NameFolding` mirrors APFS (capitals and normalisation); a name with a trailing newline is refused. Not done: a real AppKit sheet cannot be tested off screen (a borderless, transparent, far-off window with `constrainFrameRect` overridden is still moved onto the display by `beginSheet`); the stand-in `SheetHost` stays and the real path is for the owner's hands.
- **Tests (new):** `PipelinePlanTests`, `DirectoryGuardTests`, `SetupPlanTests`, `SetupSettingsTests`, `LibraryPipelineTests` (Core); `PipelineTests` (a stand-in engine, real processes; Stop, Cancel, failure, window away, quit through the real gate, a second launch, two writers, names, one click one world, remembered values), `SetupModelTests`, `SetupViewTests` (every state of the page and the pipeline, light and dark, 1,100 x 700) (App); `NameAuditTests` extended over everything new, and the retired words now include "the Lab", "Mode:" and "Director". Edited: the tests that named `.lab` or the Lab's words (`AlifeModeTests`, `RunListPollingTests`, `ShellTests`, `ShellViewTests`, `WorldLibraryTests`, `LibraryLiveTests`, `WorldsGuideTests`, `LibrarySafetyTests`, `LibraryStateTests`, `ActivityGuideTests`).

### 7.4 As built in PR-E (the safety rules in the shell; what differs from the row above)
- **R3 and D3 = C, closing the window.** `WindowCloseObserver` puts a forwarding delegate (`CloseGuardDelegate`) in front of the main window's own, so that the red button, Command-W and `performClose` all reach `AppDelegate.mainWindowShouldClose` before the window closes (seen with the real bundle: the window's delegate is the guard, and `performClose` and a click on the red button ask). With a run going (any run job: the first stage of a planet counts) it asks: **Keep Running** (the default, Return), **Stop and Quit**, **Cancel** (Escape), and a checkbox "Remember Keep Running (change it in Settings)". The title names the run ("The world for 'planet-1' is still growing."). *Keep Running* closes the window, **saves and closes the open world named in that same sentence** (a world still starting is stopped, and the sentence says nothing of it is saved), and lets the run carry on; the text says the Dock icon shows its progress and bounces at its end, that a click on it brings the window back, and that **alife stays open until you quit it**. *Stop and Quit* asks the run to write a save point first (up to 120 s) and says to continue it later "with Keep growing, on its card in Worlds": the window **stays up** while the quit goes through PR-A's gate and panel, and PR-A's question is not asked again. *Cancel* changes nothing, the window included. **Only Keep Running can be remembered** (`shell.closeWithRun`: ask or keep): a remembered Stop and Quit would let one Command-W end a run of hours, so the box stores nothing for it and a stored "stop" (an earlier build) is read as the question. Settings, *While you are away*, shows the remembered answer with its effect line and *Ask me again*. **The facts are read again when the question returns:** a run that ended or began to stop while the person read it leaves nothing to keep (the window closes and the app ends with it; it never stays open with nothing running), and a planet that began meanwhile is asked about on its own, so Keep Running can never interrupt a planet being built while saying it was saved. *A planet being made* (stage 2, no run job) or a world being opened asks too, with two buttons: **Cancel** (default, Escape) and **Stop and Quit**, "Stop making 'X' and quit?" (it needs the window; nothing is deleted, the build stays under Unfinished); nothing is remembered. Closing with only a *live* world open still asks nothing: it is saved by the quit's gate and panel (an open question for the owner, HANDOVER). No question either when the run is already stopping. The old modes are unchanged (`LaunchPlan.asksWhenClosingWithRun`, `alife` only; `quitsAfterLastWindow` and its pinned tests untouched). **Every pipeline state against closing:** growing (a run: asks); 1 to 2 (instant; the question is taken again if it happens while the question is up); making (asks, above); live (a world: no question); parked, failed, done (nothing runs: nothing asked).
- **A windowless app.** The Dock tile shows the run's progress (`DockTileController`, a bar and the percentage on the icon, from the run's own target: the pipeline's, else `run.steps` of its `config.json` added to the `start_step` of its `meta.json`; unknown means "Growing", never a made-up number; the numbers of the run before are cleared when a run starts, so the bar never shows an old step over a new target). When the run ends and it was not the person's own Stop, the icon bounces once if the app is not in front (the system ignores a bounce for an app that is) and carries a badge, **Done** or **Failed**, until the window is shown again (a person who is looking at the window gets no badge). The stop-after timer's stop counts as news (`RunExit.byTimer`). A Dock click, `open` on the running app, and any menu command that names a page bring the window back (`applicationShouldHandleReopen` and `showMainWindowIfNeeded`, S3), **a minimised window is taken out of the Dock** (`deminiaturize`) and no second one is asked for. The window comes back **on the page the player left**, except that Watch, which has no world to show once its world was closed, becomes Worlds; a note over whatever page that is says what became of the world that was open: "is being saved and closed" until its process has gone, then "was saved and closed", or "did not finish saving (why)" when it ended badly (`PlayController.lastRetiredEnd`: the exit status of a world that was asked to stop is recorded, a signal that killed it or a non-zero exit is a failed save), or "was still starting, so it was stopped before it opened". A pipeline that finished while no window was shown is parked at "finished growing ... not started behind your back" with Make planet, Keep growing, Done. **Stage 2 never starts invisibly and a windowless app cannot start it, by any route:** `windowIsLooking` needs a visible, unoccluded main or set-up window and an active app and the quit does not count, and the switching sheet's completion needs the main window shown too. Quit from the Dock with the window gone and a run going asks PR-A's question; Cancel keeps the app and the run.
- **R4 and D4 = A, leaving Watch.** `AwayPause` (pure) decides, `WorldModel+Away` reads the clock and sends. Selecting another page pauses a world that is **running**; coming back runs it again when *Pick up where I left off* is on (Settings; the default). Only the app's own pause is ever undone: a world the player paused, or stopped at a chapter, an alert, its end or an error, is left as it is on leaving and on return (`run` would acknowledge the chapter card). A world about to stop by itself (its chapter or alert clock is announcing while the save point is written) is not paused under it; a chapter that takes the world before it read the app's pause makes the app give the pause back. A world that goes live, or starts running, while the player is on another page is paused at its first clock. Deactivating the app (Command-Tab) calls none of this. **Honest state:** the sidebar's Watch row says "planet-2 · paused: you left", the Worlds page's open-world card and Activity say it too (and, with the setting off, that it stays paused until Run), and with the setting off Watch shows "Paused when you left this page. [Run]" and the clock says "paused when you left Watch". The Dock badge and bounce keep their meaning (`viewerIsLooking`). **Found by drawing the page:** a Metal view that attaches again re-delivers everything to the model and reset the pause, which would have left the world paused for ever; the hold is cleared by a different world or none (`play.onReset`, a clock of another world), not by that reset (a test pins it).
- **R5, the switching sheet.** Play on world B, or Make planet (the set-up page, or the button of a parked pipeline), while a world is **live**: one sheet, "Stop 'A' to play 'B'?", "Only one world can be played at a time.", "'A' is saved first, which takes about N seconds. You can play it again from Worlds whenever you like." ("Nothing is lost" was dropped: it is only true once the save has ended well.) Buttons **Stop and Play** (or *Stop and Make Planet*) and **Keep Playing**, the safe one, the default and Escape. N is an estimate from the size of the newest save point (3 s plus 0.01 s per MB, 0.05 s per MB for the compressed form, up to a multiple of five, between 5 and 40 s: DESIGN §33.4.4 and §34.11's measured times); **a world with no save point says "usually a few seconds; it has no save point yet, so the first one may take longer"** and claims no number. On confirming, A is stopped through its ladder and what was asked for starts only when A's process has **gone**, saved well, and the main window is shown, and never while the app quits: the sheet shows "Saving 'A'..." with the seconds waited and **Close Panel** (it closes the panel and nothing else; a click on Play or Make planet while the save goes on brings the panel back instead of being swallowed). If A's process ended badly (a signal, a non-zero exit) or is still being written after the longest wait (60 s; the ladder ends it by 40 s) **the other world is not started**: the sheet says what happened ("'A' did not finish saving: it exited with code 1: disk full. Its last save point is still there. 'B' was not opened.") with **OK**, and it comes back even if it had been closed. If the window was closed meanwhile (Keep Running), the action is dropped, A stays closed and the Worlds page says "The window was closed while 'A' was being saved, so 'B' was not opened. 'A' was saved and closed." A world that is still **starting** is not stopped by a sheet: Play and Make planet are grey with "'A' is still starting", and a parked pipeline's *Make planet* says so in its bar ("... is still starting. Wait until it is open, or cancel it on the Worlds page; then press Make planet again.") instead of doing nothing. The automatic second stage of a pipeline still never replaces a world: it parks ("Make planet asks to stop 'A' first; it is saved"), and the button asks. `WorldsGuide.reason(forPlaying:)`, `reasonToMakePlanet` and `SetupRules.reason(.makePlanet)` no longer answer "Close A first" for another live world (the tests that pinned it are edited); Make planet *from the open world itself* is still "Close it first" (its save point is being written).
- **R1.** One played world, one run job, one observer stream stand as in the rules; the second Watch cannot exist, and a second world is reached only through the sheet. While a run and a world are both going Watch, Activity and the Worlds page's open-world card say "A run is using the GPU; the world may be slower."
- **Differences from the plan.** (1) No Dock progress for the planet's second stage (no run job); its end is the planet opening. (2) Cancelling the quit question while windowless leaves the window closed (a Dock click brings it back). (3) `ax` access to SwiftUI buttons was not available to the scripted runs, so the sheet's buttons were pressed by posted clicks. (4) The question and the sheet are not shown on a lit display here; see "Not seen". (5) PR-A's old note (i) ("closing the last Lab window while a run goes and answering Cancel leaves the app with no window") is gone for `alife`: the question comes before the window closes. (6) **Known and left (review L2):** if SwiftUI ever put its own delegate back between two view updates, or a window were closed by code (`window.close()`), the guard would be bypassed; the outcome is bounded: the window closes, `keepsAliveForRun` is false and AppKit goes through the quit and PR-A's question (Cancel keeps the run); the observer puts the guard back at every update. (7) The Watch page's two notes (paused when you left, GPU) sit at the top of the map and were not seen over the HUD on a lit display (HANDOVER).
- **Seen with the real bundle and the real engine** (ad-hoc `alife Preview.app`, a private alife folder, a private preferences suite, scripted by a driver loaded into the process; the session's display was asleep or locked, the app never became active, `occlusionState` said not visible): a Small-planet Create (6,000 steps), close with `performClose` (and the red button): the question appeared; **Keep Running**: no window, the run went on, the Dock tile held a `DockProgressView` (98% at the last look), the run ended with nobody looking and the icon had **Done** and no progress; a reopen event brought the window back on the same page with the bar "planet-3 has finished growing: not started behind your back" and nothing running; Quit then ended the app. **Stop and Quit** at the close question: the run stopped at a save point (step 5,832, status interrupted), the app ended 3 s later, no process left. **Cancel**: the window and the run stayed. Dock Quit with no window and a run going: the PR-A question, Cancel kept both, a later Stop and Quit ended both. A played world (planet-2 from `--world`): Space paused it (the picture was the same after leaving and returning, hash equal); running, leaving to Activity: the sidebar said "paused: you left", Activity said "Paused while you are away from Watch. It continues when you go back.", the Worlds card said Paused; back on Watch: "live", 1.00x of 1x, the step advancing. With the setting off: the note and "paused when you left Watch". Switching planet-2 to planet-1 through the sheet: planet-2's save point (step 31,884) was written at the moment of the click and planet-1 went live in Watch, planet-2 listed "Saved just now"; the world's save is only a few MB here, so the "Saving" state was too short to see (the unit tests run it with a slow engine). A real world that had reached its chapter end (step 36,000, the "Chapter 1 complete" card) was left for Activity and returned to: still stopped at step 36,000 with the card, nothing run. Quit in the states above, and with the switching sheet up, left no process (`ps`) and every world saved.
- **Seen again after the independent review's fixes** (final bundle, same method, display asleep): a run that ended while the close question was up, answered Keep Running: the window closed and **the app ended** (it did not stay open with nothing running); closing while the planet was being made (made by hand from the parked bar, since the unattended app is never "active") showed "Stop making 'planet-9' and quit?" with Cancel and Stop and Quit only: Cancel kept the window and the planet went on to open; Stop and Quit left `planet-10` as an unfinished build (`meta.json` interrupted), the app ended and no process was left; Stop and Play, the panel closed with Escape, then the window closed: planet-10 opened, the window closed with no question (a live world only) and the quit saved it (step 6,122); Space and, at once, leaving Watch, before the clock could say so: back on Watch it stayed paused (the picture was identical 4 s later); a minimised window and a reopen event: the window came out of the Dock (`isMiniaturized` 1 to 0, visible). **Not drivable with a real engine here:** a window closed while a hidden switch is still saving (a small world saves in under a second, and the cards' Play buttons are grey while a run grows, so the sequence cannot even be made by hand; the guard is tested with a slow stand-in engine); Command-W; the sheet's "Saving" and failure states with a real world.
- **Not seen** (and so the owner's): the question and the sheet drawn on a lit display (the scripted run's windows were not on screen; the pictures are the same views drawn off screen in `app_e_shots`), Command-W (the menu item was disabled because the window was not key; `performClose` and the red button's own action were exercised), a real Dock click and the Dock icon itself (the tile's view was read from the process and drawn to a PNG), the bounce, a world with a 686 MB save, the race in which a chapter takes a world just before it reads the app's pause (a test only), VoiceOver.
- **Tests.** `SessionRulesTests`, `AwayPauseTests`, `WorldSwitchTests`, `DockProgressTests` (Core); `SessionTests` (the close decision for every combination of run, world, remembered answer and mode, the guard on a real window, reopen, every pipeline state against closing and quitting, leaving and returning for every state of the world's clock), `SwitchAndDockTests` (the sheet with a slow stand-in engine and the pipeline's buttons, the Dock with a recording surface, the tile's drawing), `SessionViewTests` (the dialogs and states drawn, light and dark). **Mutation test after the review** (32 more hand-made mutants of what the review changed: the second look at the facts, the remembered answer, the planet question, the switch sheet's window, failure and timeout, the retired world's end and its note, the Dock's target and stale numbers, the minimised window, the page the player comes back to, plus the review's own P17 `requestedPause` mutant, which the adopted test kills): 31 killed after the tests were completed (eight first survivors were killed by new tests), one survives and is a guard, not a behaviour (`m1_endless`: the question is taken again at most twice; the situation can change only a few times, so the cap is never reached). **Mutation test** (86 hand-made mutants of the close decision, the remembered answer, the pause and resume logic, the sheet, the Dock and the guard, each applied to a copy and run against the suites above): 82 killed, 4 survive and are equivalent (`estimate_no_floor`: the rounded estimate is never below five, so the floor changes nothing; `switch_keep_runs`: the next request overwrites the held action; `switch_confirm_twice`: a second confirm only starts a redundant waiting task, the action having been taken by the first; `guard_not_reinstalled`: one observer per window, so "a guard" and "my guard" are the same). Seven first survivors were killed by new tests (the viewer's stream, a remembered Keep Running on a fresh delegate, a world no longer open, a pause that could not be written, the clock's words while away, a form pressed behind the sheet, the stop-after timer's exit); the first pass of the app mutants was void (the copy had no `tests/data`) and was run again. Edited: `WorldsGuideTests`, `LibraryStateTests`, `DirectoryGuardTests`, `PipelinePlanTests`, `PipelineTests` (the sentences that said "Close A first"), `NameAuditTests` (new files and labels).

**Headless tests.** Pure logic in `AlifeCoreTests` (migration precedence and idempotence, `SessionRules`, `DirectoryGuard`, `PipelinePlan` state table, library merge, `NameAuditTests`, key-table-versus-menu
hints); view construction through `NSHostingView` as `PresetGuideViewTests` does (ran here in 1.1 s); the key gate and delivery-closure tests of §2.4; `check-apps.sh` and `tests/test_app_bundles.py`.
**Golden tests untouched through PR-G:** `ArgvIdentityTests` and `ArgvCases`, `PlayLaunchTests`, `PlayCommandTests`, `World3DLeverTests`, `SphereLeverTests`, `FeatureLeverTests`, `PresetGuideTests`,
`StreamDecoderTests`, `AppTerminationTests` (extended only by new tests), `tests/test_gui_sync.py`; the opt-in GPU tests are unaffected. **WorldRendererOwnershipTests** hosts `WorldView(play:)` directly and
tests that the renderer is retained, not page switching: it needs rewriting if `WorldView` is split (PR-B).
**Only the owner's hands can verify:** spikes S1-S3 outcomes on screen, Metal view identity feel, window and sidebar behaviour, Dock behaviour, Cmd-Q with a 233 MB save and with a run, time to first picture of
recipes, TCC prompts, icon, 13-inch fit, the chapter rhythm after R4. Each PR appends a click-through list to HANDOVER.md (base: HANDOVER.md:758-765).

---

## 8. Decisions for the owner (six)

**D0. Who is the app for?** (a) The owner and collaborators who have a checkout; (b) real newcomers, with no checkout and no engine knowledge; (c) (a) now, built so (b) is a later layer.
*What it changes:* with (a), Welcome is a one-card checklist, one recipe (Small planet), plain-language work only on the default path (about a fifth of PR-D/G); with (b) the full Welcome, three recipes,
copy for most of Advanced, D2 becomes unavoidable, and about a third of the program is for that person. *Costs:* (a) is the cheapest and fastest, and matches who can use it today (the engine needs a checkout and `uv`), but
leaves the brief's "newcomers who do not know the engine" served only by the default path; (b) serves that brief fully but spends a third of the work on people the app cannot reach until D2 is solved, and the
first draft's promise of "no terminal" is still false on a fresh Mac. **Recommendation: (c).** Build the default path to newcomer standard, keep Welcome minimal, decide (b) after the owner has used the preview.

**D1. A newcomer's first world.** (A) A planet straight from a preset with `alife play --preset` (GUI-only; fastest; no evolved organisms; quality and warm-up unverified [U]); (B) grow a small preset (about 80 s for
`tiny3s` at 30,000 steps; 992 organisms in the preset guide's measurement) and make a planet automatically (verified code paths; watchable wait; whether 992 organisms is satisfying is [U]); (C) ship or download a pre-evolved
starter world (best experience; content, hosting, maybe an engine-side crop tool: out of this program). Also: what is the smallest world they would call worth playing (sets "Small planet"'s numbers).
*Costs:* A is quick but may show a dull world; B is slower (about two minutes plus warm-up) but shows life evolving and uses only verified paths; C costs content work and hosting. **Recommendation: B first, A tried
by the owner on their Mac and made the default only if watchable; C later.**

**D2. Where the engine comes from.** (A) Guided use of an existing checkout (choose, set up with `uv`); (B) the app clones the repository and runs `uv sync` (needs the URL, git, network); (C) embed the engine in a
signed, notarized app (HANDOVER.md:756 calls it a separate project); (A') the app runs the official `uv` installer script (executes a downloaded script: a security choice). A fresh Mac still needs a terminal or Homebrew for `uv` and `git`
under A and B. *Costs:* A is small and safe but needs a human to hand over a folder; B and A' remove a terminal step but add network and script-execution risk; C is the only no-terminal path and a separate project.
**Recommendation: A in the core plan, with copyable instructions; B only if the repository is meant to be public.**

**D3. Does closing the window stop a long run?** Provenance: quit-on-last-window is recorded as the builder's "single-window-game convention" for Storm Worlds (DESIGN.md:15100, row I59; :10924) and as "unchanged"
for the Lab (DESIGN.md:10840); it was not an owner decision (the request on 2026-10-23 was "launch without the terminal"). Options: (A) the app stays alive while a run exists; (B) today: closing the last window quits and stops the run
(a checkpoint is written, so it is lost time, not lost data); (C) closing with a run asks "Keep it running / Stop and quit / Cancel" (remembered if wished), and shows run progress in the Dock tile.
*Costs:* A survives mistakes but leaves a windowless app in the Dock, no completion signal (a notification would need a permission prompt), an invisible keep-awake, and a Quit that now waits and asks; B is the
simplest and has no zombie app but surprises whoever closes a window; C is the usual macOS pattern and keeps today's default, but adds a dialog and still produces the windowless state when "Keep running" is chosen.
**Recommendation: C**, with the Dock-tile progress and a Dock bounce at the end; fall back to B if spike S3 shows a closed `Window` cannot be reopened reliably.

**D4. Pause when leaving the Watch page?** (A) Yes, on selecting another page in the app, **not** on app deactivation (Cmd-Tab), so the Dock badge and bounce for alerts keep their meaning
(WorldModel+Decide.swift:213-234: attention is "window key and app active"; I checked that A is consistent with it, and that the predicate must also require `watchActive`, otherwise another page counts as
"looking"); (B) keep running, with a Dock badge and a sidebar-row badge for alerts, relying on the world's own chapter and alert pauses; (C) ask each time. *Costs:* A keeps "observe, decide, watch" but changes pacing for
anyone who wants a world to run while they read a report elsewhere; B keeps today's pacing but a world can advance through a storm unseen, and a hidden key monitor is a hazard (fixed by §2.4); C is safe and tiresome.
**Recommendation: A, with the Settings toggle** (*Pick up where I left off*).

**D5. Is viewer legibility in this program?** The owner's 2026-10-23/24 first-play notes (too dark, single organisms invisible, repeated tiles confusing, hard to know where you are, abrupt zoom, no sense of space) are
**out of scope today**: this spec changes structure, not rendering; orientation aids (readout, scale bar, compass, minimap) already exist (CORE/WorldReadout.swift:1-12) and an unmerged branch `gui/viewer2-a1` exists
(HANDOVER.md:733). The owner should choose knowingly. (A) Include a legibility pass as its own PRs after PR-F (renderer defaults, a persistent "where am I" readout, smooth zoom); (B) a parallel track, with the
preview used first for a play-through that produces a ranked list; (C) leave it. *Costs:* A answers the owner's most concrete findings and matches "user-friendly", but touches renderers with GPU and golden render tests and
risks delaying the merge; B keeps the merge independent but those findings stay open longer; C ignores them. **Recommendation: B**: the shell does not make legibility worse (the Watch canvas is unchanged), the preview
gives a better list than guesses, and renderer work has its own tests and branch; decide A after that play-through.

---

## 9. Risks and what is left out

### 9.1 Risks
| Risk | Mitigation |
|---|---|
| The Metal view cannot be kept across pages, or the key monitor leaks | spike S1 with pass criteria and two fallbacks (§2.4); the key gate and its test are needed in all variants |
| `.terminateLater` shows no panel | spike S2 passed (§5.4 item 5); fallback kept written |
| `Window` + `@SceneStorage` + Dock reopen unproven | spike S3; `applicationShouldHandleReopen`; D3 falls back to B |
| Pause-on-leave surprises | D4; Settings toggle; chip "Paused (you left)" |
| Quit now waits and may escalate to SIGTERM | confirmation states the loss; saves are atomic (runio.py:253-265) |
| New bundle id re-asks folder, removable volume, Local Network | documented; preview shows it before the final id |
| Two apps or two processes writing one directory | R6 guard; foreign runs only warned [I] |
| Archived `runs/` on an unmounted drive | counted, not shown as broken (§5.3) |
| Plain-language mapping drifts | limited to the default path; keyed by engine key; falls back to the raw message; a test lists the keys |
| The recipes are not good | data, D1; times from the manifest, warm-up [U] |
| Window minimum 1,100 x 700 on small screens | judged by hand; sidebar hides in Watch |
| CI has no GUI job | local `swift test` and `check-apps.sh`, as today |

### 9.2 Deliberately left out
- **Move to Trash** (was D4). Reasons from the review: a world with a headless run, an open stream, a Branch reading it, or another app's play is "unlocked" for `play.lock` (taken only by `alife play`), and moving it makes the
  run fail (atomic temp-and-rename on a vanished path: exit 4) and leaves a partial directory in the Trash; a hard-linked origin keeps its bytes (DESIGN §34.11) and the Trash frees nothing until emptied; a path on the
  external volume goes to that volume's `.Trashes`; a symlinked `runs/NAME` trashes only the link; and it is the program's only destructive host-side action, which a merge does not need. Finder and the storage strip remain.
- **In-app comparison, run queue, "Vary a setting", charts** (was PR9/D5): outside a merge; the Branch workflow of §3.6 remains. A possible later project, with no PR number.
- **Cut from the core plan, no loss to the request:** `UndoManager` for form edits (text fields undo themselves); the Dock menu beyond reopening; Open Recent and Cmd-O; Save Snapshot on Cmd-S (the viewer's Snapshot button
  covers it; the World view has none); plain-language text for every engine key (the default path only); the `alife://` URL scheme; Settings > Play and Settings > Display and "default dock side" (the dock's View menu is the one home);
  toolbar transport (the clock strip is the one home); a second Close World control; "Open World..." (sidebar search); a rate remembered between sessions.
- Embedding the engine, notarization, an updater; several played worlds or several Watch windows; any engine or config change, a starter-world pipeline, a crop tool; renderer, narration, powers or report changes
  (unless D5 = A); permanent deletion; localization; an AppleScript interface; sharing worlds; automatic stop of a long-paused world; VoiceOver work beyond keeping existing labels; renaming the SwiftPM target.

---

## 10. Decisions recorded by this spec (revisitable)

| ID | Decision | Reason |
|---|---|---|
| M1 | Final bundle id `io.github.magicbundle.alife`; preview id `...alife.preview` | the id is already the shared suite; the preview coexists with the old apps |
| M2 | Leave the old apps; print instructions; remove nothing | an installed app may be running a world |
| M3 | Keep the SwiftPM target and directory names | 46 tests and a Python path pin them |
| M4 | One main window, plus a single-instance set-up window | no second Lab window (ContentView.swift:58-62); a form beside a Watch (H7) |
| M5 | Opening the app starts nothing; stage 2 of a pipeline starts only when its Watch is visible | R7 |
| M6 | Kinds Played/Research, shapes separate | H2 |
| M7 | One word per action, tested; "Cast" kept, "Overview" is the camera only; "Resume" removed from the UI | H1 |
| M8 | "Storm Worlds" retired as a name | the owner chose "alife" |
| M9 | Advanced is a remembered disclosure | principle 3 |
| M10 | The log is global (Activity) and also in the Run panel | F18 |
| M11 | Auto-detect only the stored path, `~/artificial-life`, `~/alife` | avoids folder-access prompts |
| M12 | One minimum window size, 1,100 x 700 | no abrupt resize on entering Watch |
| M13 | Bare letters are labels, never menu equivalents | M6 of the review |
| M14 | Display preferences, Pauses and confirmations stay in the dock's View menu | one home each |
| M15 | Keep "Autopilot" for the director; the engine names stay in docs and command lines | DIRECTOR.md is normative; note planets have `PlayDirector.rules` with no UI |
| M16 | Flat world is research-only | weather refuses 2D; a played 2D world is allowed by the launcher but unverified [U] |
| M17 | The set-up page is the sidebar's fourth row, *New world* (Cmd-4, Cmd-N), in the place of the Lab; the Lab page and the planet sheet are removed from the `alife` window in PR-D | the page replaced them; the old modes keep both until PR-H |
| M18 | A planet that Create grows has a seed named `NAME-grow`; nothing is persisted for a pipeline, and the library says "Planet not made yet" for a seed whose planet does not exist | what the folder shows is honest after a relaunch with no state file to go stale; a hand-named `x-grow` gets the same offer, harmlessly |
| M19 | Stage 2 starts by itself only when the window is shown, not occluded, the app is active and not quitting; otherwise the pipeline parks | R7: nothing starts invisibly (the unattended path was shown to park in a locked session) |
| M20 | Remembered lever edits come back in the next form with a chip, a line and an entry in the Changes box, and are never applied silently to a world they do not fit | L4: a hidden edit never travels into a new world |

---

## 11. Verified and not verified
Verified by reading: everything with a `file:line`, including the review's claims I rechecked (key monitors, delivery closures, `tiny3s` steps, lock behaviour, `shutdown()` guard, attention predicate, test pins, counts).
Not verified: any running behaviour of the app (windows, menus, Dock, restoration, quit panel); `NavigationSplitView` with a Metal view (S1); `.terminateLater` drawing (S2); `@SceneStorage` and Dock reopen in a lone `Window` (S3);
the engine end to end (`alife play --preset`, warm-up times, a played flat world); TCC behaviour; `alife --help` as a health check; the 10 s write-activity heuristic; that the hidden canvas's draw loop can be paused. Since PR-A: `.terminateLater` drawing is verified in the running app (S2, §5.4 item 5), with the limits stated there.

---

## 12. Review findings and what was done
Review verdict: "ready after the listed fixes". I rechecked each claim in the code; I found none wrong in substance. Three of the review's own statements needed adjusting (marked **adjusted**).

| Finding | Check | What was done |
|---|---|---|
| C1 hidden Metal view grabs keys; chrome and min size leak | confirmed: WorldMetalView.swift:53-83, SphereMetalView.swift:44, WorldView.swift:53-81 | §2.4: hoist, `watchActive` key gate and attention predicate, chrome by page, spike S1 with criteria and fallbacks F1/F2; delivery closures kept correct (WorldModel.swift:303-308); one 1,100 x 700 minimum (L6) |
| C2 New-world page undefined | confirmed: Steps blank = 500 for tiny3s; chain is two processes | §3.2: page, Goal, no "until I stop", all three recipes, table of what Create produces, pipeline state table, grouping and nesting, two-block Show command; builders untouched |
| H1 words reused | confirmed (also clock strip "Resume": ControlCenterView.swift:180) | §2.5 one word per action, `NameAuditTests`, F5 recounted; "Cast" kept, "Overview" camera only; Continue = chapter card only |
| H2 two kinds, two key tables | confirmed: Viewer3DView.swift:98-153 vs WorldKeys.swift:46-56 | §2.2 kinds and shapes, §3.4 comparison table, per-form key tables and menu; not merged |
| H3 Trash preconditions | confirmed (lock only by `alife play`) | moved out of scope with the review's reasons (§9.2); the library never deletes |
| H4 state x action; two writers | confirmed: cli.py:652-657 vs 1342-1347; PlayController.swift:123-127 | §5.2 R6 `DirectoryGuard`, the matrix, "Branch does not disturb" corrected for Make planet |
| H5 pause-on-leave is an owner decision; deactivation | confirmed: WorldModel+Decide.swift:213-234 | D4; recommendation A holds, with the correction that `windowIsKey` needs `&& watchActive` (one window makes "another page" look like "looking") |
| H6 quit path | confirmed: AppTermination.swift:19-36; RunController.swift:446,570-574 | §5.4: pre-confirmation, `abandonWait()` and `onWaiting`, independent panel, loss wording from `base_rate`, SIGTERM escalation for orphans; S2 spike; Revision 0's "same last-resort kill as today" retracted |
| H7 expert loses picture + form + log | confirmed | §3.7: Run panel docked in the observer Watch, set-up page can open in its own window; R1 intact |
| H8 PR1/PR2 pins; shippability; sequencing | confirmed: LaunchModeTests.swift:44, LaunchMode.swift:81-86, test_app_bundles.py:42,49-51 | §6 and §7: the edits are listed in PR-B; preview id keeps :49-51 green; safety PR-A first and gated by plan flags; launcher kept until PR-D; preview installable from PR-B; ranges |
| M1 D3 provenance and costs | confirmed: DESIGN.md:10840, 10924, 15100 | D3 rewritten with option C and both sides |
| M2 Unfinished stranded | confirmed | §3.8 row 24: reason, Reveal, Make again (session) |
| M3 library scan on main thread | confirmed | §5.3 |
| M4 Details coupled to `targetRun`; status extras | confirmed (RunController.swift:211-219,376-422) | PR-F decouples; Numbers lists Height/Grounded/Surface and feature groups |
| M5 incomplete View/Checkpoint mapping | confirmed | the dock's View menu is kept whole (row 32); Checkpoint picker kept (row 23) |
| M6 shortcuts | partly unconfirmed (SwiftUI behaviour) | bare letters are labels only; Cmd-/ and Shift-Cmd-W and Opt-Cmd-S dropped; the system sidebar toggle is Ctrl-Cmd-S |
| M7 duplicate controls | confirmed | one home each (§3.8 rows 26-32, §9.2) |
| M8 `SceneStorage`, Dock reopen | unconfirmed in the review too | spike S3, `applicationShouldHandleReopen` |
| M9 wrong facts | confirmed: rate (WorldLauncherModel.swift:65-67), counts, "same kill" | fixed (§3.3, §6, §5.4). **Adjusted:** the review says CLAUDE.md has "no app-name mention"; line 17 names the path `gui/Sources/AlifeControl/...`, which does not change |
| M10 TCC | confirmed as reasoning | §6 adds Local Network, Full Disk Access, Removable Volumes |
| L1 title "under a minute" | confirmed | §3.2 is "What Create does"; the target is "three actions", with the machine wait stated |
| L2 Autopilot | judgment | kept, noted (M15) |
| L3 typed domain; `world.autopause` JSON | confirmed (WorldModel+Decide.swift:129) | PR-B typed protocol (§6) |
| L4 hidden remembered levers | confirmed | chips and "Remembered from last time" line (§3.5) |
| L5 fresh Mac needs a terminal | confirmed | D2 and §3.1 say so; D0 added |
| L6 minimum jumps | confirmed | M12 |
| L7 viewer legibility unstated | confirmed | D5 |
| Accuracy 1-11 | all confirmed except as noted | corrected in the sections above; item 8 (embedding `Viewer3DView` changes its stop and size): handled by "Watch page" semantics in §3.4 and PR-B/F; item 11: `WorldRendererOwnershipTests` rewrite noted in §7. **Adjusted:** review item 5 counts occurrences of "Storm Worlds" (20 in DESIGN.md) and I count lines naming either app (60); both are stated |
| Coordinator cuts and resolutions C1, C2, H1-H8, D3-D5 | applied | §9.2 lists the cuts; Trash, comparison and queue are out; owner decisions are exactly D0-D5 |
