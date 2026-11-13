# PPO Rover Lab — English GUI and parameterized targets

This is an update of the working `rover_ppo_gui` project. The interface,
plot labels, tooltips, status messages, validation errors and this manual
are in English. The PPO agent, optimizer and rover motion are retained.

## Run

Extract the **entire** `rover_ppo_gui_en` folder into a new directory.
Do not copy only `window.py`: it depends on the updated configuration,
environment, geometry and plotting modules.

```bash
cd rover_ppo_gui_en
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.
An existing working virtual environment can also be used; no new third-party
dependencies were added by this update. CPU remains the default. CUDA needs
a CUDA-enabled PyTorch installation and a supported GPU; this project does
not install drivers or modify the system configuration.

## Target geometry

Choose **Environment → Target shape**. The **Target geometry** panel then
shows only the relevant inputs. Inputs are **map coordinates**, not pixels.
The map origin `(0, 0)` is at the lower-left; X increases to the right and
Y increases upwards. The full workspace is `[0, width] × [0, height]`.

| Shape | Inputs | Definition |
|---|---|---|
| Circle | Center `(x, y)`, radius | A filled circular scanning target. |
| Ellipse | Center `(x, y)`, Radius X `(a)`, Radius Y `(b)` | Horizontal and vertical **semi-axes**, not full widths. Either radius may be larger. |
| Triangle | Point 1 `(x1, y1)`, Point 2 `(x2, y2)`, Point 3 `(x3, y3)` | Three distinct, non-collinear vertices; either orientation is accepted. |
| Rectangle | Point 1 and Point 2 | **Opposite corners**, in either order. Sides are parallel to the map axes. |
| Square | Point 1 and Point 2 | Opposite corners of an axis-aligned square; `abs(x2-x1) == abs(y2-y1)` is required. |

Ellipses, squares and rectangles are **axis-aligned**; this release has no
independent rotation parameter for those shapes. A triangle can have any
orientation because all three vertices are editable. All shapes describe
filled target areas, not only their outlines.

The target is not automatically centered, rescaled or clipped when the map
size changes. **The complete target must fit inside the map.** Invalid
coordinates, nonpositive radii, degenerate triangles and invalid square
corners produce an English error message. For unequal side lengths, choose
Rectangle rather than Square. Changing a coordinate does not silently
change another coordinate.

Click **Apply / preview environment** to display the selected shape, the
rover start and the laser footprint before training. Start also validates
and uses the current fields. The dashed target outline follows the exact
specified geometry; the gray/green image shows the discrete coverage mask.

### Examples for a 100 × 100 map

| Shape | Example settings |
|---|---|
| Circle | Center `(50, 50)`, radius `25` |
| Ellipse | Center `(50, 50)`, Radius X `30`, Radius Y `20` |
| Triangle | `(20, 20)`, `(80, 20)`, `(50, 80)` |
| Rectangle | Opposite corners `(15, 20)` and `(85, 65)` |
| Square | Opposite corners `(20, 20)` and `(80, 80)` |

## Rover starting point

In **Rover and laser → Start point**, enter X and Y directly. These values
are used by the environment, preview and every episode reset. They are not
overridden by an automatic center-start checkbox.

**Place start at map center** is an optional convenience button: it writes
`width / 2` and `height / 2` into the two coordinate fields. Subsequent map
changes do not keep the rover automatically centered.

**Start heading** remains independent: 0° points right (+X), 90° points up
(+Y), and positive angles turn counterclockwise. The rover can start outside
the target as long as it stays inside the rectangular map. Only the rover's
center is constrained; this is not a full vehicle-body collision model.

## Training and plots

The left panel also contains the existing laser, movement and PPO settings.
**Laser diameter** is the diameter of a circular footprint. The full swept
footprint between positions is counted, rather than only the endpoint.
Actions remain `left`, `forward`, `right`: turning changes heading in place.
The arrow makes those stationary turns visible.

The center contains the rover/coverage plot, episode cost, coverage history
and a **PPO diagnostics** tab. Cost is:

```
episode_cost = -sum(unscaled_episode_rewards)
```

Lower is better; values can be negative. This is not the optimizer loss or
pure path length. PPO diagnostics show the policy, value, total loss or
entropy separately. Only completed training episodes appear in the episode
charts; policy simulation does not modify weights or training history.

The reward function is unchanged:

- New target cells: `min(10, 10 * new_cells / (beam_width * step_size))`.
- Previously scanned target only: `-0.5`.
- Laser outside target or blocked by map boundary: `-1`.
- Turn in place: `-0.15`.
- Coverage goal reached: additional `+40`.

One coverage cell is 1 × 1 map unit. A target includes cell centers on or
inside its boundary. Tiny targets that contain no cell centers are rejected.
The initial footprint is scanned without reward, as before. A target already
meeting the coverage goal at the start is rejected for training. Coverage
percentages describe the raster mask, not an exact continuous-area proof.

## Controls

| Control | Behavior |
|---|---|
| Start / Continue training | Performs the configured number of additional PPO updates; continuing keeps weights and Adam state. |
| Stop | Cooperatively stops after the current environment step or PPO update. |
| Reset simulation | Resets the preview path and coverage, preserving the model and training history. |
| Reset training | After confirmation, discards in-memory weights/history and unlocks settings. Saved files stay on disk. |
| Policy simulation | Runs a separate episode without learning, using the current saved-in-memory policy. |

**Stop is not an exact pause:** an incomplete rollout is discarded; the next
Start begins a new episode with the latest completed weights. Parameters,
including shape geometry and rover start, are locked once a model exists.
Use **Reset training** before changing the task, so history and model settings
are not silently mixed. Slow-down and preview timing can still be changed.

Training continues in the existing Qt worker thread; UI updates remain on
the main thread. A snapshot contains copies of the map, path and outline.
Preview refresh is throttled and may skip displayed frames during fast
training, but the environment still calculates every step.

## Save, load and export

The **File** menu supports model/checkpoint saving and loading, CSV history,
and separate PNG plots. Checkpoints now include every target coordinate,
radius and rover start coordinate. Loading restores the configuration along
with the weights, optimizer and history.

**Older GUI checkpoints:** `target_scale` is converted to explicit circle or
triangle coordinates. The old `start_center` flag is converted to explicit
start X/Y. This migration is tested for both original shapes and preserves
the original geometry. Old weights-only checkpoints still require the user
to set the correct training environment before loading.

Load only trusted model files; the loader retains `weights_only=True`.
Weights are not retrained or improved merely by adding new target types.
A policy trained on one geometry may not perform well on another.

## Files and compatibility

- `main.py`: application entry point; English numeric locale.
- `window.py`: English controls, shape-specific forms, start-point inputs.
- `config.py`: explicit parameters, validation and legacy-config migration.
- `geometry.py`: circle/ellipse/triangle/rectangle/square masks and outlines.
- `envinronment.py`: environment; original misspelling retained for imports.
- `engine.py`: existing training/evaluation flow, snapshot and status updates.
- `plots.py`: English labels and target outline.
- `agent.py`, `new_rover.py`, `worker.py`: byte-identical to the working GUI release.
- `original_sources/`: archived input sources for regression tests; not the active GUI.

Observation dimension remains 105, with the existing 7 × 7 local maps and
pose/global features. PPO architecture, GAE, clipping, reward scaling and
episode-end/bootstrap semantics were not changed by this update.

## Tests and verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

For headless Qt testing on a computer with PySide6 installed:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q tests
```

See **TEST_REPORT.txt** for the results actually obtained for this release.
The core tests cover all five geometries, coordinate order, invalid inputs,
start/reset behavior, swept laser coverage, training, evaluation, checkpoint
roundtrips, legacy migration and the unchanged default PPO update.

The Qt tests are included but are skipped when PySide6 is unavailable. Do
not interpret the headless Matplotlib preview images in `verification/` as
screenshots of a successfully launched Qt window or as trained PPO results.
No claim is made that these changes guarantee complete coverage or an
optimal route. CUDA and your desktop display stack require local testing.

## API references

- Qt stacked parameter panels: https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QStackedWidget.html
- Qt numeric inputs: https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QDoubleSpinBox.html
