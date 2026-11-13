# Changes in the English geometry release

## Requested changes

- Translated active GUI labels, menus, tooltips, validation, status messages
  and chart annotations into English.
- Replaced the single target-scale control with shape-specific coordinate inputs.
- Added Circle center/radius, Ellipse center/X-Y radii, Triangle three points,
  Rectangle opposite corners and Square opposite corners with equal-side validation.
- Rover start is now always an explicit X/Y point; heading remains editable.
- Added an optional button that writes the map center into the rover start fields.
- Added an exact target outline alongside the discrete coverage image.
- Saved/restored all geometry parameters and migrated prior GUI checkpoint configs.
- Reject invalid/out-of-map targets instead of silently moving or clipping them.

## Preserved

`agent.py`, `new_rover.py` and `worker.py` are unchanged from the working GUI
archive. PPO's objective, network, reward function, 105-feature observation,
training thread, control behavior and history/export functions are preserved.
Default circle and triangle masks and transition outputs match the originals.
For arbitrary triangle inputs, cell centers on the boundary are explicitly included.

## Test scope

See TEST_REPORT.txt. Core tests were executed on CPU. Qt tests are supplied
but require PySide6; no claim is made that the new GUI ran on this machine.
