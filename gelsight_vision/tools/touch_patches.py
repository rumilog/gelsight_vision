"""Tool 2: drive the robot to touch each requested capture_index.

Production replacement for the agentic_sr dataset stub. For each capture_index
in the input, this:

  1. Looks up the plate-frame (x, y) via `pipeline.grid.capture_index_to_xy`.
  2. Moves the gelsight tooltip above (x, y) at `safe_z` and descends until
     the z-force exceeds `force_max` (`gelsight/touch.py:touch`).
  3. Triggers a single gelsight capture; the node's callback writes
     `gelsight_<ci:04d>.png/.npy` and parks the height array on
     `node.gelsight_depth` so this function can pick it up.

The returned dict matches agentic_sr's `touch_patches` stub shape:
`{capture_index: (240, 320) float32 depth}`.

Node contract:
- `pose_goal_client`, `frame_id`, `run_action` -- from the entry-point Node
- `force`, `offset`, `move_z`, `stop`, `calibrate_sensor` -- from `touch_io.setup_*`
  + thin method wrappers on the Node class
- `gelsight_capture_image`, `current_capture_index`, `gelsight_depth` -- callback
  protocol: caller sets `current_capture_index`, then `gelsight_capture_image=True`,
  then waits; the callback writes files using the index and stores the height
  array on `gelsight_depth`.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from geometry_msgs.msg import Point, Quaternion

from gelsight_vision.gelsight.touch import touch
from gelsight_vision.pipeline.grid import capture_index_to_xy


def touch_patches(
    node,
    capture_indexes: Iterable[int],
    *,
    orientation: Quaternion,
    force_max: float = 4.0,
    safe_z: float = 0.02,
    mock: bool = False,
) -> dict[int, np.ndarray]:
    """Touch each capture_index in `capture_indexes`, return depth dict."""
    observed: dict[int, np.ndarray] = {}
    for ci in capture_indexes:
        ci = int(ci)
        x, y = capture_index_to_xy(ci)
        node.get_logger().info(
            f"touch_patches: capture_index={ci} at (x={x:.4f}, y={y:.4f}) m"
        )
        node.current_capture_index = ci
        node.gelsight_depth = None
        touch(
            node=node,
            frame_id=node.frame_id,
            mock=mock,
            position=Point(x=x, y=y, z=safe_z),
            orientation=orientation,
            force_max=force_max,
        )
        if node.gelsight_depth is None:
            node.get_logger().error(
                f"touch_patches: no gelsight depth captured for ci={ci}"
            )
            continue
        observed[ci] = node.gelsight_depth.copy()
    return observed
