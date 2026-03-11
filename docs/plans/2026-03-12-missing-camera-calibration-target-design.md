# Missing Camera Calibration Target Design

**Goal:** When `--camera-calibration` or `CAMERA_CALIBRATION` points to a file that does not exist, run calibration instead of failing and save the generated settings YAML to that exact path.

**Decision:** Treat a missing calibration path as an output target, not an error. Existing files remain explicit inputs. Missing files trigger the existing runtime calibration flow and persist the final YAML to the requested path, creating parent directories when needed.

**Why this approach:** It matches user intent for a stable, user-chosen calibration file location without adding new flags. It preserves the current explicit-file fast path and reuses the existing calibration pipeline.

## Scope

- Keep existing behavior when the explicit calibration file already exists.
- If the explicit path is missing, do not raise `RuntimeError`.
- If self-calibration is enabled, run warm-up/refinement and write the final YAML to the explicit path.
- If self-calibration is disabled, write the approximate calibration YAML to the explicit path.
- Do not also store the final result in the cache when an explicit output target was requested.

## Affected Areas

- `src/obj_recog/auto_calibration.py`
  Add the missing-path-as-output-target behavior inside `ensure_runtime_calibration`.
- `tests/test_auto_calibration.py`
  Cover missing explicit path generation, parent directory creation, and disabled-mode output.

## Tradeoffs

- A missing explicit path no longer signals a typo immediately; it now triggers calibration. That is acceptable because the user explicitly asked for generation behavior.
- The promoted SLAM bridge may still have been initialized from a temporary warm-up settings file, while the saved final YAML lives at the explicit target. This is acceptable because the bridge is already running with equivalent calibration values, and subsequent runs will use the saved target directly.

## Verification

- Focused tests for existing explicit path behavior, missing explicit path generation, and disabled-mode generation.
- Re-run the touched auto-calibration suite after implementation.
