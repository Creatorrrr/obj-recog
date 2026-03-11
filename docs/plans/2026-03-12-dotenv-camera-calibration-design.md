# Dotenv Camera Calibration Design

**Goal:** Allow the runtime to pick up a calibration file path from `.env` without changing the existing CLI contract.

**Decision:** Add support for `CAMERA_CALIBRATION=/abs/path/camera.yaml` in `.env` with the precedence `--camera-calibration` CLI value first, then `.env`, then existing automatic calibration/cache behavior.

**Why this approach:** It preserves explicit CLI behavior, requires no new user-facing flags, and fits the current startup flow because `.env` is already loaded in `main.run()` before any camera or SLAM setup begins.

## Scope

- Read `CAMERA_CALIBRATION` from process environment when parsing config.
- Keep `--camera-calibration` as the highest-priority source.
- Preserve the current automatic calibration and cache reuse flow when neither CLI nor `.env` provides a path.
- Do not add path existence validation to config parsing; keep runtime validation where it already exists.

## Affected Areas

- `src/obj_recog/config.py`
  Resolve the default value for `camera_calibration` from `os.getenv("CAMERA_CALIBRATION")` when the CLI flag is absent.
- `src/obj_recog/main.py`
  No behavior change needed beyond relying on the existing `.env` loader before `parse_config()`.
- `tests/test_config.py`
  Add coverage for `.env`-style environment fallback and CLI precedence over environment values.
- `tests/test_main_smoke.py`
  Add a smoke test proving that `.env`-provided `CAMERA_CALIBRATION` is seen by startup and routed through runtime calibration.

## Risks

- `parse_config()` may be called in tests or utility code before `.env` is loaded. This is acceptable because direct environment lookup still works, and `main()` already loads `.env` first.
- Invalid paths from `.env` will fail at runtime, matching the current behavior for invalid CLI paths.

## Verification

- Focused config tests for default, env fallback, and CLI precedence.
- Focused main smoke test proving `.env` affects startup.
- Re-run the touched focused tests after implementation.
