from __future__ import annotations


def __getattr__(name: str):
    raise RuntimeError(
        "obj_recog.sim_assets has been retired. "
        "The living-room simulation builds geometry procedurally from obj_recog.sim_scene."
    )
