from __future__ import annotations


def __getattr__(name: str):
    raise RuntimeError(
        "obj_recog.asset_bootstrap has been retired. "
        "The simulation stack now uses a procedural living-room scene and no external asset bootstrap path."
    )
