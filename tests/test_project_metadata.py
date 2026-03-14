from __future__ import annotations

from pathlib import Path
import tomllib


def test_pyproject_pins_windows_python312_torch_to_cuda_wheels() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]

    assert (
        "torch @ https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp312-cp312-win_amd64.whl ; "
        "platform_system == 'Windows' and platform_machine == 'AMD64' and python_version >= '3.12' and python_version < '3.13'"
    ) in dependencies
    assert (
        "torchvision @ https://download.pytorch.org/whl/cu128/torchvision-0.25.0%2Bcu128-cp312-cp312-win_amd64.whl ; "
        "platform_system == 'Windows' and platform_machine == 'AMD64' and python_version >= '3.12' and python_version < '3.13'"
    ) in dependencies


def test_pyproject_keeps_generic_torch_fallback_for_other_platforms() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]

    assert (
        "torch ; not (platform_system == 'Windows' and platform_machine == 'AMD64' and python_version >= '3.12' and python_version < '3.13')"
    ) in dependencies
    assert (
        "torchvision ; not (platform_system == 'Windows' and platform_machine == 'AMD64' and python_version >= '3.12' and python_version < '3.13')"
    ) in dependencies
