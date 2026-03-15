from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath


class UnityVendorCheckError(RuntimeError):
    pass


def default_unity_project_root() -> Path:
    return Path(__file__).resolve().parents[2] / "unity"


def default_vendor_manifest_path() -> Path:
    return default_unity_project_root() / "vendor_manifest.json"


def validate_unity_vendor_setup(
    *,
    unity_project_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> None:
    project_root = (
        default_unity_project_root()
        if unity_project_root is None
        else Path(unity_project_root).expanduser().resolve()
    )
    manifest_file = (
        default_vendor_manifest_path()
        if manifest_path is None
        else Path(manifest_path).expanduser().resolve()
    )
    manifest = _load_manifest(manifest_file)
    problems: list[str] = []

    if not project_root.is_dir():
        raise UnityVendorCheckError(
            f"Unity project root not found: {project_root}\n"
            "Clone the repo with the tracked Unity project, then rerun the vendor check."
        )

    expected_editor_version = str(manifest.get("unity_editor_version", "")).strip()
    if not expected_editor_version:
        raise UnityVendorCheckError(f"vendor manifest is missing unity_editor_version: {manifest_file}")
    actual_editor_version = _read_unity_editor_version(project_root)
    if actual_editor_version != expected_editor_version:
        problems.append(
            f"Unity editor version mismatch: expected {expected_editor_version}, found {actual_editor_version}"
        )

    packages = manifest.get("packages")
    if not isinstance(packages, list) or not packages:
        raise UnityVendorCheckError(f"vendor manifest does not define any packages: {manifest_file}")

    for package in packages:
        problems.extend(_validate_package(project_root=project_root, package=package))

    if problems:
        package_summaries = ", ".join(
            f"{package.get('name', 'unknown')} {package.get('version', 'unknown')}" for package in packages
        )
        required_paths = ", ".join(
            str(path)
            for package in packages
            for path in package.get("required_paths", ())
            if isinstance(path, str) and path
        )
        raise UnityVendorCheckError(
            "Unity Apartment Kit vendor check failed.\n"
            + "\n".join(f"- {problem}" for problem in problems)
            + "\n"
            + f"Expected Unity Editor: {expected_editor_version}\n"
            + f"Expected Asset Store package(s): {package_summaries}\n"
            + f"Required import path(s): {required_paths}\n"
            + "Import the exact package version into the default Unity asset path and rerun:\n"
            + "python -m obj_recog.unity_vendor_check --unity-project-root unity"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the local Unity Apartment Kit vendor setup before running sim mode."
    )
    parser.add_argument(
        "--unity-project-root",
        type=str,
        default=str(default_unity_project_root()),
        help="Path to the Unity project root. Defaults to the repo's unity folder.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(default_vendor_manifest_path()),
        help="Path to the tracked vendor manifest JSON file.",
    )
    args = parser.parse_args(argv)
    try:
        validate_unity_vendor_setup(
            unity_project_root=args.unity_project_root,
            manifest_path=args.manifest,
        )
    except UnityVendorCheckError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("Unity Apartment Kit vendor check passed.", file=sys.stdout)
    return 0


def _load_manifest(manifest_file: Path) -> dict[str, object]:
    if not manifest_file.is_file():
        raise UnityVendorCheckError(f"vendor manifest not found: {manifest_file}")
    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UnityVendorCheckError(f"vendor manifest is not valid JSON: {manifest_file}") from exc
    if not isinstance(payload, dict):
        raise UnityVendorCheckError(f"vendor manifest must be a JSON object: {manifest_file}")
    return payload


def _read_unity_editor_version(project_root: Path) -> str:
    version_file = project_root / "ProjectSettings" / "ProjectVersion.txt"
    if not version_file.is_file():
        raise UnityVendorCheckError(f"Unity ProjectVersion.txt not found: {version_file}")
    match = re.search(
        r"^m_EditorVersion:\s*(?P<version>[^\r\n]+)\s*$",
        version_file.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise UnityVendorCheckError(f"Unity editor version not found in: {version_file}")
    return match.group("version").strip()


def _validate_package(*, project_root: Path, package: object) -> list[str]:
    if not isinstance(package, dict):
        raise UnityVendorCheckError("vendor manifest package entries must be JSON objects")

    package_name = str(package.get("name", "unknown")).strip() or "unknown"
    package_version = str(package.get("version", "unknown")).strip() or "unknown"
    problems: list[str] = []

    required_paths = package.get("required_paths")
    if not isinstance(required_paths, list) or not required_paths:
        raise UnityVendorCheckError(f"vendor manifest package {package_name} is missing required_paths")
    for required_path in required_paths:
        if not isinstance(required_path, str) or not required_path:
            raise UnityVendorCheckError(f"vendor manifest package {package_name} has an invalid required path")
        resolved_path = _asset_path(project_root, required_path)
        if not resolved_path.exists():
            problems.append(
                f"{package_name} {package_version} is missing required path {required_path}"
            )

    asset_origin_sample = package.get("asset_origin_sample")
    if asset_origin_sample is not None:
        problems.extend(
            _validate_asset_origin_sample(
                project_root=project_root,
                package_name=package_name,
                package_version=package_version,
                sample=asset_origin_sample,
            )
        )

    guid_samples = package.get("guid_samples")
    if not isinstance(guid_samples, list) or not guid_samples:
        raise UnityVendorCheckError(f"vendor manifest package {package_name} is missing guid_samples")
    for guid_sample in guid_samples:
        problems.extend(
            _validate_guid_sample(
                project_root=project_root,
                package_name=package_name,
                package_version=package_version,
                sample=guid_sample,
            )
        )
    return problems


def _validate_asset_origin_sample(
    *,
    project_root: Path,
    package_name: str,
    package_version: str,
    sample: object,
) -> list[str]:
    if not isinstance(sample, dict):
        raise UnityVendorCheckError(f"vendor manifest asset_origin_sample for {package_name} must be an object")
    asset_path = _required_string(sample, "asset_path", package_name)
    expected_name = _required_string(sample, "package_name", package_name)
    expected_version = _required_string(sample, "package_version", package_name)
    expected_product_id = str(sample.get("product_id", "")).strip()
    meta_path = _meta_path(project_root, asset_path)
    problems: list[str] = []
    if not meta_path.is_file():
        problems.append(
            f"{package_name} {package_version} asset origin sample is missing meta file {asset_path}.meta"
        )
        return problems
    meta_text = meta_path.read_text(encoding="utf-8")
    if f"packageName: {expected_name}" not in meta_text:
        problems.append(
            f"{package_name} {package_version} asset origin sample has unexpected packageName in {asset_path}.meta"
        )
    if f"packageVersion: {expected_version}" not in meta_text:
        problems.append(
            f"{package_name} {package_version} asset origin sample has unexpected packageVersion in {asset_path}.meta"
        )
    if expected_product_id and f"productId: {expected_product_id}" not in meta_text:
        problems.append(
            f"{package_name} {package_version} asset origin sample has unexpected productId in {asset_path}.meta"
        )
    return problems


def _validate_guid_sample(
    *,
    project_root: Path,
    package_name: str,
    package_version: str,
    sample: object,
) -> list[str]:
    if not isinstance(sample, dict):
        raise UnityVendorCheckError(f"vendor manifest guid_samples entry for {package_name} must be an object")
    asset_path = _required_string(sample, "asset_path", package_name)
    expected_guid = _required_string(sample, "guid", package_name)
    meta_path = _meta_path(project_root, asset_path)
    problems: list[str] = []
    if not _asset_path(project_root, asset_path).exists():
        problems.append(f"{package_name} {package_version} guid sample asset is missing {asset_path}")
        return problems
    if not meta_path.is_file():
        problems.append(f"{package_name} {package_version} guid sample meta is missing {asset_path}.meta")
        return problems
    actual_guid = _read_guid(meta_path)
    if actual_guid != expected_guid:
        problems.append(
            f"{package_name} {package_version} guid mismatch for {asset_path}: expected {expected_guid}, found {actual_guid}"
        )
    return problems


def _required_string(payload: dict[str, object], key: str, package_name: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise UnityVendorCheckError(f"vendor manifest package {package_name} is missing {key}")
    return value.strip()


def _asset_path(project_root: Path, unity_asset_path: str) -> Path:
    relative_path = PurePosixPath(unity_asset_path)
    return project_root.joinpath(*relative_path.parts)


def _meta_path(project_root: Path, unity_asset_path: str) -> Path:
    asset_path = _asset_path(project_root, unity_asset_path)
    return asset_path.with_name(f"{asset_path.name}.meta")


def _read_guid(meta_path: Path) -> str:
    match = re.search(
        r"^guid:\s*(?P<guid>[0-9a-fA-F]{32})\s*$",
        meta_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise UnityVendorCheckError(f"Unity guid not found in meta file: {meta_path}")
    return match.group("guid").lower()


if __name__ == "__main__":
    raise SystemExit(main())
