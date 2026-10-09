"""Stdlib JSON Schema checks for Dataset 1 artifacts.

Covers the Draft 2020-12 features used by Dataset 1 schemas. This module is
not imported by the TruthNet production runtime.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

try:
    from scripts.dataset1_common import ValidationIssue
except ModuleNotFoundError:  # pragma: no cover
    from dataset1_common import ValidationIssue

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "data" / "claim_understanding" / "schemas"


def load_schema(name: str) -> dict[str, Any]:
    path = SCHEMA_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def validate_against_schema(instance: object, schema: Mapping[str, Any], *, base_dir: Path | None = None, path: str = "$") -> list[ValidationIssue]:
    directory = base_dir or SCHEMA_DIR
    return _validate(instance, schema, schema, directory, path)


def _resolve_ref(ref: str, root: Mapping[str, Any], base_dir: Path) -> tuple[Mapping[str, Any], Mapping[str, Any], Path]:
    file_part, _, fragment = ref.partition("#")
    current_root, current_dir = root, base_dir
    if file_part:
        path = (base_dir / file_part).resolve()
        current_root = json.loads(path.read_text(encoding="utf-8"))
        current_dir = path.parent
        schema: Any = current_root
    else:
        schema = current_root
    if fragment:
        current = schema
        for part in fragment.strip("/").split("/"):
            if part:
                current = current[part.replace("~1", "/").replace("~0", "~")]
        schema = current
    if not isinstance(schema, Mapping):
        raise ValueError(f"unsupported $ref target: {ref}")
    return schema, current_root, current_dir


def _validate(instance: object, schema: Mapping[str, Any], root: Mapping[str, Any], base_dir: Path, path: str) -> list[ValidationIssue]:
    if "$ref" in schema:
        target, target_root, target_dir = _resolve_ref(str(schema["$ref"]), root, base_dir)
        return _validate(instance, target, target_root, target_dir, path)
    issues: list[ValidationIssue] = []
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            option_issues = _validate(instance, option, root, base_dir, path)
            if not option_issues:
                return []
        return [ValidationIssue("schema_mismatch", "value matches none of anyOf", path)]
    if "allOf" in schema:
        for option in schema["allOf"]:
            issues.extend(_validate(instance, option, root, base_dir, path))
    expected_type = schema.get("type")
    if expected_type is not None and not _type_matches(instance, expected_type):
        issues.append(ValidationIssue("schema_type", f"expected {expected_type}", path))
        return issues
    if "const" in schema and instance != schema["const"]:
        issues.append(ValidationIssue("schema_const", "const mismatch", path))
    if "enum" in schema and instance not in schema["enum"]:
        issues.append(ValidationIssue("schema_enum", "value not in enum", path))
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            issues.append(ValidationIssue("schema_minLength", "string shorter than minLength", path))
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            issues.append(ValidationIssue("schema_pattern", "string does not match pattern", path))
    if isinstance(instance, bool):
        pass
    elif isinstance(instance, (int, float)) and "minimum" in schema and instance < schema["minimum"]:
        issues.append(ValidationIssue("schema_minimum", "numeric value below minimum", path))
    if isinstance(instance, list) and "items" in schema:
        if "minItems" in schema and len(instance) < schema["minItems"]:
            issues.append(ValidationIssue("schema_minItems", "array shorter than minItems", path))
        for index, item in enumerate(instance):
            issues.extend(_validate(item, schema["items"], root, base_dir, f"{path}[{index}]"))
    if isinstance(instance, Mapping) and (schema.get("type") == "object" or "properties" in schema or "required" in schema):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                issues.append(ValidationIssue("schema_required", f"missing required property {key}", f"{path}.{key}"))
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in properties:
                issues.extend(_validate(value, properties[key], root, base_dir, f"{path}.{key}"))
            elif additional is False:
                issues.append(ValidationIssue("schema_additional", f"unexpected property {key}", f"{path}.{key}"))
            elif isinstance(additional, Mapping):
                issues.extend(_validate(value, additional, root, base_dir, f"{path}.{key}"))
    return issues


def _type_matches(instance: object, expected: object) -> bool:
    if isinstance(expected, list):
        return any(_type_matches(instance, item) for item in expected)
    mapping = {
        "object": lambda value: isinstance(value, Mapping),
        "array": lambda value: isinstance(value, list),
        "string": lambda value: isinstance(value, str),
        "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": lambda value: isinstance(value, bool),
        "null": lambda value: value is None,
    }
    checker = mapping.get(str(expected))
    return bool(checker and checker(instance))
