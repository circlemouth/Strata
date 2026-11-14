"""Small, dependency-free validator for the closed schemas shipped with these tools."""
import json
import math
import re
from pathlib import Path

SCHEMAS = Path(__file__).with_name("schemas")


def load_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    with open(path, encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=pairs,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON number")))


def validate(value, schema, where="document"):
    supported = {"$schema", "title", "type", "properties", "required", "additionalProperties",
                 "const", "enum", "minItems", "uniqueItems", "items", "pattern", "minLength",
                 "minimum", "maximum"}
    if schema.keys() - supported:
        raise ValueError("schema uses an unsupported validation keyword")
    kind = schema.get("type")
    valid = {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: type(value) is int,
        "number": lambda: type(value) in (int, float) and math.isfinite(value),
        "boolean": lambda: type(value) is bool,
        "null": lambda: value is None,
    }
    if kind and not valid[kind]():
        raise ValueError(f"{where}: expected {kind}")
    if "const" in schema and (type(value) is not type(schema["const"]) or value != schema["const"]):
        raise ValueError(f"{where}: unsupported value")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{where}: unsupported enum")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False and value.keys() - props.keys():
            raise ValueError(f"{where}: unknown keys")
        if set(schema.get("required", [])) - value.keys():
            raise ValueError(f"{where}: missing required keys")
        for key in value.keys() & props.keys():
            validate(value[key], props[key], f"{where}.{key}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{where}: too few items")
        if schema.get("uniqueItems") and len({json.dumps(x, sort_keys=True) for x in value}) != len(value):
            raise ValueError(f"{where}: duplicate items")
        for item in value:
            validate(item, schema.get("items", {}), f"{where}[]")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or ("pattern" in schema and not re.search(schema["pattern"], value)):
            raise ValueError(f"{where}: invalid string")
    if type(value) in (int, float):
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            raise ValueError(f"{where}: out of range")
    return value


def validate_document(value, name):
    return validate(value, load_json(SCHEMAS / (name + ".schema.json")))


def write_private_json(path, value):
    """Never overwrite an existing file or follow an output symlink."""
    import os
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
