from __future__ import annotations

# JSON Schema for Apex Manifest v1
# Keep permissive defaults to avoid breaking existing flows while providing
# strong guidance and validation for new manifests.

MANIFEST_SCHEMA_V1: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Apex Manifest v1",
    "type": "object",
    "required": ["api_version", "kind", "metadata", "spec"],
    "properties": {
        "api_version": {"type": "string", "pattern": r"^apex(/ai)?/v1$|^apex/v1$"},
        "kind": {
            "type": "string",
            "enum": [
                "Model",
                "Pipeline",
            ],
        },
        "metadata": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string", "min_length": 1},
                "version": {
                    "type": "string",
                    "pattern": r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+].*)?$",
                },
                "description": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "author": {"type": "string"},
                "license": {"type": "string"},
                "homepage": {"type": "string"},
                "registry": {"type": "string"},
                "annotations": {"type": "object", "additional_properties": True},
            },
            "additional_properties": True,
        },
        "spec": {
            "type": "object",
            "required": ["engine", "model_type"],
            "properties": {
                "engine": {"type": "string"},
                "model_type": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
                "model_types": {"type": "array", "items": {"type": "string"}},
                "engine_type": {"type": "string", "enum": ["torch", "mlx"]},
                "denoise_type": {"type": "string"},
                "shared": {"type": "array", "items": {"type": "string"}},
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "scheduler",
                                    "vae",
                                    "text_encoder",
                                    "transformer",
                                    "helper",
                                ],
                            },
                            "name": {"type": "string"},
                            "base": {"type": "string"},
                            "model_path": {"type": "string"},
                            "config_path": {"type": "string"},
                            "file_pattern": {"type": "string"},
                            "tag": {"type": "string"},
                            "key_map": {
                                "type": "object",
                                "additional_properties": True,
                            },
                            "extra_kwargs": {
                                "type": "object",
                                "additional_properties": True,
                            },
                            "save_path": {"type": "string"},
                            "converter_kwargs": {
                                "type": "object",
                                "additional_properties": True,
                            },
                            "model_key": {"type": "string"},
                            "extra_model_paths": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "converted_model_path": {"type": "string"},
                        },
                        "additional_properties": True,
                    },
                },
                "preprocessors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": {"type": "string"},
                            "name": {"type": "string"},
                            "model_path": {"type": "string"},
                            "config_path": {"type": "string"},
                            "save_path": {"type": "string"},
                            "kwargs": {
                                "type": "object",
                                "additional_properties": True,
                            },
                        },
                        "additional_properties": True,
                    },
                },
                "postprocessors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": {"type": "string"},
                            "name": {"type": "string"},
                            "model_path": {"type": "string"},
                            "config_path": {"type": "string"},
                            "kwargs": {
                                "type": "object",
                                "additional_properties": True,
                            },
                        },
                        "additional_properties": True,
                    },
                },
                "defaults": {
                    "type": "object",
                    "additional_properties": True,
                },
                "loras": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string"},
                                    "path": {"type": "string"},
                                    "url": {"type": "string"},
                                    "scale": {"type": "number"},
                                    "name": {"type": "string"},
                                },
                                "additional_properties": True,
                            },
                        ]
                    },
                },
                "save": {
                    "type": "object",
                    "additional_properties": True,
                },
                "ui": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["simple", "advanced", "complex"],
                        },
                        "simple": {
                            "type": "object",
                            "properties": {
                                "inputs": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "required": ["id"],
                                        "properties": {
                                            "id": {"type": "string"},
                                            "label": {"type": "string"},
                                            "description": {"type": "string"},
                                            "type": {
                                                "type": "string",
                                                "enum": [
                                                    "text",
                                                    "number",
                                                    "float",
                                                    "bool",
                                                    "list",
                                                    "file",
                                                    "select",
                                                    "slider",
                                                ],
                                            },
                                            "default": {},
                                            "required": {"type": "boolean"},
                                            "options": {
                                                "type": "array",
                                                "items": {"type": ["string", "number"]},
                                            },
                                            "min": {"type": ["number", "integer"]},
                                            "max": {"type": ["number", "integer"]},
                                            "step": {"type": ["number", "integer"]},
                                            "group": {"type": "string"},
                                            "order": {"type": "integer"},
                                            "component": {"type": "string"},
                                            "mapping": {
                                                "type": "object",
                                                "properties": {
                                                    "target": {"type": "string"},
                                                    "param": {"type": "string"},
                                                    "path": {"type": "string"},
                                                },
                                                "additional_properties": True,
                                            },
                                        },
                                        "additional_properties": True,
                                    },
                                }
                            },
                            "additional_properties": True,
                        },
                        "advanced": {
                            "type": "object",
                            "properties": {
                                "expose": {
                                    "oneOf": [
                                        {"type": "string", "enum": ["all"]},
                                        {"type": "array", "items": {"type": "string"}},
                                    ]
                                },
                                "inputs": {
                                    "$ref": "#/properties/spec/properties/ui/properties/simple/properties/inputs"
                                },
                            },
                            "additional_properties": True,
                        },
                    },
                    "additional_properties": True,
                },
            },
            "additional_properties": True,
        },
        # Back-compat: allow uppercase UI at top-level or under spec
        "UI": {"$ref": "#/properties/spec/properties/ui"},
    },
    "additional_properties": True,
}
