from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.ui.nodes import (
    TextNode,
    NumberNode,
    FloatNode,
    BoolNode,
    ListNode,
    FileNode,
    UINode,
)


NODE_COMPONENTS = {
    "text": TextNode,
    "number": NumberNode,
    "float": FloatNode,
    "bool": BoolNode,
    "list": ListNode,
    "file": FileNode,
    # For selection/slider we still map to a base node type; UI renderer can use metadata
    "select": TextNode,
    "slider": FloatNode,
}


def build_ui_nodes(ui_spec: Dict[str, Any] | None) -> List[UINode]:
    """
    Convert a manifest ui spec (v1) into concrete UINode instances.
    This is a helper for building a rendering engine later; it does not wire
    them into the engine. It preserves useful UI metadata on the node objects.
    """
    if not ui_spec:
        return []

    mode = (ui_spec.get("mode") or "simple").lower()
    decl = (
        ui_spec.get("simple", {}) if mode == "simple" else ui_spec.get("advanced", {})
    )
    inputs = decl.get("inputs", [])

    result: List[UINode] = []
    for item in inputs:
        comp_key = (item.get("component") or item.get("type") or "text").lower()
        NodeCls = NODE_COMPONENTS.get(comp_key, TextNode)
        node = NodeCls(
            id=item.get("id", ""),
            name=item.get("label", item.get("id", "")),
            description=item.get("description", ""),
            default_value=item.get("default", None),
        )
        # attach UI metadata for rendering engines
        setattr(
            node,
            "ui_meta",
            {
                "component": comp_key,
                "options": item.get("options"),
                "min": item.get("min"),
                "max": item.get("max"),
                "step": item.get("step"),
                "group": item.get("group"),
                "order": item.get("order"),
                "mapping": item.get("mapping"),
                "required": item.get("required", False),
            },
        )
        result.append(node)

    return result


def enumerate_bindings(
    ui_spec: Dict[str, Any] | None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Extract a normalized list of (param_id, mapping) pairs from the UI spec.
    Useful for wiring values into engine.run and preprocessors later.
    """
    if not ui_spec:
        return []
    bindings: List[Tuple[str, Dict[str, Any]]] = []
    for item in ui_spec.get("simple", {}).get("inputs", []) or []:
        if "mapping" in item and item.get("id"):
            bindings.append((item["id"], item["mapping"]))
    for item in ui_spec.get("advanced", {}).get("inputs", []) or []:
        if "mapping" in item and item.get("id"):
            bindings.append((item["id"], item["mapping"]))
    return bindings
