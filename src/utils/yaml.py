import yaml
from pathlib import Path
import pprint


# 1) Create a Loader subclass that we can attach extra state to
class LoaderWithInclude(yaml.FullLoader):
    # we'll attach .shared_manifests = { alias: Path(...) }
    pass


def _dummy_include(loader, node):
    # we don't care about the value, just skip it
    return None


def _real_include(loader: LoaderWithInclude, node):
    """
    Called when parsing !include alias:component_name.
    """
    # the scalar is e.g. "shared:wan/vae"
    prefix, comp_name = loader.construct_scalar(node).split(":", 1)

    if prefix == "shared":
        # e.g. comp_name is "wan/vae", so we extract "wan" as the alias
        alias = comp_name.split("/", 1)[0]
    else:
        alias = prefix

    manifest = loader.shared_manifests.get(alias)
    if manifest is None:
        raise yaml.constructor.ConstructorError(
            None, None, f"Unknown shared alias {alias!r}", node.start_mark
        )

    # load the shared manifest _with_ our LoaderWithInclude (so nested !includes also work)
    shared_doc = yaml.load(manifest.read_text(), Loader=LoaderWithInclude)

    # find the component by name in any top-level list
    for value in shared_doc.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("name") == comp_name:
                    return item

    raise yaml.constructor.ConstructorError(
        None,
        None,
        f"Component named {comp_name!r} not found in {manifest}",
        node.start_mark,
    )


yaml.FullLoader.add_constructor("!include", _dummy_include)
LoaderWithInclude.add_constructor("!include", _real_include)


def load_yaml(file_path: str | Path):
    file_path = Path(file_path)
    text = file_path.read_text()
    # --- PASS 1: extract your `shared:` list with a loader that skips !include tags ---
    prelim = yaml.load(text, Loader=yaml.FullLoader)
    # prelim.get("shared", [...]) is now a list of file-paths strings.
    # build alias → manifest Path
    shared_manifests = {}
    for entry in prelim.get("shared", []):
        p = (file_path.parent / entry).resolve()
        # assume e.g. "shared_wan.yml" → alias "wan"
        alias = p.stem.split("_", 1)[1]
        shared_manifests[alias] = p
    # attach it to our custom loader
    LoaderWithInclude.shared_manifests = shared_manifests
    # --- PASS 2: real load with !include expansion ---
    return yaml.load(text, Loader=LoaderWithInclude)
