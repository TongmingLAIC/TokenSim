"""Locate bundled runtime data in a checkout or an installed wheel."""

from importlib.metadata import distributions
import os
from pathlib import Path
import sys


_DATA_DIRECTORY = Path("share/tokensim/data")
_CATALOG_ENTRY = _DATA_DIRECTORY / "topologies/links.yaml"


def _installed_data_root():
    # Wheel data may live outside sys.prefix, for example with a user install.
    module = Path(__file__).resolve()
    for installed in distributions(name="tokensim"):
        if Path(installed.locate_file("TokenSim/config/paths.py")).resolve() != module:
            continue
        for file in installed.files or ():
            if file.parts[-len(_CATALOG_ENTRY.parts) :] == _CATALOG_ENTRY.parts:
                candidate = Path(installed.locate_file(file)).resolve()
                if candidate.is_file():
                    return candidate.parents[1]
    return None


def default_data_root():
    override = os.environ.get("TOKENSIM_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2] / "data"
    if (checkout / "devices").is_dir():
        return checkout
    return _installed_data_root() or (Path(sys.prefix) / _DATA_DIRECTORY)
