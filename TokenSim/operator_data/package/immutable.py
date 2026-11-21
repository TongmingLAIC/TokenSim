"""Read-only containers for operator rows, provenance, and shared indexes."""

from collections.abc import Mapping


class FrozenMapping(dict):
    def _reject(self, *args, **kwargs):
        raise TypeError("operator package data is immutable; register a replacement")

    __setitem__ = __delitem__ = __ior__ = _reject
    clear = pop = popitem = setdefault = update = _reject

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return type(self), (dict(self),)


def freeze(value):
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    return value


def to_data(value):
    """Return independent dictionaries and sequences for the YAML file format."""
    if isinstance(value, Mapping):
        return {key: to_data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_data(item) for item in value]
    return value
