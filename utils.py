"""Alias importable de 00_config_and_utils (los nombres que empiezan por dígito no se pueden importar)."""
import importlib as _il
_m = _il.import_module("00_config_and_utils")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
