import ast
from dataclasses import fields

from .base import BaseConfig

def parse_cli_kv(kv_list: list[str]) -> dict:
    ''' ["gamma=0.95", "critic_num=10"] -> {"gamma": 0.95, "critic_num": 10} '''
    out = {}
    
    for item in (kv_list or []):
        if '=' not in item:
            raise ValueError(f'override format error: {item!r}, should be key=value')

        k, v = item.split('=', 1)
        try:
            out[k.strip()] = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            out[k.strip()] = v.strip()
    
    return out

def apply_overrides(config: BaseConfig, overrides: dict):
    if not overrides:
        return config
    
    valid_keys = {f.name for f in fields(config)}
    unknown = set(overrides) - valid_keys
    if unknown:
        raise ValueError(f'{type(config).__name__} does not include: {unknown}')
    return config.override(**overrides)