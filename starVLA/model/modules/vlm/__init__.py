from __future__ import annotations

import json
from pathlib import Path


def _read_local_hf_config_name(base_vlm: str) -> str:
    config_path = Path(base_vlm).expanduser() / "config.json"
    if not config_path.is_file():
        return ""

    try:
        with config_path.open("r") as f:
            hf_config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""

    names = []
    for key in ("model_type", "architectures"):
        value = hf_config.get(key)
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, list):
            names.extend(str(item) for item in value)
    return " ".join(names)


def get_vlm_family(base_vlm: str) -> str:
    """Resolve the VLM wrapper family from a HF id or local checkpoint path."""
    vlm_name = str(base_vlm)
    haystack = f"{vlm_name} {_read_local_hf_config_name(vlm_name)}".lower()
    compact = (
        haystack.replace("_", "")
        .replace("-", "")
        .replace(".", "")
        .replace("/", "")
        .replace(" ", "")
    )

    if "florence" in haystack:
        return "florence"
    if "qwen25vl" in compact or "qwen25" in compact or "qwen2vl" in compact:
        return "qwen2.5vl"
    if "qwen35" in compact or "qwen3vl" in compact or "qwen3" in compact:
        return "qwen3vl"

    raise NotImplementedError(
        f"VLM model {base_vlm} not implemented. "
        "Supported base_vlm families: Qwen2.5-VL, Qwen3-VL, Qwen3.5, Florence."
    )


def get_vlm_data_model_type(base_vlm: str) -> str:
    family = get_vlm_family(base_vlm)
    if family == "qwen3vl":
        return "qwen3vl"
    if family == "qwen2.5vl":
        return "qwen2.5vl"
    return family


def get_vlm_model(config):

    vlm_name = config.framework.qwenvl.base_vlm
    vlm_family = get_vlm_family(vlm_name)

    if vlm_family == "qwen2.5vl":
        from .QWen2_5 import _QWen_VL_Interface 
        return _QWen_VL_Interface(config)
    elif vlm_family == "qwen3vl":
        from .QWen3 import _QWen3_VL_Interface

        return _QWen3_VL_Interface(config)
    elif vlm_family == "florence": # temp for some ckpt
        from .Florence2 import _Florence_Interface 
        return _Florence_Interface(config)
    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")
