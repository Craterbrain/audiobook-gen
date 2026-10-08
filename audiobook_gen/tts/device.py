"""Device selection: Intel XPU (Arc B580) preferred, CPU fallback. No CUDA APIs used."""
import warnings


def pick_device(pref: str = "auto") -> str:
    import torch

    if pref == "cpu":
        return "cpu"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        n = torch.xpu.device_count()
        # prefer the discrete Arc card over an iGPU
        names = [torch.xpu.get_device_name(i) for i in range(n)]
        best = next((i for i, nm in enumerate(names) if "Arc" in nm or "B580" in nm), 0)
        torch.xpu.set_device(best)
        print(f"[device] xpu:{best} = {names[best]}")
        return f"xpu:{best}"
    if pref == "xpu":
        raise RuntimeError("XPU requested but torch.xpu is unavailable (check PyTorch XPU wheel + drivers)")
    warnings.warn("XPU not available; falling back to CPU (slow)")
    return "cpu"


def sync(device: str) -> None:
    import torch

    if device.startswith("xpu"):
        torch.xpu.synchronize()
