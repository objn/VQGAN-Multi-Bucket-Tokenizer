try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

_lpips_model = None


def get_lpips_model(device):
    """Lazily construct and cache a frozen LPIPS (VGG) model on `device`."""
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="vgg").to(device)
        _lpips_model.eval()
        for p in _lpips_model.parameters():
            p.requires_grad_(False)
    return _lpips_model
