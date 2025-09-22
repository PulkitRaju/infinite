from typing import Dict, Any
import functools
import torch

DATA_PARAMS: Dict[str, Any] = {}

def _unsqueeze_minibatch(minibatch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Add a batch dimension so Hugging Face models receive [1, L] tensors."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in minibatch.items():
        if isinstance(v, torch.Tensor) and v.dim() == 1:
            out[k] = v.unsqueeze(0)
        else:
            out[k] = v
    return out

def _squeeze_output(output):
    """Squeeze batch dim added in _unsqueeze_minibatch; supports tuples."""
    if isinstance(output, tuple):
        return tuple(_squeeze_output(o) for o in output)
    if isinstance(output, torch.Tensor) and output.dim() > 0 and output.size(0) == 1:
        return output.squeeze(0)
    return output

def ring_attn_manager(func):
    """Wrapper ensuring ring-attention code handles batched tensors."""
    @functools.wraps(func)
    def wrapper(self, minibatch, *args, **kwargs):
        minibatch_batched = _unsqueeze_minibatch(minibatch)
        output = func(self, minibatch_batched, *args, **kwargs)
        return _squeeze_output(output)
    return wrapper
