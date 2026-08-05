"""Activation functions used by the TabFM modules."""

import torch.nn.functional as F

def _gelu_tanh(x):
  # jax.nn.gelu defaults to the tanh approximation -> match it.
  return F.gelu(x, approximate="tanh")


def get_activation(name):
  # Activations are stored as module attributes (e.g. MLP.act), so they must be
  # picklable for AutoGluon/TabArena's pickle-based save. A module-level
  # function pickles by reference; a lambda would not.
  return {"relu": F.relu,
          "gelu": _gelu_tanh,
          "silu": F.silu}[name]

