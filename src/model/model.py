"""Backward-compatible exports for the refactored TabFM model."""

try:
    from .tabfm_model.activations import _gelu_tanh, get_activation
    from .tabfm_model.attention import MultiheadAttention, MultiheadAttentionBlock
    from .tabfm_model.cache import (
        ICLearningCache,
        QuantizedTensor,
        _QUANTIZATION_RANGES,
        _quantize_tensor,
        detach_cache,
        move_cache_to_device,
    )
    from .tabfm_model.embeddings import (
        CellEmbedder,
        ColEmbedding,
        MLP,
        OneHotAndLinear,
        RowInteraction,
    )
    from .tabfm_model.icl import ICLearning
    from .tabfm_model.normalisation import RMSNorm
    from .tabfm_model.positional import RoPE, rope_interleaved
    from .tabfm_model.race_context import RaceSetEncoder, RaceSetHead
    from .tabfm_model.tabfm import (
        _CACHE_FORMAT_VERSION,
        _COL_CHUNK_SIZE,
        _FFN_CHUNK_SIZE,
        _ROW_CHUNK_SIZE,
        TabFM,
    )
    from .tabfm_model.transformers import (
        Encoder,
        InducedSelfAttentionBlock,
        SetTransformer,
    )
    from .tabfm_model.validation import (
        _INTEGER_DTYPES,
        _validate_cache_depth,
        _validate_classification_labels,
        _validate_contiguous_context_labels,
        _validate_runtime_inputs,
    )
except ImportError:
    # Compatibility for callers importing this file as top-level ``model``.
    from tabfm_model.activations import _gelu_tanh, get_activation
    from tabfm_model.attention import MultiheadAttention, MultiheadAttentionBlock
    from tabfm_model.cache import (
    ICLearningCache,
    QuantizedTensor,
    _QUANTIZATION_RANGES,
    _quantize_tensor,
    detach_cache,
    move_cache_to_device,
    )
    from tabfm_model.embeddings import (
    CellEmbedder,
    ColEmbedding,
    MLP,
    OneHotAndLinear,
    RowInteraction,
    )
    from tabfm_model.icl import ICLearning
    from tabfm_model.normalisation import RMSNorm
    from tabfm_model.positional import RoPE, rope_interleaved
    from tabfm_model.race_context import RaceSetEncoder, RaceSetHead
    from tabfm_model.tabfm import (
    _CACHE_FORMAT_VERSION,
    _COL_CHUNK_SIZE,
    _FFN_CHUNK_SIZE,
    _ROW_CHUNK_SIZE,
    TabFM,
    )
    from tabfm_model.transformers import Encoder, InducedSelfAttentionBlock, SetTransformer
    from tabfm_model.validation import (
    _INTEGER_DTYPES,
    _validate_cache_depth,
    _validate_classification_labels,
    _validate_contiguous_context_labels,
    _validate_runtime_inputs,
    )

__all__ = [
    "TabFM", "RMSNorm", "RoPE", "rope_interleaved",
    "MultiheadAttention", "MultiheadAttentionBlock",
    "InducedSelfAttentionBlock", "Encoder", "SetTransformer", "MLP",
    "OneHotAndLinear", "CellEmbedder", "ColEmbedding", "RowInteraction",
    "QuantizedTensor", "ICLearningCache", "ICLearning", "RaceSetEncoder",
    "RaceSetHead", "move_cache_to_device", "detach_cache", "get_activation",
]
