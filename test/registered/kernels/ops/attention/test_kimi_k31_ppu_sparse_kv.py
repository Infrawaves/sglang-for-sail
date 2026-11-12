"""PPU K3.1 sparse prefill and paged decode over one FP8 DSA KV cache."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.utils.common import is_sail
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not is_sail(), reason="PPU FlashMLA kernels are required")
def test_sparse_prefill_and_kv_decode_share_fp8_cache():
    from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
        dequantize_k_cache_paged,
    )
    from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache
    from sglang.srt.hardware_backend.ppu.attention.flash_mla import (
        FlashMLASchedMeta,
        flash_mla_sparse_fwd,
        flash_mla_with_kvcache,
    )

    torch.manual_seed(17)
    kv = (torch.randn(4, 64, 1, 576, device="cuda") * 0.25).to(torch.bfloat16)
    q = (torch.randn(16, 64, 576, device="cuda") * 0.25).to(torch.bfloat16)
    fp8_cache = quantize_k_cache(kv, pow2_scale=True)
    slots = torch.arange(1, 129, device="cuda", dtype=torch.int32)
    selected_kv = dequantize_k_cache_paged(fp8_cache, slots)
    topk = torch.arange(128, device="cuda", dtype=torch.int32)
    indices = topk.view(1, 1, 128).expand(16, 1, 128).contiguous()
    lengths = torch.full((16,), 128, device="cuda", dtype=torch.int32)
    scale = 576**-0.5

    prefill, _, _ = flash_mla_sparse_fwd(
        q, selected_kv, indices, scale, d_v=512, topk_length=lengths
    )
    keys = selected_kv[:, 0].float()
    reference = torch.softmax(q.float() @ keys.T * scale, dim=-1) @ keys[:, :512]
    torch.testing.assert_close(prefill.float(), reference, atol=0.02, rtol=0.02)

    decode, _ = flash_mla_with_kvcache(
        q[:1, None],
        fp8_cache,
        torch.empty((1, 0), device="cuda", dtype=torch.int32),
        torch.tensor([256], device="cuda", dtype=torch.int32),
        512,
        FlashMLASchedMeta(),
        softmax_scale=scale,
        indices=slots.view(1, 1, 128),
        topk_length=lengths[:1],
        is_fp8_kvcache=False,
        kv_format="V32",
    )
    torch.testing.assert_close(
        decode.reshape(64, 512).float(), reference[0], atol=0.02, rtol=0.02
    )


@pytest.mark.skipif(
    not is_sail() or torch.cuda.get_device_capability() != (8, 9),
    reason="quantized sparse prefill requires ZW-M890P",
)
@pytest.mark.parametrize("heads,has_prefix", [(12, False), (96, True)])
def test_quantized_prefill(heads, has_prefix):
    from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
        dequantize_k_cache_paged,
    )
    from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

    torch.manual_seed(17)
    kv = (torch.randn(2, 64, 1, 576, device="cuda") * 0.5).bfloat16()
    kv_cache = quantize_k_cache(kv, pow2_scale=True).view(-1, 1, 656)
    fresh = kv.view(-1, 1, 576)
    slots = None
    if has_prefix:
        # Two requests can reference the same physical slot in different ragged rows.
        slots = (torch.arange(128, device="cuda", dtype=torch.int32) * 17) % 97
        keys = dequantize_k_cache_paged(kv_cache, slots)
    else:
        keys = fresh

    q = (torch.randn(3, heads, 576, device="cuda") * 0.5).bfloat16()
    cols = torch.arange(128, dtype=torch.int32, device="cuda")
    indices = cols.roll(13).expand(len(q), -1).contiguous()
    indices[:, 11] = -1
    lengths = torch.tensor([32, 95, 128], dtype=torch.int32, device="cuda")
    layer = SimpleNamespace(scaling=192**-0.5, v_head_dim=512)
    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend.device_sm_major = torch.cuda.get_device_capability()[0]
    backend.kimi_k31 = True
    backend.attn_dcp_size = 1
    backend.forward_metadata = SimpleNamespace(
        page_table_1_flattened=slots, dsa_cache_seqlens_int32=lengths
    )
    result = backend._forward_flashmla_sparse_quantized(
        q, kv_cache, fresh[..., :512], fresh[..., 512:], indices, layer
    )

    keys = keys[:, 0].to(torch.float8_e4m3fn).float()[indices.clamp_min(0).long()]
    scores = (q.to(torch.float8_e4m3fn).float() @ keys.transpose(1, 2)) * layer.scaling
    valid = (indices >= 0) & (cols < lengths[:, None])
    weights = scores.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
    expected = weights @ keys[..., :512]
    assert result.shape == (len(q), heads, 512)
    assert result.dtype == torch.bfloat16
    assert (result.float() - expected).norm() < 0.04 * expected.norm()
