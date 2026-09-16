# RMS dynamic quantization Meta validation

Temporary branch: `validate/v0.23.0-rms-quant-meta`, based on `27d02d817`.

The existing `enable_custom_op()` initialization imports `ops.rms_quant_meta`
after the C++ extension. This branch overrides its RMS dynamic quantization
Meta with INT8 output and FP32 scales, retaining symbolic token dimensions.
The existing NPU kernel is unchanged; no C++ rebuild is needed. Restart all
workers after switching branches. This requires the tested PyTorch 2.10
`Library.impl(..., allow_override=True)` API.

For both direct FXRT and the experimental `gm_forward` backend, remove the
previous vLLM-only test override:

```bash
unset VLLM_GM_AUDIT_RMS_META
```

Keep the model's existing dummy compatibility setting: this correction does
not replace `VLLM_ASCEND_FXRT_DUMMY_QUANT`. Real W8A8 weights should not enable
dummy compatibility solely for this Meta fix.

After loading the usual CANN/vendor environment and selecting a free NPU:

```bash
python tests/e2e/pull_request/one_card/test_rms_quant_meta.py
```

This compares actual/Meta shapes and dtypes and exact eager/compiled outputs
for BF16 and FP16 at 8, 16 and 32 tokens. The fullgraph dynamic trace must not
insert a second dynamic quantization. It uses `gm.forward` to isolate tracing;
it does not establish full-model or FXRT execution accuracy.
