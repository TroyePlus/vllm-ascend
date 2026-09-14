// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/extension.h>

#include "aclnn_common.h"

namespace {

at::Tensor EngramFetchLegacy(const at::Tensor &context, const at::Tensor &indices, int64_t hiddenSize,
                             int64_t numEntries)
{
    TORCH_CHECK(context.device().type() == c10::DeviceType::PrivateUse1, "Engram context must be on NPU");
    TORCH_CHECK(indices.device().type() == c10::DeviceType::PrivateUse1, "Engram indices must be on NPU");
    TORCH_CHECK(indices.dim() == 1, "Engram indices must be 1D");
    TORCH_CHECK(indices.scalar_type() == at::kInt, "Engram indices must be int32");
    TORCH_CHECK(hiddenSize > 0, "Engram hidden_size must be positive");
    TORCH_CHECK(numEntries >= 0, "Engram num_entries_per_rank must be non-negative");

    const int64_t numTokens = indices.size(0);
    auto fetched = at::empty(
        {numTokens, hiddenSize}, at::TensorOptions().dtype(at::kBFloat16).device(indices.device()));
    if (numTokens == 0) {
        return fetched;
    }

    auto fetchedSf = at::empty({0}, at::TensorOptions().dtype(at::kFloat).device(indices.device()));
    aclTensor *nullTensor = nullptr;
    int64_t zero = 0;
    int64_t sfTableAddr = 0;
    ACLNN_CMD(aclnnEngramFetch, context, indices, nullTensor, fetched, nullTensor, nullTensor, nullTensor, nullTensor,
              nullTensor, fetchedSf, hiddenSize, numEntries, zero, zero, zero, sfTableAddr);
    return fetched;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("engram_fetch", &EngramFetchLegacy, pybind11::arg("context"), pybind11::arg("indices"),
          pybind11::arg("hidden_size"), pybind11::arg("num_entries_per_rank"));
}
