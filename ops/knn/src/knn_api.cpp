// Python bindings for the batched k-nearest-neighbor operator.

#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "knn_gpu.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_batch", &knn_batch, "knn_batch");
    m.def("knn_batch_mlogk", &knn_batch_mlogk, "knn_batch_mlogk");
}
