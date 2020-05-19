#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include <iostream>

namespace oneflow {

namespace {

constexpr int kBlockSize = sizeof(int64_t) * 8;

template<typename T>
__host__ __device__ __forceinline__ T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template<typename T>
__host__ __device__ __forceinline__ T IoU(T const* const a, T const* const b, bool x1y1x2y2=true) {
  T interS = 0;
  T Sa = 0;
  T Sb = 0;
  if (x1y1x2y2){
    interS = max(min(a[2], b[2]) - max(a[0], b[0]) + 1, 0.f)
           * max(min(a[3], b[3]) - max(a[1], b[1]) + 1, 0.f);
    Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
    Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  } else {
    interS = max((min(a[0] + a[2] / 2, b[0] + b[2] / 2) - max(a[0] - a[2] / 2, b[0] - b[2] / 2)), 0.f)
      * max((min(a[1] + a[3] / 2, b[1] + b[3] / 2) - max(a[1] - a[3] / 2, b[1] - b[3] / 2)), 0.f);
    Sa = (a[2] * a[3]);
    Sb = (b[2] * b[3]);
  }
  return interS / (Sa + Sb - interS);
}

template<typename T>
__global__ void CalcSuppressionBitmaskMatrix(int num_boxes, float iou_threshold, const T* boxes,
                                             int64_t* suppression_bmask_matrix, const T* probs) {
  if (probs[0] == 0) { return; }
  const int row = blockIdx.y;
  const int col = blockIdx.x;

  const int row_size = min(num_boxes - row * kBlockSize, kBlockSize);
  const int col_size = min(num_boxes - col * kBlockSize, kBlockSize);

  __shared__ T block_boxes[kBlockSize * 4];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 4 + 0] = boxes[(kBlockSize * col + threadIdx.x) * 4 + 0];
    block_boxes[threadIdx.x * 4 + 1] = boxes[(kBlockSize * col + threadIdx.x) * 4 + 1];
    block_boxes[threadIdx.x * 4 + 2] = boxes[(kBlockSize * col + threadIdx.x) * 4 + 2];
    block_boxes[threadIdx.x * 4 + 3] = boxes[(kBlockSize * col + threadIdx.x) * 4 + 3];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = kBlockSize * row + threadIdx.x;
    const T* cur_box_ptr = boxes + cur_box_idx * 4;
    // int i = 0;
    int64_t bits = 0;
    int start = 0;
    bool x1y1x2y2=false;
    if (row == col) { start = threadIdx.x + 1; }
    for (int i = start; i < col_size; i++) {
      if (IoU(cur_box_ptr, block_boxes + i * 4, x1y1x2y2) > iou_threshold) { bits |= 1ll << i; }
    }
    const int col_blocks = CeilDiv<int>(num_boxes, kBlockSize);
    suppression_bmask_matrix[cur_box_idx * col_blocks + col] = bits;
  }
}

template<typename T>
__global__ void ScanSuppression(int num_boxes, int num_blocks, int num_keep,
                                int64_t* suppression_bmask, int8_t* keep_mask, const T* probs) {
  if (probs[0] == 0) { return; }
  extern __shared__ int64_t remv[];
  remv[threadIdx.x] = 0;
  __syncthreads();
  for (int i = 0; i < num_boxes; ++i) {
    int block_n = i / kBlockSize;
    int block_i = i % kBlockSize;
    if (!(remv[block_n] & (1ll << block_i))) {
      remv[threadIdx.x] |= suppression_bmask[i * num_blocks + threadIdx.x];
      if (threadIdx.x == block_n && num_keep > 0) {
        keep_mask[i] = 1;
        num_keep -= 1;
      }
    }
    __syncthreads();
  }
}

}  // namespace

template<typename T>
class YoloNmsGpuKernel final : public user_op::OpKernel {
 public:
  YoloNmsGpuKernel() = default;
  ~YoloNmsGpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override  {
    const user_op::Tensor* boxes_blob = ctx->Tensor4ArgNameAndIndex("bbox", 0);
    const user_op::Tensor* probs_blob = ctx->Tensor4ArgNameAndIndex("probs", 0);
    user_op::Tensor* keep_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_blob = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const T* boxes = boxes_blob->dptr<T>();
    const T* probs = probs_blob->dptr<T>();
    int8_t* keep = keep_blob->mut_dptr<int8_t>();
    int64_t* suppression_mask = tmp_blob->mut_dptr<int64_t>();

    //boxes dims = batch_dims+2
    int32_t batch_dims = ctx->Attr<int>("batch_dims");
    CHECK_EQ(batch_dims+2, boxes_blob->shape().NumAxes());
    const int num_boxes = boxes_blob->shape().At(batch_dims);
    const int batch_size = boxes_blob->shape().elem_cnt() / boxes_blob->shape().Count(batch_dims);
    int num_keep = ctx->Attr<int>("keep_n");
    if (num_keep <= 0 || num_keep > num_boxes) { num_keep = num_boxes; }
    const int num_blocks = CeilDiv<int>(num_boxes, kBlockSize);
    Memset<DeviceType::kGPU>(ctx->device_ctx(), suppression_mask, 0,
                             batch_size * num_boxes * num_blocks * sizeof(int64_t));
    Memset<DeviceType::kGPU>(ctx->device_ctx(), keep, 0, batch_size * num_boxes * sizeof(int8_t));

    dim3 blocks(num_blocks, num_blocks);
    dim3 threads(kBlockSize);
    FOR_RANGE(int64_t, idx, 0, batch_size) {
      CalcSuppressionBitmaskMatrix<<<blocks, threads, 0, ctx->device_ctx()->cuda_stream()>>>(
          num_boxes, ctx->Attr<float>("iou_threshold"), boxes + idx * boxes_blob->shape().Count(batch_dims), suppression_mask + idx * num_boxes * num_blocks, probs + idx * probs_blob->shape().Count(batch_dims));
      ScanSuppression<<<1, num_blocks, num_blocks, ctx->device_ctx()->cuda_stream()>>>(
          num_boxes, num_blocks, num_keep, suppression_mask + idx * num_boxes * num_blocks, keep+ idx * num_boxes, probs + idx * probs_blob->shape().Count(batch_dims));
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

};

#define REGISTER_YOLO_NMS_GPU_KERNEL(dtype)                                                           \
  REGISTER_USER_KERNEL("yolo_nms")                                                                    \
      .SetCreateFn<YoloNmsGpuKernel<dtype>>()                                                    \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                               \
        const user_op::TensorDesc* in_desc = ctx.TensorDesc4ArgNameAndIndex("bbox", 0);          \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);          \
        if (ctx.device_type() == DeviceType::kGPU && out_desc->data_type() == DataType::kInt8    \
            && in_desc->data_type() == GetDataType<dtype>::value) {                              \
          return true;                                                                           \
        }                                                                                        \
        return false;                                                                            \
      })                                                                                         \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                        \
        Shape* bbox_shape = ctx->Shape4ArgNameAndIndex("bbox", 0);                               \
        int32_t batch_dims= ctx->Attr<int>("batch_dims");                                       \
        int64_t batch_size = bbox_shape->elem_cnt() / bbox_shape->Count(batch_dims);               \
        int64_t num_boxes = bbox_shape->At(batch_dims);                                          \
        int64_t blocks = CeilDiv<int64_t>(num_boxes, kBlockSize);                                \
        return batch_size * num_boxes * blocks * sizeof(int64_t);                                \
      });

REGISTER_YOLO_NMS_GPU_KERNEL(float)
REGISTER_YOLO_NMS_GPU_KERNEL(double)

}  // namespace oneflow
