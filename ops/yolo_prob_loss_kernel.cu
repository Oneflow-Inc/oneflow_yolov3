#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "yolo_kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void CalcObjnessDiffGpu(const int32_t* valid_num_ptr, const int32_t* inds_ptr,
                                   const T* bbox_objness_ptr, T* bbox_objness_out_ptr,
                                   int32_t value) {
  size_t num;
  if (value == 1) {
    num = valid_num_ptr[0];
  } else {
    num = valid_num_ptr[1];
  }
  CUDA_1D_KERNEL_LOOP(i, num) {
    int32_t box_index = inds_ptr[i];
    bbox_objness_out_ptr[box_index] = bbox_objness_ptr[box_index] - value;
  }
}

template<typename T>
__global__ void CopyValidClsProbGpu(const int32_t* valid_num_ptr, const int32_t num_clsprobs,
                                    const int32_t* pos_inds_ptr, const T* bbox_clsprob_ptr,
                                    T* bbox_clsprob_out_ptr) {
  CUDA_1D_KERNEL_LOOP(index, valid_num_ptr[0] * num_clsprobs) {
    size_t i = index / num_clsprobs;
    size_t j = index % num_clsprobs;
    int32_t box_index = pos_inds_ptr[i];
    (bbox_clsprob_out_ptr + num_clsprobs * box_index)[j] =
        (bbox_clsprob_ptr + num_clsprobs * box_index)[j];
  }
}

template<typename T>
__global__ void CalcClsProbDiffGpu(const int32_t* valid_num_ptr, const int32_t num_clsprobs,
                                   const int32_t* pos_inds_ptr, const int32_t* pos_cls_label_ptr,
                                   T* bbox_clsprob_out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, valid_num_ptr[0]) {
    int32_t box_index = pos_inds_ptr[i];
    if (pos_cls_label_ptr[box_index] >= 0) {
      int32_t idx = num_clsprobs * box_index + pos_cls_label_ptr[box_index];
      bbox_clsprob_out_ptr[idx]--;
    }
  }
}

}  // namespace

template<typename T>
class YoloProbLossKernel final : public user_op::OpKernel {
 public:
  YoloProbLossKernel() = default;
  ~YoloProbLossKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* bbox_objness = ctx->Tensor4ArgNameAndIndex("bbox_objness", 0);
    const user_op::Tensor* bbox_clsprob = ctx->Tensor4ArgNameAndIndex("bbox_clsprob", 0);
    const user_op::Tensor* pos_inds = ctx->Tensor4ArgNameAndIndex("pos_inds", 0);
    const user_op::Tensor* neg_inds = ctx->Tensor4ArgNameAndIndex("neg_inds", 0);
    const user_op::Tensor* pos_cls_label = ctx->Tensor4ArgNameAndIndex("pos_cls_label", 0);
    const user_op::Tensor* valid_num = ctx->Tensor4ArgNameAndIndex("valid_num", 0);

    user_op::Tensor* bbox_objness_out = ctx->Tensor4ArgNameAndIndex("bbox_objness_out", 0);
    user_op::Tensor* bbox_clsprob_out = ctx->Tensor4ArgNameAndIndex("bbox_clsprob_out", 0);
    const int32_t num_classes = ctx->Attr<int32_t>("num_classes");
    Memset<DeviceType::kGPU>(ctx->device_ctx(), bbox_objness_out->mut_dptr(), 0,
                             bbox_objness_out->shape().elem_cnt() * sizeof(T));
    Memset<DeviceType::kGPU>(ctx->device_ctx(), bbox_clsprob_out->mut_dptr(), 0,
                             bbox_clsprob_out->shape().elem_cnt() * sizeof(T));

    const size_t pos_num = 50;
    const size_t neg_num = 20000;
    FOR_RANGE(int32_t, im_index, 0, bbox_objness->shape().At(0)) {
      const int32_t* pos_inds_ptr =
          pos_inds->dptr<int32_t>() + im_index * pos_inds->shape().Count(1);
      const int32_t* neg_inds_ptr =
          neg_inds->dptr<int32_t>() + im_index * neg_inds->shape().Count(1);
      const int32_t* pos_cls_label_ptr =
          pos_cls_label->dptr<int32_t>() + im_index * pos_cls_label->shape().Count(1);
      const int32_t* valid_num_ptr =
          valid_num->dptr<int32_t>() + im_index * valid_num->shape().Count(1);
      const T* bbox_objness_ptr =
          bbox_objness->dptr<T>() + im_index * bbox_objness->shape().Count(1);
      T* bbox_objness_out_ptr =
          bbox_objness_out->mut_dptr<T>() + im_index * bbox_objness_out->shape().Count(1);
      const T* bbox_clsprob_ptr =
          bbox_clsprob->dptr<T>() + im_index * bbox_clsprob->shape().Count(1);
      T* bbox_clsprob_out_ptr =
          bbox_clsprob_out->mut_dptr<T>() + im_index * bbox_clsprob_out->shape().Count(1);

      CalcObjnessDiffGpu<T><<<BlocksNum4ThreadsNum(pos_num), kCudaThreadsNumPerBlock, 0,
                              ctx->device_ctx()->cuda_stream()>>>(
          valid_num_ptr, pos_inds_ptr, bbox_objness_ptr, bbox_objness_out_ptr, 1);
      CalcObjnessDiffGpu<T><<<BlocksNum4ThreadsNum(neg_num), kCudaThreadsNumPerBlock, 0,
                              ctx->device_ctx()->cuda_stream()>>>(
          valid_num_ptr, neg_inds_ptr, bbox_objness_ptr, bbox_objness_out_ptr, 0);

      CopyValidClsProbGpu<T><<<BlocksNum4ThreadsNum(pos_num * num_classes), kCudaThreadsNumPerBlock,
                               0, ctx->device_ctx()->cuda_stream()>>>(
          valid_num_ptr, num_classes, pos_inds_ptr, bbox_clsprob_ptr, bbox_clsprob_out_ptr);
      CalcClsProbDiffGpu<T><<<BlocksNum4ThreadsNum(pos_num), kCudaThreadsNumPerBlock, 0,
                              ctx->device_ctx()->cuda_stream()>>>(
          valid_num_ptr, num_classes, pos_inds_ptr, pos_cls_label_ptr, bbox_clsprob_out_ptr);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_YOLO_PROB_LOSS_KERNEL(dtype)                                                    \
  REGISTER_USER_KERNEL("yolo_prob_loss")                                                         \
      .SetCreateFn<YoloProbLossKernel<dtype>>()                                                  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                            \
                       & (user_op::HobDataType("bbox_objness", 0) == GetDataType<dtype>::value)  \
                       & (user_op::HobDataType("bbox_clsprob", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](const oneflow::user_op::InferContext*) { return 0; });

REGISTER_YOLO_PROB_LOSS_KERNEL(float)
}  // namespace oneflow
