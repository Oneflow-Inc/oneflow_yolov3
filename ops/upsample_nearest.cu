#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpsampleNearestForward(const int64_t nthreads, const T* in_dptr,
                                       const int64_t channel_num, const int64_t height,
                                       const int64_t width, const int64_t new_height,
                                       const int64_t new_width, const float scale_h,
                                       const float scale_w, const bool align_corners, T* out_dptr) {
  const int64_t new_area = new_height * new_width;
  const int64_t channel_area = channel_num * height * width;
  const int64_t channel_new_area = channel_num * new_height * new_width;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t h = (index / new_width) % new_height;
    const int64_t w = index % new_width;
    const int64_t c = (index / new_area) % channel_num;
    const int64_t n = index / channel_new_area;
    const int64_t in_h = min((align_corners) ? static_cast<int64_t>(roundf(h * scale_h))
                                             : static_cast<int64_t>(floorf(h * scale_h)),
                             height - 1);
    const int64_t in_w = min((align_corners) ? static_cast<int64_t>(roundf(w * scale_w))
                                             : static_cast<int64_t>(floorf(w * scale_w)),
                             width - 1);
    out_dptr[index] = in_dptr[n * channel_area + (c * height + in_h) * width + in_w];
  }
}

template<typename T>
__global__ void UpsampleNearestBackward(const int64_t nthreads, const T* dy_dptr,
                                        const int64_t channel_num, const int64_t height,
                                        const int64_t width, const int64_t new_height,
                                        const int64_t new_width, const float scale_h,
                                        const float scale_w, const bool align_corners, T* dx_dptr) {
  const int64_t area = height * width;
  const int64_t new_area = new_height * new_width;
  const int64_t channel_area = channel_num * height * width;
  const int64_t channel_new_area = channel_num * new_height * new_width;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t h = (index / new_width) % new_height;
    const int64_t w = index % new_width;
    const int64_t c = (index / new_area) % channel_num;
    const int64_t n = index / channel_new_area;
    const int64_t in_h = min((align_corners) ? static_cast<int64_t>(roundf(h * scale_h))
                                             : static_cast<int64_t>(floorf(h * scale_h)),
                             height - 1);
    const int64_t in_w = min((align_corners) ? static_cast<int64_t>(roundf(w * scale_w))
                                             : static_cast<int64_t>(floorf(w * scale_w)),
                             width - 1);
    atomicAdd(dx_dptr + n * channel_area + (c * height + in_h) * width + in_w, dy_dptr[index]);
  }
}

}  // namespace

template<typename T>
class UpsampleNearestGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleNearestGPUKernel() = default;
  ~UpsampleNearestGPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_blob = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t scale = ctx->Attr<int32_t>("scale");
    const int64_t elem_cnt = y_blob->shape().elem_cnt();
    UpsampleNearestForward<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), 1024, 0, ctx->device_ctx()->cuda_stream()>>>(
            elem_cnt, x_blob->dptr<T>(), x_blob->shape().At(1), x_blob->shape().At(2),
            x_blob->shape().At(3), y_blob->shape().At(2), y_blob->shape().At(3), 1.f / scale,
            1.f / scale, false, y_blob->mut_dptr<T>());
  }
    bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

};

template<typename T>
class UpsampleNearestGradGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleNearestGradGPUKernel() = default;
  ~UpsampleNearestGradGPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    if (dx_blob == nullptr) { return; }
    Memset<DeviceType::kGPU>(ctx->device_ctx(), dx_blob->mut_dptr<T>(), 0,
                             dx_blob->shape().elem_cnt() * sizeof(T));
    const user_op::Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const int32_t scale = ctx->Attr<int32_t>("scale");
    const int64_t elem_cnt = dy_blob->shape().elem_cnt();
    UpsampleNearestBackward<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), 1024, 0, ctx->device_ctx()->cuda_stream()>>>(
            elem_cnt, dy_blob->dptr<T>(), dx_blob->shape().At(1), dx_blob->shape().At(2),
            dx_blob->shape().At(3), dy_blob->shape().At(2), dy_blob->shape().At(3), 1.f / scale,
            1.f / scale, false, dx_blob->mut_dptr<T>());
  }
    bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

};

#define REGISTER_UPSAMPLE_NEAREST_GPU_KERNEL(dtype)                                 \
  REGISTER_USER_KERNEL("upsample_nearest")                                          \
      .SetCreateFn<UpsampleNearestGPUKernel<dtype>>()                               \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) { return true; }); \
  REGISTER_USER_KERNEL("upsample_nearest_grad")                                     \
      .SetCreateFn<UpsampleNearestGPUKernel<dtype>>()                               \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) { return true; });

REGISTER_UPSAMPLE_NEAREST_GPU_KERNEL(float)

REGISTER_USER_OP("upsample_nearest")
    .Input("x")
    .Output("y")
    .Attr("scale", UserOpAttrType::kAtInt32)
    .Attr("data_format", UserOpAttrType::kAtString)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      const int32_t scale = ctx->Attr<int32_t>("scale");
      if (ctx->Attr<std::string>("data_format") != "channels_first" || x_shape->NumAxes() != 4) {
        LOG(FATAL) << "upsample_nearest only supports NCHW";
      }
      *y_shape =
          Shape({x_shape->At(0), x_shape->At(1), scale * x_shape->At(2), scale * x_shape->At(3)});
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("upsample_nearest_grad")
    .Input("dy")
    .Output("dx")
    .Attr("scale", UserOpAttrType::kAtInt32)
    .Attr("data_format", UserOpAttrType::kAtString)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);
      Shape* dx_shape = ctx->Shape4ArgNameAndIndex("dx", 0);
      const int32_t scale = ctx->Attr<int32_t>("scale");
      if (ctx->Attr<std::string>("data_format") != "channels_first"
          || dy_shape->NumAxes() != 4) {
        LOG(FATAL) << "upsample_nearest only supports NCHW";
      }
      *dx_shape = Shape(
          {dy_shape->At(0), dy_shape->At(1), dy_shape->At(2) / scale, dy_shape->At(3) / scale});
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("dy", 0), 0).Split(user_op::OpArg("dx", 0), 0).Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("upsample_nearest")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("upsample_nearest_grad")
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Output("dx")
                .Attr("scale", op.attr<int32_t>("scale"))
                .Attr("data_format", op.attr<std::string>("data_format"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
