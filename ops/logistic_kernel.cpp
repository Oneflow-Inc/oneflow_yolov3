#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class LogisticKernel final : public user_op::OpKernel {
 public:
  LogisticKernel() = default;
  ~LogisticKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);
    NewKernelUtil<device_type>::Sigmoid(ctx->device_ctx(), x->shape().elem_cnt(), x->dptr<T>(),
                                        y->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_LOGISTIC_KERNEL(device, dtype)                                                 \
  REGISTER_USER_KERNEL("logistic")                                                              \
      .SetCreateFn<LogisticKernel<device, dtype>>()                                             \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                     \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))         \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_LOGISTIC_KERNEL(DeviceType::kCPU, float)
REGISTER_LOGISTIC_KERNEL(DeviceType::kCPU, double)
REGISTER_LOGISTIC_KERNEL(DeviceType::kGPU, float)
REGISTER_LOGISTIC_KERNEL(DeviceType::kGPU, double)

}  // namespace

}  // namespace oneflow
