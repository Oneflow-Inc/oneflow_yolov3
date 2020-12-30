#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("yolo_nms")
    .Input("bbox")
    .Input("probs")
    .Output("out")
    .Attr<float>("iou_threshold")
    .Attr<int32_t>("keep_n")
    .Attr<int32_t>("batch_dims")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* bbox_shape = ctx->Shape4ArgNameAndIndex("bbox", 0);
      DimVector dim_vec(bbox_shape->NumAxes() - 1);
      FOR_RANGE(size_t, i, 0, dim_vec.size()) { dim_vec[i] = bbox_shape->At(i); }
      *ctx->Shape4ArgNameAndIndex("out", 0) = Shape(dim_vec);
      *ctx->Dtype4ArgNameAndIndex("out", 0) = DataType::kInt8;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("bbox", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("bbox", 0), 0)
          .Split(user_op::OpArg("probs", 0), 0)
          .Split(user_op::OpArg("out", 0), 0)
          .Build();

      return Maybe<void>::Ok();
    });

}  // namespace oneflow
