#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("yolo_detect")
    .Input("bbox")
    .Input("probs")
    .Input("origin_image_info")
    .Output("out_bbox")
    .Output("out_probs")
    .Output("valid_num")
    .Attr("image_height", UserOpAttrType::kAtInt32)
    .Attr("image_width", UserOpAttrType::kAtInt32)
    .Attr("layer_height", UserOpAttrType::kAtInt32)
    .Attr("layer_width", UserOpAttrType::kAtInt32)
    .Attr("prob_thresh", UserOpAttrType::kAtFloat)
    .Attr("num_classes", UserOpAttrType::kAtInt32)
    .Attr("anchor_boxes", UserOpAttrType::kAtListInt32)
    .Attr("max_out_boxes", UserOpAttrType::kAtInt32)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      // bbox : (n, h*w*3, 4) probs : (n, h*w*3, 81)
      // out_bbox : (n, max_out_boxes, 4) out_probs : (n, max_out_boxes, 81)
      const Shape* bbox_shape = ctx->Shape4ArgNameAndIndex("bbox", 0);
      const Shape* probs_shape = ctx->Shape4ArgNameAndIndex("probs", 0);
      const Shape* origin_image_info_shape = ctx->Shape4ArgNameAndIndex("origin_image_info", 0);
      CHECK_EQ_OR_RETURN(bbox_shape->At(0), origin_image_info_shape->At(0));
      CHECK_EQ_OR_RETURN(bbox_shape->NumAxes(), probs_shape->NumAxes());
      CHECK_EQ_OR_RETURN(bbox_shape->At(1), probs_shape->At(1));
      CHECK_EQ_OR_RETURN(bbox_shape->At(2), 4);
      CHECK_EQ_OR_RETURN(probs_shape->At(2), ctx->Attr<int32_t>("num_classes") + 1);
      Shape* out_bbox_shape = ctx->Shape4ArgNameAndIndex("out_bbox", 0);
      Shape* out_probs_shape = ctx->Shape4ArgNameAndIndex("out_probs", 0);
      Shape* valid_num_shape = ctx->Shape4ArgNameAndIndex("valid_num", 0);
      *out_bbox_shape = *bbox_shape;
      *out_probs_shape = *probs_shape;
      *ctx->Dtype4ArgNameAndIndex("out_bbox", 0) = *ctx->Dtype4ArgNameAndIndex("bbox", 0);
      *ctx->Dtype4ArgNameAndIndex("out_probs", 0) = *ctx->Dtype4ArgNameAndIndex("probs", 0);
      *ctx->Dtype4ArgNameAndIndex("valid_num", 0) =
          *ctx->Dtype4ArgNameAndIndex("origin_image_info", 0);
      int32_t max_out_boxes =
          ctx->Attr<int32_t>("max_out_boxes");  // todo, in python set, optional->required
      CHECK_GT_OR_RETURN(max_out_boxes, 0);
      CHECK_LE_OR_RETURN(max_out_boxes, bbox_shape->At(1));
      out_bbox_shape->Set(1, max_out_boxes);
      out_probs_shape->Set(1, max_out_boxes);
      valid_num_shape->Set(0, bbox_shape->At(0));
      valid_num_shape->Set(1, 2);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out_bbox", 0) = *ctx->BatchAxis4ArgNameAndIndex("bbox", 0);
      *ctx->BatchAxis4ArgNameAndIndex("out_probs", 0) = *ctx->BatchAxis4ArgNameAndIndex("bbox", 0);
      *ctx->BatchAxis4ArgNameAndIndex("valid_num", 0) = *ctx->BatchAxis4ArgNameAndIndex("bbox", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("bbox", 0), 0)
          .Split(user_op::OpArg("probs", 0), 0)
          .Split(user_op::OpArg("origin_image_info", 0), 0)
          .Split(user_op::OpArg("out_bbox", 0), 0)
          .Split(user_op::OpArg("out_probs", 0), 0)
          .Split(user_op::OpArg("valid_num", 0), 0)
          .Build();

      return Maybe<void>::Ok();
    });

}  // namespace oneflow
