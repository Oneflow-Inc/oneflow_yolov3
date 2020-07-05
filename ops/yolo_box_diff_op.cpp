#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("yolo_box_diff")
    .Input("bbox")
    .Input("gt_boxes")
    .Input("gt_labels")
    .Input("gt_valid_num")
    .Output("bbox_loc_diff")
    .Output("pos_inds")
    .Output("pos_cls_label")
    .Output("neg_inds")
    .Output("valid_num")
    .Attr("image_height", UserOpAttrType::kAtInt32)
    .Attr("image_width", UserOpAttrType::kAtInt32)
    .Attr("layer_height", UserOpAttrType::kAtInt32)
    .Attr("layer_width", UserOpAttrType::kAtInt32)
    .Attr("ignore_thresh", UserOpAttrType::kAtFloat)
    .Attr("truth_thresh", UserOpAttrType::kAtFloat)
    .Attr("anchor_boxes", UserOpAttrType::kAtListInt32)
    .Attr("box_mask", UserOpAttrType::kAtListInt32)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      // input: bbox : (n, r, 4)  r = h*w*3
      const user_op::TensorDesc* bbox_desc = ctx->TensorDesc4ArgNameAndIndex("bbox", 0);
      // input: gt_boxes (n, g, 4) T
      const user_op::TensorDesc* gt_boxes_desc = ctx->TensorDesc4ArgNameAndIndex("gt_boxes", 0);
      // input: gt_labels (n, g) int32_t
      const user_op::TensorDesc* gt_labels_desc = ctx->TensorDesc4ArgNameAndIndex("gt_labels", 0);
      const user_op::TensorDesc* gt_valid_num_desc =
          ctx->TensorDesc4ArgNameAndIndex("gt_valid_num", 0);

      const int64_t num_images = bbox_desc->shape().At(0);
      CHECK_EQ_OR_RETURN(num_images, gt_boxes_desc->shape().At(0));
      CHECK_EQ_OR_RETURN(num_images, gt_labels_desc->shape().At(0));
      const int64_t num_boxes = bbox_desc->shape().At(1);
      const int64_t max_num_gt_boxes = gt_boxes_desc->shape().At(1);
      CHECK_EQ_OR_RETURN(max_num_gt_boxes, gt_labels_desc->shape().At(1));
      CHECK_EQ(bbox_desc->data_type(), gt_boxes_desc->data_type());

      // output: bbox_loc_diff (n, r, 4)
      user_op::TensorDesc* bbox_loc_diff_desc = ctx->TensorDesc4ArgNameAndIndex("bbox_loc_diff", 0);
      *bbox_loc_diff_desc->mut_shape() = Shape({num_images, num_boxes, 4});
      *bbox_loc_diff_desc->mut_data_type() = bbox_desc->data_type();
      // output: pos_cls_label (n, r)
      user_op::TensorDesc* pos_cls_label_desc = ctx->TensorDesc4ArgNameAndIndex("pos_cls_label", 0);
      *pos_cls_label_desc->mut_shape() = Shape({num_images, num_boxes});
      *pos_cls_label_desc->mut_data_type() = DataType::kInt32;
      // output: pos_inds (n, r) dynamic
      user_op::TensorDesc* pos_inds_desc = ctx->TensorDesc4ArgNameAndIndex("pos_inds", 0);
      *pos_inds_desc->mut_shape() = Shape({num_images, num_boxes});
      *pos_inds_desc->mut_data_type() = DataType::kInt32;
      // output: neg_inds (n, r) dynamic
      user_op::TensorDesc* neg_inds_desc = ctx->TensorDesc4ArgNameAndIndex("neg_inds", 0);
      *neg_inds_desc->mut_shape() = Shape({num_images, num_boxes});
      *neg_inds_desc->mut_data_type() = DataType::kInt32;

      user_op::TensorDesc* valid_num_desc = ctx->TensorDesc4ArgNameAndIndex("valid_num", 0);
      *valid_num_desc->mut_shape() = Shape({num_images, 2});
      *valid_num_desc->mut_data_type() = DataType::kInt32;

      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* gt_boxes = GetInputArgModifierFn("gt_boxes", 0);
      gt_boxes->set_requires_grad(false);
      user_op::InputArgModifier* gt_labels = GetInputArgModifierFn("gt_labels", 0);
      gt_labels->set_requires_grad(false);
      user_op::InputArgModifier* gt_valid_num = GetInputArgModifierFn("gt_valid_num", 0);
      gt_valid_num->set_requires_grad(false);
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("bbox_loc_diff", 0) =
          *ctx->BatchAxis4ArgNameAndIndex("bbox", 0);
      *ctx->BatchAxis4ArgNameAndIndex("pos_cls_label", 0) =
          *ctx->BatchAxis4ArgNameAndIndex("bbox", 0);
      *ctx->BatchAxis4ArgNameAndIndex("pos_inds", 0) = *ctx->BatchAxis4ArgNameAndIndex("bbox", 0);
      *ctx->BatchAxis4ArgNameAndIndex("neg_inds", 0) = *ctx->BatchAxis4ArgNameAndIndex("bbox", 0);
      *ctx->BatchAxis4ArgNameAndIndex("valid_num", 0) = *ctx->BatchAxis4ArgNameAndIndex("bbox", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("bbox", 0), 0)
          .Split(user_op::OpArg("gt_boxes", 0), 0)
          .Split(user_op::OpArg("gt_labels", 0), 0)
          .Split(user_op::OpArg("gt_valid_num", 0), 0)
          .Split(user_op::OpArg("bbox_loc_diff", 0), 0)
          .Split(user_op::OpArg("pos_inds", 0), 0)
          .Split(user_op::OpArg("pos_cls_label", 0), 0)
          .Split(user_op::OpArg("neg_inds", 0), 0)
          .Split(user_op::OpArg("valid_num", 0), 0)
          .Build();

      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("yolo_box_diff")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("bbox", 0)) {
        user_op::UserOpConfWrapper bbox_grad_op =
            user_op::UserOpConfWrapperBuilder(op.op_name() + "_grad")
                .Op("multiply")
                .Input("x", op.GetGradTensorWithOpOutput("bbox_loc_diff", 0))
                .Input("y", op.output("bbox_loc_diff", 0))
                .Output("out")
                .Build();
        op.BindGradTensorWithOpInput(bbox_grad_op.output("out", 0), "bbox", 0);
        AddOp(bbox_grad_op);
      }
    });

}  // namespace oneflow
