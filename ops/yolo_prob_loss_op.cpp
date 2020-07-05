#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("yolo_prob_loss")
    .Input("bbox_objness")
    .Input("bbox_clsprob")
    .Input("pos_cls_label")
    .Input("pos_inds")
    .Input("neg_inds")
    .Input("valid_num")
    .Output("bbox_objness_out")
    .Output("bbox_clsprob_out")
    .Attr("num_classes", UserOpAttrType::kAtInt32)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      // input: bbox_objness : (n, r, 1)  r = h*w*3
      const user_op::TensorDesc* bbox_objness_desc =
          ctx->TensorDesc4ArgNameAndIndex("bbox_objness", 0);
      // input: bbox_clsprob (n, r, 80) T
      const user_op::TensorDesc* bbox_clsprob_desc =
          ctx->TensorDesc4ArgNameAndIndex("bbox_clsprob", 0);
      // input: pos_cls_label (n, r)
      const user_op::TensorDesc* pos_cls_label_desc =
          ctx->TensorDesc4ArgNameAndIndex("pos_cls_label", 0);
      // input: pos_inds (n, r) int32_t
      const user_op::TensorDesc* pos_inds_desc = ctx->TensorDesc4ArgNameAndIndex("pos_inds", 0);
      // input: neg_inds (n, r) int32_t
      const user_op::TensorDesc* neg_inds_desc = ctx->TensorDesc4ArgNameAndIndex("neg_inds", 0);

      const int32_t num_images = bbox_objness_desc->shape().At(0);
      CHECK_EQ(num_images, bbox_clsprob_desc->shape().At(0));
      CHECK_EQ(num_images, pos_cls_label_desc->shape().At(0));
      CHECK_EQ(num_images, pos_inds_desc->shape().At(0));
      CHECK_EQ(num_images, neg_inds_desc->shape().At(0));
      const int32_t num_boxes = bbox_objness_desc->shape().At(1);
      const int32_t num_clsprobs = ctx->Attr<int32_t>("num_classes");
      CHECK_EQ(num_boxes, pos_cls_label_desc->shape().At(1));
      CHECK_EQ(num_boxes, pos_inds_desc->shape().At(1));
      CHECK_EQ(num_boxes, neg_inds_desc->shape().At(1));
      CHECK_EQ(1, bbox_objness_desc->shape().At(2));
      CHECK_EQ(num_clsprobs, bbox_clsprob_desc->shape().At(2));

      // output: bbox_objness_out (n, r, 1)
      *ctx->TensorDesc4ArgNameAndIndex("bbox_objness_out", 0) = *bbox_objness_desc;
      // output: bbox_clsprob_out (n, r, 80)
      *ctx->TensorDesc4ArgNameAndIndex("bbox_clsprob_out", 0) = *bbox_clsprob_desc;

      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* pos_cls_label = GetInputArgModifierFn("pos_cls_label", 0);
      pos_cls_label->set_requires_grad(false);
      user_op::InputArgModifier* pos_inds = GetInputArgModifierFn("pos_inds", 0);
      pos_inds->set_requires_grad(false);
      user_op::InputArgModifier* neg_inds = GetInputArgModifierFn("neg_inds", 0);
      neg_inds->set_requires_grad(false);
      user_op::InputArgModifier* valid_num = GetInputArgModifierFn("valid_num", 0);
      valid_num->set_requires_grad(false);
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("bbox_objness_out", 0) =
          *ctx->BatchAxis4ArgNameAndIndex("bbox_objness", 0);
      *ctx->BatchAxis4ArgNameAndIndex("bbox_clsprob_out", 0) =
          *ctx->BatchAxis4ArgNameAndIndex("bbox_clsprob", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("bbox_objness", 0), 0)
          .Split(user_op::OpArg("bbox_clsprob", 0), 0)
          .Split(user_op::OpArg("pos_cls_label", 0), 0)
          .Split(user_op::OpArg("pos_inds", 0), 0)
          .Split(user_op::OpArg("neg_inds", 0), 0)
          .Split(user_op::OpArg("valid_num", 0), 0)
          .Split(user_op::OpArg("bbox_objness_out", 0), 0)
          .Split(user_op::OpArg("bbox_clsprob_out", 0), 0)
          .Build();

      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("yolo_prob_loss")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("bbox_objness", 0)) {
        user_op::UserOpConfWrapper bbox_objness_grad_op =
            user_op::UserOpConfWrapperBuilder(op.op_name() + "_bbox_objness_grad")
                .Op("multiply")
                .Input("x", op.GetGradTensorWithOpOutput("bbox_objness_out", 0))
                .Input("y", op.output("bbox_objness_out", 0))
                .Output("out")
                .Build();
        op.BindGradTensorWithOpInput(bbox_objness_grad_op.output("out", 0), "bbox_objness", 0);
        AddOp(bbox_objness_grad_op);
      }
      if (op.NeedGenGradTensor4OpInput("bbox_clsprob", 0)) {
        user_op::UserOpConfWrapper bbox_clsprob_grad_op =
            user_op::UserOpConfWrapperBuilder(op.op_name() + "_bbox_clsprob_grad")
                .Op("multiply")
                .Input("x", op.GetGradTensorWithOpOutput("bbox_clsprob_out", 0))
                .Input("y", op.output("bbox_clsprob_out", 0))
                .Output("out")
                .Build();
        op.BindGradTensorWithOpInput(bbox_clsprob_grad_op.output("out", 0), "bbox_clsprob", 0);
        AddOp(bbox_clsprob_grad_op);
      }
    });
}  // namespace oneflow
