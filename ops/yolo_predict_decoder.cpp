#include "oneflow/core/framework/framework.h"
#include "darknet.h"

namespace oneflow {

REGISTER_USER_OP("yolo_predict_decoder")
    .Output("out")
    .Output("origin_image_info")
    .Attr("batch_size", UserOpAttrType::kAtInt32)
    .Attr("image_height", UserOpAttrType::kAtInt32)
    .Attr("image_width", UserOpAttrType::kAtInt32)
    .Attr("image_list_path", UserOpAttrType::kAtString)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      Shape* origin_image_info_shape = ctx->Shape4ArgNameAndIndex("origin_image_info", 0);
      const int32_t batch_size = ctx->GetAttr<int32_t>("batch_size");
      const int32_t image_height = ctx->GetAttr<int32_t>("image_height");
      const int32_t image_width = ctx->GetAttr<int32_t>("image_width");
      *out_shape = Shape({batch_size, 3, image_height, image_width});
      *origin_image_info_shape = Shape({batch_size, 2});
      *ctx->Dtype4ArgNameAndIndex("out", 0) = DataType::kFloat;
      *ctx->Dtype4ArgNameAndIndex("origin_image_info", 0) = DataType::kInt32;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      SbpSignatureBuilder()
          .Split(ctx->outputs(), 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

class YoloPredictDecoderKernel final : public oneflow::user_op::OpKernel {
 public:
  YoloPredictDecoderKernel(oneflow::user_op::KernelInitContext* ctx) : oneflow::user_op::OpKernel(ctx) {
    batch_id_ = 0;
    std::string image_list_path = ctx->GetAttr<std::string>("image_list_path");
    char *cstr = new char[image_list_path.length() + 1];
    strcpy(cstr, image_list_path.c_str());
    list* plist = get_paths(cstr);
    delete [] cstr;
    dataset_size_=plist->size;
    paths = (char **)list_to_array(plist);
  }
  YoloPredictDecoderKernel() = default;
  ~YoloPredictDecoderKernel() = default;

 private:
  int32_t batch_id_;
  int32_t dataset_size_;
  char **paths;

  void Compute(oneflow::user_op::KernelContext* ctx) override {
    const int32_t batch_size = ctx->GetAttr<int32_t>("batch_size");
    const int32_t image_height = ctx->GetAttr<int32_t>("image_height");
    const int32_t image_width = ctx->GetAttr<int32_t>("image_width");
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* origin_image_info_blob = ctx->Tensor4ArgNameAndIndex("origin_image_info", 0);
    user_op::MultiThreadLoopInOpKernel(batch_size, [&out_blob, &origin_image_info_blob, batch_size, image_height, image_width, this](size_t i){
      int img_idx = (batch_id_ * batch_size + i) % dataset_size_;
      image im = load_image_color(paths[img_idx], 0, 0);
      image sized = letterbox_image(im, image_height, image_width);
      *(origin_image_info_blob->mut_dptr<int32_t>()+ i * origin_image_info_blob->shape().Count(1)) = im.h;
      *(origin_image_info_blob->mut_dptr<int32_t>()+ i * origin_image_info_blob->shape().Count(1) + 1) = im.w;
      memcpy(out_blob->mut_dptr()+ i * out_blob->shape().Count(1) * sizeof(float), sized.data, out_blob->shape().Count(1) * sizeof(float));      
      free_image(im);
      free_image(sized);
    });
    batch_id_++; 
  }
};

REGISTER_USER_KERNEL("yolo_predict_decoder")
    .SetCreateFn([](oneflow::user_op::KernelInitContext* ctx) { return new YoloPredictDecoderKernel(ctx); })
    .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) { return true; })
    .SetInferTmpSizeFn([](const oneflow::user_op::InferContext*) { return 0; });

}  // namespace oneflow
