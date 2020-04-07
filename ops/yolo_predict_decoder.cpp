#include "oneflow/core/framework/framework.h"
#include "darknet.h"
#include <vector>

namespace oneflow {

namespace {
  void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
 {
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;
  while(std::string::npos != pos2)
  {
    v.push_back(s.substr(pos1, pos2-pos1));
 
    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if(pos1 != s.length())
    v.push_back(s.substr(pos1));
 }
}

REGISTER_USER_OP("yolo_predict_decoder")
    .Output("out")
    .Output("origin_image_info")
    .Attr("batch_size", UserOpAttrType::kAtInt32)
    .Attr("image_height", UserOpAttrType::kAtInt32)
    .Attr("image_width", UserOpAttrType::kAtInt32)
    .Attr("image_path_list", UserOpAttrType::kAtString)
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
    std::string image_path_list = ctx->GetAttr<std::string>("image_path_list");  
    std::string splitter=" ";
    SplitString(image_path_list, paths, splitter);
    dataset_size_ = paths.size();
  }
  YoloPredictDecoderKernel() = default;
  ~YoloPredictDecoderKernel() = default;

 private:
  int32_t batch_id_;
  int32_t dataset_size_;
  std::vector<std::string> paths;

  void Compute(oneflow::user_op::KernelContext* ctx) override {
    const int32_t batch_size = ctx->GetAttr<int32_t>("batch_size");
    const int32_t image_height = ctx->GetAttr<int32_t>("image_height");
    const int32_t image_width = ctx->GetAttr<int32_t>("image_width");
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* origin_image_info_blob = ctx->Tensor4ArgNameAndIndex("origin_image_info", 0);
    user_op::MultiThreadLoopInOpKernel(batch_size, [&out_blob, &origin_image_info_blob, batch_size, image_height, image_width, this](size_t i){
      int img_idx = (batch_id_ * batch_size + i) % dataset_size_;
      char *img_path = new char[paths[img_idx].length() + 1];
      strcpy(img_path, paths[img_idx].c_str());
      image im = load_image_color(img_path, 0, 0);
      delete [] img_path;
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
