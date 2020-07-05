#include "oneflow/core/framework/framework.h"
#include "darknet.h"
#include <vector>

namespace oneflow {

namespace {

class DecodeOpKernelState final : public user_op::OpKernelState {
 public:
  explicit DecodeOpKernelState(user_op::KernelInitContext* ctx) {
    image_paths_ = ctx->Attr<std::vector<std::string>>("image_paths");
    dataset_size_ = image_paths_.size();
    batch_id_ = 0;
  }
  ~DecodeOpKernelState() override = default;

  int32_t batch_id() const { return batch_id_; }
  int32_t dataset_size() const { return dataset_size_; }
  void set_batch_id(const int32_t batch_id) { batch_id_ = batch_id; }
  std::string path(int32_t idx) { return image_paths_[idx]; }

 private:
  int32_t batch_id_;
  int32_t dataset_size_;
  std::vector<std::string> image_paths_;
};

}  // namespace

REGISTER_USER_OP("yolo_predict_decoder")
    .Output("out")
    .Output("origin_image_info")
    .Attr("batch_size", UserOpAttrType::kAtInt32)
    .Attr("image_height", UserOpAttrType::kAtInt32)
    .Attr("image_width", UserOpAttrType::kAtInt32)
    .Attr("image_paths", UserOpAttrType::kAtListString)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      Shape* origin_image_info_shape = ctx->Shape4ArgNameAndIndex("origin_image_info", 0);
      const int32_t batch_size = ctx->Attr<int32_t>("batch_size");
      const int32_t image_height = ctx->Attr<int32_t>("image_height");
      const int32_t image_width = ctx->Attr<int32_t>("image_width");
      *out_shape = Shape({batch_size, 3, image_height, image_width});
      *origin_image_info_shape = Shape({batch_size, 2});
      *ctx->Dtype4ArgNameAndIndex("out", 0) = DataType::kFloat;
      *ctx->Dtype4ArgNameAndIndex("origin_image_info", 0) = DataType::kInt32;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    });

class YoloPredictDecoderKernel final : public oneflow::user_op::OpKernel {
 public:
  YoloPredictDecoderKernel() = default;
  ~YoloPredictDecoderKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    // return std::make_shared<DecodeOpKernelState>(ctx);
    std::shared_ptr<user_op::OpKernelState> reader(new DecodeOpKernelState(ctx));
    return reader;
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* dataset = dynamic_cast<DecodeOpKernelState*>(state);
    CHECK_NOTNULL(dataset);
    const int32_t batch_size = ctx->Attr<int32_t>("batch_size");
    const int32_t image_height = ctx->Attr<int32_t>("image_height");
    const int32_t image_width = ctx->Attr<int32_t>("image_width");
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* origin_image_info_blob = ctx->Tensor4ArgNameAndIndex("origin_image_info", 0);
    user_op::MultiThreadLoopInOpKernel(
        batch_size, [&out_blob, &origin_image_info_blob, dataset, batch_size, image_height,
                     image_width, this](size_t i) {
          int img_idx = (dataset->batch_id() * batch_size + i) % dataset->dataset_size();
          std::string image_path = dataset->path(img_idx);
          char* img_path = new char[image_path.length() + 1];
          strcpy(img_path, image_path.c_str());
          image im = load_image_color(img_path, 0, 0);
          delete[] img_path;
          image sized = letterbox_image(im, image_height, image_width);
          *(origin_image_info_blob->mut_dptr<int32_t>()
            + i * origin_image_info_blob->shape().Count(1)) = im.h;
          *(origin_image_info_blob->mut_dptr<int32_t>()
            + i * origin_image_info_blob->shape().Count(1) + 1) = im.w;
          memcpy(out_blob->mut_dptr() + i * out_blob->shape().Count(1) * sizeof(float), sized.data,
                 out_blob->shape().Count(1) * sizeof(float));
          free_image(im);
          free_image(sized);
        });
    dataset->set_batch_id(dataset->batch_id() + 1);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("yolo_predict_decoder")
    .SetCreateFn<YoloPredictDecoderKernel>()
    .SetIsMatchedHob((user_op::HobDataType("out", 0) == DataType::kFloat)
                     & (user_op::HobDataType("origin_image_info", 0) == DataType::kInt32))
    .SetInferTmpSizeFn([](const oneflow::user_op::InferContext*) { return 0; });

}  // namespace oneflow
