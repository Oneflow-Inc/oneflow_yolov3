#include "oneflow/core/framework/framework.h"
#include "darknet.h"
#include <vector>

namespace oneflow {

namespace {

class DecodeOpKernelState final : public user_op::OpKernelState {
 public:
  explicit DecodeOpKernelState(std::string image_path_file) : image_path_file_(image_path_file) {
    char* train_images = new char[image_path_file_.length() + 1];
    strcpy(train_images, image_path_file_.c_str());
    list* plist = get_paths(train_images);
    paths_ = (char**)list_to_array(plist);
    dataset_size_ = plist->size;
  }
  ~DecodeOpKernelState() override = default;

  int32_t dataset_size() const { return dataset_size_; }
  char** paths() { return paths_; }

 private:
  std::string image_path_file_;
  int32_t dataset_size_;
  char** paths_;
};

}  // namespace

REGISTER_USER_OP("yolo_train_decoder")
    .Output("data")
    .Output("ground_truth")
    .Output("gt_valid_num")
    .Attr<int32_t>("batch_size")
    .Attr<int32_t>("image_height")
    .Attr<int32_t>("image_width")
    .Attr<int32_t>("classes")
    .Attr<int32_t>("num_boxes")
    .Attr<float>("hue")
    .Attr<float>("jitter")
    .Attr<float>("saturation")
    .Attr<float>("exposure")
    .Attr<std::string>("image_path_file")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* data_desc = ctx->TensorDesc4ArgNameAndIndex("data", 0);
      user_op::TensorDesc* ground_truth_desc = ctx->TensorDesc4ArgNameAndIndex("ground_truth", 0);
      user_op::TensorDesc* gt_valid_num_desc = ctx->TensorDesc4ArgNameAndIndex("gt_valid_num", 0);
      int32_t batch_size = ctx->Attr<int32_t>("batch_size");
      const int32_t image_height = ctx->Attr<int32_t>("image_height");
      const int32_t image_width = ctx->Attr<int32_t>("image_width");
      *data_desc->mut_shape() = Shape({batch_size, 3, image_height, image_width});
      *data_desc->mut_data_type() = DataType::kFloat;
      *ground_truth_desc->mut_shape() = Shape({batch_size, 90, 5});
      *ground_truth_desc->mut_data_type() = DataType::kFloat;
      *gt_valid_num_desc->mut_shape() = Shape({batch_size, 1});
      *gt_valid_num_desc->mut_data_type() = DataType::kInt32;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    });

class YoloTrainDecoderKernel final : public oneflow::user_op::OpKernel {
 public:
  YoloTrainDecoderKernel() = default;
  ~YoloTrainDecoderKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    std::shared_ptr<user_op::OpKernelState> reader(
        new DecodeOpKernelState(ctx->Attr<std::string>("image_path_file")));
    return reader;
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* dataset = dynamic_cast<DecodeOpKernelState*>(state);
    CHECK_NOTNULL(dataset);
    int imgs = 1;
    const int32_t batch_size = ctx->Attr<int32_t>("batch_size");
    const int32_t image_height = ctx->Attr<int32_t>("image_height");
    const int32_t image_width = ctx->Attr<int32_t>("image_width");
    const int32_t classes = ctx->Attr<int32_t>("classes");
    const float hue = ctx->Attr<float>("hue");
    const float jitter = ctx->Attr<float>("jitter");
    const float saturation = ctx->Attr<float>("saturation");
    const float exposure = ctx->Attr<float>("exposure");
    const int32_t num_boxes = ctx->Attr<int32_t>("num_boxes");

    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("data", 0);
    user_op::Tensor* ground_truth_blob = ctx->Tensor4ArgNameAndIndex("ground_truth", 0);
    user_op::Tensor* gt_valid_num_blob = ctx->Tensor4ArgNameAndIndex("gt_valid_num", 0);

    user_op::MultiThreadLoopInOpKernel(
        batch_size,
        [&out_blob, &ground_truth_blob, &gt_valid_num_blob, dataset, imgs, classes, hue, jitter,
         num_boxes, saturation, image_width, image_height, exposure, this](size_t i) {
          data dt = load_data_detection(imgs, dataset->paths(), dataset->dataset_size(),
                                        image_width, image_height, num_boxes, classes, jitter, hue,
                                        saturation, exposure);
          memcpy(out_blob->mut_dptr<char>() + i * out_blob->shape().Count(1) * sizeof(float),
                 dt.X.vals[0], out_blob->shape().Count(1) * sizeof(float));
          memcpy(ground_truth_blob->mut_dptr<char>()
                     + i * ground_truth_blob->shape().Count(1) * sizeof(float),
                 dt.y.vals[0], ground_truth_blob->shape().Count(1) * sizeof(float));
          for (int idx = 0; idx < ground_truth_blob->shape().At(1); idx++) {
            if (dt.y.vals[0][idx * 5 + 2] == 0 && dt.y.vals[0][idx * 5 + 3] == 0
                && dt.y.vals[0][idx * 5 + 4] == 0) {
              *(gt_valid_num_blob->mut_dptr<int32_t>() + i) = idx;
              break;
            }
          }
          free_data(dt);
        });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("yolo_train_decoder")
    .SetCreateFn<YoloTrainDecoderKernel>()
    .SetIsMatchedHob((user_op::HobDataType("data", 0) == DataType::kFloat)
                     & (user_op::HobDataType("ground_truth", 0) == DataType::kFloat)
                     & (user_op::HobDataType("gt_valid_num", 0) == DataType::kInt32))
    .SetInferTmpSizeFn([](const oneflow::user_op::InferContext*) { return 0; });

}  // namespace oneflow
