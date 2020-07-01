#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "yolo_kernel_util.cuh"

namespace oneflow {

namespace {

class TmpBufferManager final {
 public:
  TmpBufferManager(int32_t capacity, void* ptr, const int32_t box_num, const int32_t anchor_boxes_elem_cnt)
      : capacity_{capacity},
        select_inds_elem_cnt_{box_num},
        anchor_boxes_tmp_elem_cnt_{anchor_boxes_elem_cnt} {
    const int32_t select_inds_aligned_bytes = GetCudaAlignedSize(select_inds_elem_cnt_ * sizeof(int32_t));
    const int32_t anchor_boxes_tmp_aligned_bytes = GetCudaAlignedSize(anchor_boxes_tmp_elem_cnt_ * sizeof(int32_t));
    select_inds_ptr_ = reinterpret_cast<int32_t*>(ptr);
    anchor_boxes_tmp_ptr_ = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(select_inds_ptr_)
                                              + select_inds_aligned_bytes);
    temp_storage_ptr_ =
        reinterpret_cast<void*>(reinterpret_cast<char*>(anchor_boxes_tmp_ptr_) + anchor_boxes_tmp_aligned_bytes);
    temp_storage_bytes_ = capacity_ - select_inds_aligned_bytes - anchor_boxes_tmp_aligned_bytes;
    CHECK_GE(temp_storage_bytes_, 0);
  }
  OF_DISALLOW_COPY_AND_MOVE(TmpBufferManager);
  ~TmpBufferManager() = default;

  int32_t* SelectIndsPtr() const { return select_inds_ptr_; }
  int32_t* AnchorBoxesTmpPtr() const { return anchor_boxes_tmp_ptr_; }
  void* TempStoragePtr() const { return temp_storage_ptr_; }

  int32_t SelectIndsElemCnt() const { return select_inds_elem_cnt_; }
  int32_t AnchorBoxesTmpElemCnt() const { return anchor_boxes_tmp_elem_cnt_; }
  int32_t TempStorageBytes() const { return temp_storage_bytes_; }

 private:
  int32_t capacity_;

  int32_t* select_inds_ptr_;
  int32_t* anchor_boxes_tmp_ptr_;
  void* temp_storage_ptr_;

  int32_t select_inds_elem_cnt_;
  int32_t anchor_boxes_tmp_elem_cnt_;
  int32_t temp_storage_bytes_;
};

template<typename T>
__global__ void SetOutProbs(const int32_t probs_num, const T* probs_ptr,
                            const int32_t* select_inds_ptr, const int32_t* valid_num_ptr,
                            T* out_probs_ptr, const float prob_thresh, const int32_t max_out_boxes) {
  assert(valid_num_ptr[0] < max_out_boxes);
  const int index_num = valid_num_ptr[0];
  CUDA_1D_KERNEL_LOOP(i, index_num * probs_num) {
    const int32_t select_index = i / probs_num;
    const int32_t probs_index = i % probs_num;
    const int32_t box_index = select_inds_ptr[select_index];
    if (probs_index == 0) {
      out_probs_ptr[select_index * probs_num + probs_index] =
          probs_ptr[box_index * probs_num + probs_index];
    } else {
      T cls_prob =
          probs_ptr[box_index * probs_num + probs_index] * probs_ptr[box_index * probs_num + 0];
      out_probs_ptr[select_index * probs_num + probs_index] = cls_prob > prob_thresh ? cls_prob : 0;
    }
  }
}

template<typename T>
__global__ void SetOutBoxes(const T* bbox_ptr, const int32_t* origin_image_info_ptr,
                            const int32_t* select_inds_ptr, const int32_t* valid_num_ptr,
                            const int32_t* anchor_boxes_ptr, T* out_bbox_ptr,
                            const int32_t layer_height, const int32_t layer_width,
                            const int32_t layer_nbox, const int32_t image_height,
                            const int32_t image_width) {
  int32_t new_w = 0;
  int32_t new_h = 0;
  if (((float)image_width / origin_image_info_ptr[1])
      < ((float)image_height / origin_image_info_ptr[0])) {
    new_w = image_width;
    new_h = (origin_image_info_ptr[0] * image_width) / origin_image_info_ptr[1];
  } else {
    new_h = image_height;
    new_w = (origin_image_info_ptr[1] * image_height) / origin_image_info_ptr[0];
  }
  const int index_num = valid_num_ptr[0];
  CUDA_1D_KERNEL_LOOP(i, index_num) {
    const int32_t box_index = select_inds_ptr[i];
    int32_t iw = (box_index / layer_nbox) % layer_width;
    int32_t ih = (box_index / layer_nbox) / layer_width;
    int32_t ibox = box_index % layer_nbox;
    float box_x = (bbox_ptr[box_index * 4 + 0] + iw) / layer_width;
    float box_y = (bbox_ptr[box_index * 4 + 1] + ih) / layer_height;
    float box_w =
        std::exp(bbox_ptr[box_index * 4 + 2]) * anchor_boxes_ptr[2 * ibox] / image_width;
    float box_h =
        std::exp(bbox_ptr[box_index * 4 + 3]) * anchor_boxes_ptr[2 * ibox + 1] / image_height;
    out_bbox_ptr[i * 4 + 0] =
        (box_x - (image_width - new_w) / 2.0 / image_width) / ((float)new_w / image_width);
    out_bbox_ptr[i * 4 + 1] =
        (box_y - (image_height - new_h) / 2.0 / image_height) / ((float)new_h / image_height);
    out_bbox_ptr[i * 4 + 2] = box_w * (float)image_width / new_w;
    out_bbox_ptr[i * 4 + 3] = box_h * (float)image_height / new_h;
  }
}

}  // namespace

template<typename T>
class YoloDetectGpuKernel final : public user_op::OpKernel {
 public:
  YoloDetectGpuKernel() = default;
  ~YoloDetectGpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* bbox = ctx->Tensor4ArgNameAndIndex("bbox", 0);
    const user_op::Tensor* probs = ctx->Tensor4ArgNameAndIndex("probs", 0);
    const user_op::Tensor* origin_image_info = ctx->Tensor4ArgNameAndIndex("origin_image_info", 0);
    user_op::Tensor* out_bbox = ctx->Tensor4ArgNameAndIndex("out_bbox", 0);
    user_op::Tensor* out_probs = ctx->Tensor4ArgNameAndIndex("out_probs", 0);
    user_op::Tensor* valid_num = ctx->Tensor4ArgNameAndIndex("valid_num", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    Memset<DeviceType::kGPU>(ctx->device_ctx(), out_probs->mut_dptr<T>(), 0,
                             out_probs->shape().elem_cnt() * sizeof(T));
    Memset<DeviceType::kGPU>(ctx->device_ctx(), out_bbox->mut_dptr<T>(), 0,
                             out_bbox->shape().elem_cnt() * sizeof(T));
    
    auto anchor_boxes = ctx->Attr<std::vector<int32_t>>("anchor_boxes"); //size=6 3*hw
    const int32_t layer_nbox = anchor_boxes.size() / 2;
    const int32_t box_num = bbox->shape().At(1);
    const int32_t probs_num = ctx->Attr<int32_t>("num_classes") + 1;
    const float prob_thresh = ctx->Attr<float>("prob_thresh");
    const int32_t max_out_boxes = ctx->Attr<int32_t>("max_out_boxes");

    TmpBufferManager buf_manager(static_cast<int32_t>(tmp_buffer->shape().elem_cnt()),
                                    tmp_buffer->mut_dptr<void>(), box_num, layer_nbox*2);
    Memcpy<DeviceType::kGPU>(ctx->device_ctx(), reinterpret_cast<void*>(buf_manager.AnchorBoxesTmpPtr()), reinterpret_cast<void*>(anchor_boxes.data()), GetCudaAlignedSize(buf_manager.AnchorBoxesTmpElemCnt() * sizeof(int32_t)),
                           cudaMemcpyHostToDevice);

    FOR_RANGE(int32_t, im_index, 0, bbox->shape().At(0)) {
      const T* probs_ptr =
          probs->dptr<T>() + im_index * probs->shape().Count(1);
      const T* bbox_ptr =
          bbox->dptr<T>() + im_index * bbox->shape().Count(1);
      T* out_bbox_ptr = out_bbox->mut_dptr<T>()
                        + im_index * out_bbox->shape().Count(1);
      T* out_probs_ptr = out_probs->mut_dptr<T>()
                         + im_index * out_probs->shape().Count(1);
      int32_t* valid_num_ptr = valid_num->mut_dptr<int32_t>()
              + im_index * valid_num->shape().Count(1);
      CudaCheck(SelectOutIndexes(ctx->device_ctx()->cuda_stream(), probs_ptr,
                                 buf_manager.TempStoragePtr(),
                                 buf_manager.SelectIndsPtr(),
                                 valid_num_ptr,
                                 buf_manager.TempStorageBytes(), box_num, probs_num, prob_thresh));
      SetOutProbs<<<BlocksNum4ThreadsNum(box_num * probs_num), kCudaThreadsNumPerBlock, 0,
                    ctx->device_ctx()->cuda_stream()>>>(
          probs_num, probs_ptr, buf_manager.SelectIndsPtr(),
          valid_num_ptr,
          out_probs_ptr, prob_thresh, max_out_boxes);
      SetOutBoxes<<<BlocksNum4ThreadsNum(box_num), kCudaThreadsNumPerBlock, 0,
                    ctx->device_ctx()->cuda_stream()>>>(
          bbox_ptr,
          origin_image_info->dptr<int32_t>()
              + im_index * origin_image_info->shape().Count(1),
          buf_manager.SelectIndsPtr(),
          valid_num_ptr,
          buf_manager.AnchorBoxesTmpPtr(), out_bbox_ptr, ctx->Attr<int32_t>("layer_height"), ctx->Attr<int32_t>("layer_width"), layer_nbox,
          ctx->Attr<int32_t>("image_height"), ctx->Attr<int32_t>("image_width"));
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

};

#define REGISTER_YOLO_DETECT_GPU_KERNEL(dtype)                                                                \
  REGISTER_USER_KERNEL("yolo_detect")                                                                         \
      .SetCreateFn<YoloDetectGpuKernel<dtype>>()                                                              \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kGPU)                                     \
                       & (user_op::HobDataType("bbox", 0) == GetDataType<dtype>::value)         \
                       & (user_op::HobDataType("probs", 0) == GetDataType<dtype>::value))         \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                                     \
        const Shape* bbox_shape = ctx->Shape4ArgNameAndIndex("bbox", 0);                                      \
        const Shape* probs_shape = ctx->Shape4ArgNameAndIndex("probs", 0);                                    \
        const int32_t box_num = bbox_shape->At(1);                                                            \
        const int32_t probs_num = bbox_shape->At(1);                                                          \
        const float prob_thresh = ctx->Attr<float>("prob_thresh");                                         \
        const int32_t layer_nbox = ctx->Attr<std::vector<int32_t>>("anchor_boxes").size() / 2;             \
                                                                                                              \
        /* select_inds */                                                                                     \
        const int32_t select_inds_aligned_bytes = GetCudaAlignedSize(box_num * sizeof(int32_t));              \
        /* anchor_boxes */                                                                                    \
        const int32_t anchor_boxes_tmp_aligned_bytes = GetCudaAlignedSize(2 * layer_nbox * sizeof(int32_t));  \
        /* CUB Temp Storage */                                                                                \
        int32_t temp_storage_bytes = InferTempStorageForCUBYoloDetect(box_num, probs_num, prob_thresh);       \
        return select_inds_aligned_bytes + anchor_boxes_tmp_aligned_bytes + temp_storage_bytes;               \
      });

REGISTER_YOLO_DETECT_GPU_KERNEL(float)
//REGISTER_YOLO_DETECT_GPU_KERNEL(double)
}  // namespace oneflow
