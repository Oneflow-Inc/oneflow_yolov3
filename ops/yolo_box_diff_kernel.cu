#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "yolo_kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
class TmpBufferManager final {
 public:
  TmpBufferManager(int32_t capacity, void* ptr, const int32_t box_num, const int32_t gt_max_num,
                   const int32_t anchor_boxes_elem_cnt, const int32_t box_mask_elem_cnt)
      : capacity_{capacity},
        pred_bbox_elem_cnt_{box_num * 4},
        anchor_boxes_tmp_elem_cnt_{anchor_boxes_elem_cnt},
        box_mask_tmp_elem_cnt_{box_mask_elem_cnt},
        overlaps_elem_cnt_{box_num * gt_max_num},
        max_overlaps_elem_cnt_{box_num},
        max_overlaps_gt_indices_elem_cnt_{box_num} {
    const int32_t pred_bbox_aligned_bytes = GetCudaAlignedSize(pred_bbox_elem_cnt_ * sizeof(T));
    const int32_t anchor_boxes_tmp_aligned_bytes =
        GetCudaAlignedSize(anchor_boxes_tmp_elem_cnt_ * sizeof(int32_t));
    const int32_t box_mask_tmp_aligned_bytes =
        GetCudaAlignedSize(box_mask_tmp_elem_cnt_ * sizeof(int32_t));
    const int32_t overlaps_aligned_bytes = GetCudaAlignedSize(overlaps_elem_cnt_ * sizeof(float));
    const int32_t max_overlaps_aligned_bytes =
        GetCudaAlignedSize(max_overlaps_elem_cnt_ * sizeof(float));
    const int32_t max_overlaps_gt_indices_aligned_bytes =
        GetCudaAlignedSize(max_overlaps_gt_indices_elem_cnt_ * sizeof(int32_t));

    pred_bbox_ptr_ = reinterpret_cast<T*>(ptr);
    anchor_boxes_tmp_ptr_ = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(pred_bbox_ptr_)
                                                       + pred_bbox_aligned_bytes);
    box_mask_tmp_ptr_ = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(anchor_boxes_tmp_ptr_)
                                                   + anchor_boxes_tmp_aligned_bytes);
    overlaps_ptr_ = reinterpret_cast<float*>(reinterpret_cast<char*>(box_mask_tmp_ptr_)
                                             + box_mask_tmp_aligned_bytes);
    max_overlaps_ptr_ =
        reinterpret_cast<float*>(reinterpret_cast<char*>(overlaps_ptr_) + overlaps_aligned_bytes);
    max_overlaps_gt_indices_ptr_ = reinterpret_cast<int32_t*>(
        reinterpret_cast<char*>(max_overlaps_ptr_) + max_overlaps_aligned_bytes);
    temp_storage_ptr_ =
        reinterpret_cast<void*>(reinterpret_cast<char*>(max_overlaps_gt_indices_ptr_)
                                + max_overlaps_gt_indices_aligned_bytes);
    temp_storage_bytes_ = capacity_ - pred_bbox_aligned_bytes - anchor_boxes_tmp_aligned_bytes
                          - box_mask_tmp_aligned_bytes - overlaps_aligned_bytes
                          - max_overlaps_aligned_bytes - max_overlaps_gt_indices_aligned_bytes;
    CHECK_GE(temp_storage_bytes_, 0);
  }
  OF_DISALLOW_COPY_AND_MOVE(TmpBufferManager);
  ~TmpBufferManager() = default;

  T* PredBboxPtr() const { return pred_bbox_ptr_; }
  int32_t* AnchorBoxesTmpPtr() const { return anchor_boxes_tmp_ptr_; }
  int32_t* BoxMaskTmpPtr() const { return box_mask_tmp_ptr_; }
  float* OverlapsPtr() const { return overlaps_ptr_; }
  float* MaxOverapsPtr() const { return max_overlaps_ptr_; }
  int32_t* MaxOverapsGtIndicesPtr() const { return max_overlaps_gt_indices_ptr_; }
  void* TempStoragePtr() const { return temp_storage_ptr_; }

  int32_t BoxMaskTmpElemCnt() const { return box_mask_tmp_elem_cnt_; }
  int32_t AnchorBoxesTmpElemCnt() const { return anchor_boxes_tmp_elem_cnt_; }
  int32_t TempStorageBytes() const { return temp_storage_bytes_; }

 private:
  int32_t capacity_;

  T* pred_bbox_ptr_;
  int32_t* anchor_boxes_tmp_ptr_;
  int32_t* box_mask_tmp_ptr_;
  float* overlaps_ptr_;
  float* max_overlaps_ptr_;
  int32_t* max_overlaps_gt_indices_ptr_;
  void* temp_storage_ptr_;

  int32_t pred_bbox_elem_cnt_;
  int32_t anchor_boxes_tmp_elem_cnt_;
  int32_t box_mask_tmp_elem_cnt_;
  int32_t overlaps_elem_cnt_;
  int32_t max_overlaps_elem_cnt_;
  int32_t max_overlaps_gt_indices_elem_cnt_;
  int32_t temp_storage_bytes_;
};

template<typename T>
__global__ void CoordinateTransformGpu(const int32_t box_num, const T* bbox,
                                       const int32_t* anchor_boxes_size_ptr,
                                       const int32_t* box_mask_ptr, T* pred_bbox,
                                       const int32_t layer_height, const int32_t layer_width,
                                       const int32_t image_height, const int32_t image_width,
                                       const int32_t layer_nbox) {
  CUDA_1D_KERNEL_LOOP(i, box_num) {
    // n=4*1083 or 4*4332 or 4*17328
    const int32_t iw = (i / layer_nbox) % layer_width;
    const int32_t ih = (i / layer_nbox) / layer_width;
    const int32_t ibox = box_mask_ptr[i % layer_nbox];
    pred_bbox[4 * i + 0] = (bbox[4 * i + 0] + iw) / static_cast<T>(layer_width);
    pred_bbox[4 * i + 1] = (bbox[4 * i + 1] + ih) / static_cast<T>(layer_height);
    pred_bbox[4 * i + 2] =
        exp(bbox[4 * i + 2]) * anchor_boxes_size_ptr[2 * ibox] / static_cast<T>(image_width);
    pred_bbox[4 * i + 3] =
        exp(bbox[4 * i + 3]) * anchor_boxes_size_ptr[2 * ibox + 1] / static_cast<T>(image_height);
  }
}

template<typename T>
__device__ void CalcIouBoxForStatisticInfo(T* bbox1, T* bbox2, float* statistics_info_ptr) {
    //xywh->x1y1x2y2
    const float box1_left = bbox1[0] - 0.5f * bbox1[2];
    const float box1_right = bbox1[0] + 0.5f * bbox1[2];
    const float box1_top = bbox1[1] - 0.5f * bbox1[3];
    const float box1_bottom = bbox1[1] + 0.5f * bbox1[3];
    //xywh->x1y1x2y2
    const float box2_left = bbox2[0] - 0.5f * bbox2[2];
    const float box2_right = bbox2[0] + 0.5f * bbox2[2];
    const float box2_top = bbox2[1] - 0.5f * bbox2[3];
    const float box2_bottom = bbox2[1] + 0.5f * bbox2[3];
    
    const float iw = min(box1_right, box2_right) - max(box1_left, box2_left);
    const float ih = min(box1_bottom, box2_bottom) - max(box1_top, box2_top);
    const float inter = iw*ih;
    float iou = 0;
    if (iw<0 || ih <0) {
      iou = 0; 
    } else {
      iou = inter / (bbox1[2]*bbox1[3] + bbox2[2]*bbox2[3] - inter);
    }
    atomicAdd(statistics_info_ptr+0, iou); //iou
    if(iou > 0.5){atomicAdd(statistics_info_ptr+1, 1);} //recall
    if(iou > 0.75){atomicAdd(statistics_info_ptr+2, 1);} //recall75
    atomicAdd(statistics_info_ptr+3, 1); //count
    atomicAdd(statistics_info_ptr+4, 1); //class_count
}

template<typename T>
__global__ void CalcIouGpu(const int32_t box_num, const T* pred_bbox, const T* gt_boxes_ptr,
                           float* overlaps, const int32_t gt_max_num,
                           const int32_t* gt_valid_num_ptr) {
  CUDA_1D_KERNEL_LOOP(i, box_num * gt_valid_num_ptr[0]) {
    const int32_t box_index = i / gt_valid_num_ptr[0];
    const int32_t gt_index = i % gt_valid_num_ptr[0];
    const float gt_left = gt_boxes_ptr[gt_index * 4] - 0.5f * gt_boxes_ptr[gt_index * 4 + 2];
    const float gt_right = gt_boxes_ptr[gt_index * 4] + 0.5f * gt_boxes_ptr[gt_index * 4 + 2];
    const float gt_top = gt_boxes_ptr[gt_index * 4 + 1] - 0.5f * gt_boxes_ptr[gt_index * 4 + 3];
    const float gt_bottom = gt_boxes_ptr[gt_index * 4 + 1] + 0.5f * gt_boxes_ptr[gt_index * 4 + 3];
    const float gt_area = gt_boxes_ptr[gt_index * 4 + 2] * gt_boxes_ptr[gt_index * 4 + 3];

    const float box_left = pred_bbox[box_index * 4] - 0.5f * pred_bbox[box_index * 4 + 2];
    const float box_right = pred_bbox[box_index * 4] + 0.5f * pred_bbox[box_index * 4 + 2];
    const float box_top = pred_bbox[box_index * 4 + 1] - 0.5f * pred_bbox[box_index * 4 + 3];
    const float box_bottom = pred_bbox[box_index * 4 + 1] + 0.5f * pred_bbox[box_index * 4 + 3];
    const float iw = min(box_right, gt_right) - max(gt_left, box_left);
    const float ih = min(box_bottom, gt_bottom) - max(box_top, gt_top);
    const float inter = iw * ih;
    if (iw < 0 || ih < 0) {
      overlaps[box_index * gt_max_num + gt_index] = 0.0f;
    } else {
      overlaps[box_index * gt_max_num + gt_index] =
          inter / (pred_bbox[box_index * 4 + 2] * pred_bbox[box_index * 4 + 3] + gt_area - inter);
    }
  }
}

__global__ void SetMaxOverlapsAndGtIndex(const int32_t box_num, const int32_t* gt_valid_num_ptr,
                                         const int32_t gt_max_num, const float* overlaps,
                                         float* max_overlaps, int32_t* max_overlaps_gt_indices,
                                         const float ignore_thresh, const float truth_thresh) {
  CUDA_1D_KERNEL_LOOP(i, box_num) {
    max_overlaps[i] = 0.0f;
    max_overlaps_gt_indices[i] = -1;
    for (int j = 0; j < gt_valid_num_ptr[0]; j++) {
      if (overlaps[i * gt_max_num + j] > max_overlaps[i]) {
        max_overlaps[i] = overlaps[i * gt_max_num + j];
        if (overlaps[i * gt_max_num + j] <= ignore_thresh) {
          max_overlaps_gt_indices[i] = -1;
        }  // negative
        else if (overlaps[i * gt_max_num + j] > truth_thresh) {
          max_overlaps_gt_indices[i] = j;
        }  // postive
        else {
          max_overlaps_gt_indices[i] = -2;
        }
      }
    }
  }
}

template<typename T>
__global__ void CalcGtNearestAnchorSize(const T* pred_bbox_ptr, const int32_t* gt_valid_num_ptr, const T* gt_boxes_ptr,
                                        const int32_t* anchor_boxes_size_ptr,
                                        const int32_t* box_mask_ptr,
                                        int32_t* max_overlaps_gt_indices, float* statistics_info_ptr,
                                        const int32_t anchor_boxes_size_num,
                                        const int32_t box_mask_num, const int32_t layer_height,
                                        const int32_t layer_width, const int32_t layer_nbox,
                                        const int32_t image_height, const int32_t image_width) {
  CUDA_1D_KERNEL_LOOP(i, gt_valid_num_ptr[0]) {
    const float gt_left = 0 - 0.5f * gt_boxes_ptr[i * 4 + 2];
    const float gt_right = 0 + 0.5f * gt_boxes_ptr[i * 4 + 2];
    const float gt_bottom = 0 + 0.5f * gt_boxes_ptr[i * 4 + 3];
    const float gt_top = 0 - 0.5f * gt_boxes_ptr[i * 4 + 3];
    const float gt_area = gt_boxes_ptr[i * 4 + 2] * gt_boxes_ptr[i * 4 + 3];
    float max_overlap = 0.0f;
    int32_t max_overlap_anchor_idx = -1;
    for (int32_t j = 0; j < anchor_boxes_size_num; j++) {
      const float box_left =
          0 - 0.5f * static_cast<T>(anchor_boxes_size_ptr[2 * j]) / static_cast<T>(image_width);
      const float box_right =
          0 + 0.5f * static_cast<T>(anchor_boxes_size_ptr[2 * j]) / static_cast<T>(image_width);
      const float box_bottom =
          0
          + 0.5f * static_cast<T>(anchor_boxes_size_ptr[2 * j + 1]) / static_cast<T>(image_height);
      const float box_top =
          0
          - 0.5f * static_cast<T>(anchor_boxes_size_ptr[2 * j + 1]) / static_cast<T>(image_height);
      const float box_area =
          static_cast<T>(anchor_boxes_size_ptr[2 * j]) / static_cast<T>(image_width)
          * static_cast<T>(anchor_boxes_size_ptr[2 * j + 1]) / static_cast<T>(image_height);
      const float iw = min(box_right, gt_right) - max(gt_left, box_left);
      const float ih = min(box_bottom, gt_bottom) - max(box_top, gt_top);
      const float inter = iw * ih;
      float overlap = 0.0f;
      if (iw < 0 || ih < 0) {
        overlap = 0;
      } else {
        overlap = inter / (gt_area + box_area - inter);
      }
      if (overlap > max_overlap) {
        max_overlap = overlap;
        max_overlap_anchor_idx = j;
      }
    }
    for (int32_t j = 0; j < box_mask_num; j++) {
      if (box_mask_ptr[j] == max_overlap_anchor_idx) {
        const int32_t fm_i = static_cast<int32_t>(floor(gt_boxes_ptr[i * 4] * layer_width));
        const int32_t fm_j = static_cast<int32_t>(floor(gt_boxes_ptr[i * 4 + 1] * layer_height));
        const int32_t box_index = fm_j * layer_width * layer_nbox + fm_i * layer_nbox + j;
        max_overlaps_gt_indices[box_index] = i;
        CalcIouBoxForStatisticInfo(pred_bbox_ptr+box_index * 4, gt_boxes_ptr+i*4, statistics_info_ptr);
      }
    }
  }
}

template<typename T>
__global__ void CalcBboxLoss(const int32_t box_num, const T* bbox_ptr, const T* gt_boxes_ptr,
                             const int32_t* gt_labels_ptr, const int32_t* pos_inds_ptr,
                             const int32_t* valid_num_ptr, const int32_t* max_overlaps_gt_indices,
                             const int32_t* anchor_boxes_size_ptr, const int32_t* box_mask_ptr,
                             T* bbox_loc_diff_ptr, int32_t* labels_ptr, const int32_t layer_nbox,
                             const int32_t layer_height, const int32_t layer_width,
                             const int32_t image_height, const int32_t image_width) {
  const int32_t pos_num = valid_num_ptr[0];
  CUDA_1D_KERNEL_LOOP(i, pos_num) {
    int box_index = pos_inds_ptr[i];
    int gt_index = max_overlaps_gt_indices[box_index];
    labels_ptr[box_index] = gt_labels_ptr[gt_index];
    const float scale = 2 - gt_boxes_ptr[gt_index * 4 + 2] * gt_boxes_ptr[gt_index * 4 + 3];

    const int32_t iw = (box_index / layer_nbox) % layer_width;
    const int32_t ih = (box_index / layer_nbox) / layer_width;
    const int32_t ibox = box_mask_ptr[box_index % layer_nbox];
    float gt_x = gt_boxes_ptr[gt_index * 4] * layer_width - iw;
    float gt_y = gt_boxes_ptr[gt_index * 4 + 1] * layer_height - ih;
    float gt_w = log(gt_boxes_ptr[gt_index * 4 + 2] * image_width
                     / static_cast<T>(anchor_boxes_size_ptr[ibox * 2]));
    float gt_h = log(gt_boxes_ptr[gt_index * 4 + 3] * image_height
                     / static_cast<T>(anchor_boxes_size_ptr[ibox * 2 + 1]));
    bbox_loc_diff_ptr[box_index * 4 + 0] = scale * (bbox_ptr[box_index * 4] - gt_x);
    bbox_loc_diff_ptr[box_index * 4 + 1] = scale * (bbox_ptr[box_index * 4 + 1] - gt_y);
    bbox_loc_diff_ptr[box_index * 4 + 2] = scale * (bbox_ptr[box_index * 4 + 2] - gt_w);
    bbox_loc_diff_ptr[box_index * 4 + 3] = scale * (bbox_ptr[box_index * 4 + 3] - gt_h);
  }
}

}  // namespace

template<typename T>
class YoloBoxDiffKernel final : public user_op::OpKernel {
 public:
  YoloBoxDiffKernel() = default;
  ~YoloBoxDiffKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* bbox = ctx->Tensor4ArgNameAndIndex("bbox", 0);
    const user_op::Tensor* gt_boxes = ctx->Tensor4ArgNameAndIndex("gt_boxes", 0);
    const user_op::Tensor* gt_labels = ctx->Tensor4ArgNameAndIndex("gt_labels", 0);
    const user_op::Tensor* gt_valid_num = ctx->Tensor4ArgNameAndIndex("gt_valid_num", 0);
    user_op::Tensor* bbox_loc_diff = ctx->Tensor4ArgNameAndIndex("bbox_loc_diff", 0);
    user_op::Tensor* pos_inds = ctx->Tensor4ArgNameAndIndex("pos_inds", 0);
    user_op::Tensor* pos_cls_label = ctx->Tensor4ArgNameAndIndex("pos_cls_label", 0);
    user_op::Tensor* neg_inds = ctx->Tensor4ArgNameAndIndex("neg_inds", 0);
    user_op::Tensor* valid_num = ctx->Tensor4ArgNameAndIndex("valid_num", 0);
    user_op::Tensor* statistics_info = ctx->Tensor4ArgNameAndIndex("statistics_info", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    Memset<DeviceType::kGPU>(ctx->device_ctx(), bbox_loc_diff->mut_dptr<T>(), 0,
                             bbox_loc_diff->shape().elem_cnt() * sizeof(T));

    auto anchor_boxes = ctx->Attr<std::vector<int32_t>>("anchor_boxes");  // size=2*9 9*hw
    auto box_mask = ctx->Attr<std::vector<int32_t>>("box_mask");          // size=3

    const int32_t gt_max_num = gt_boxes->shape().At(1);
    const int32_t box_num = bbox->shape().At(1);
    const int32_t image_height = ctx->Attr<int32_t>("image_height");
    const int32_t image_width = ctx->Attr<int32_t>("image_width");
    const int32_t layer_height = ctx->Attr<int32_t>("layer_height");
    const int32_t layer_width = ctx->Attr<int32_t>("layer_width");
    const float ignore_thresh = ctx->Attr<float>("ignore_thresh");
    const float truth_thresh = ctx->Attr<float>("truth_thresh");
    const int32_t layer_nbox = box_mask.size();
    const int32_t anchor_boxes_size_num = anchor_boxes.size();

    TmpBufferManager<T> buf_manager(static_cast<int32_t>(tmp_buffer->shape().elem_cnt()),
                                    tmp_buffer->mut_dptr<void>(), box_num, gt_max_num,
                                    anchor_boxes_size_num, layer_nbox);
    Memcpy<DeviceType::kGPU>(
        ctx->device_ctx(), reinterpret_cast<void*>(buf_manager.AnchorBoxesTmpPtr()),
        reinterpret_cast<void*>(anchor_boxes.data()),
        buf_manager.AnchorBoxesTmpElemCnt() * sizeof(int32_t));
    Memcpy<DeviceType::kGPU>(ctx->device_ctx(),
                             reinterpret_cast<void*>(buf_manager.BoxMaskTmpPtr()),
                             reinterpret_cast<void*>(box_mask.data()),
                             buf_manager.BoxMaskTmpElemCnt() * sizeof(int32_t));
    Memset<DeviceType::kGPU>(ctx->device_ctx(), statistics_info->mut_dptr<float>(), 0, statistics_info->shape().elem_cnt() * sizeof(float));

    FOR_RANGE(int32_t, im_index, 0, bbox->shape().At(0)) {
      const int32_t* gt_valid_num_ptr =
          gt_valid_num->dptr<int32_t>() + im_index * gt_valid_num->shape().Count(1);
      const T* gt_boxes_ptr = gt_boxes->dptr<T>() + im_index * gt_boxes->shape().Count(1);
      const T* bbox_ptr = bbox->dptr<T>() + im_index * bbox->shape().Count(1);
      float* statistics_info_ptr = statistics_info->mut_dptr<float>() + im_index * statistics_info->shape().Count(1);

      CoordinateTransformGpu<<<BlocksNum4ThreadsNum(box_num), kCudaThreadsNumPerBlock, 0,
                               ctx->device_ctx()->cuda_stream()>>>(
          box_num, bbox_ptr, buf_manager.AnchorBoxesTmpPtr(), buf_manager.BoxMaskTmpPtr(),
          buf_manager.PredBboxPtr(), layer_height, layer_width, image_height, image_width,
          layer_nbox);
      CalcIouGpu<<<BlocksNum4ThreadsNum(box_num * gt_max_num), kCudaThreadsNumPerBlock, 0,
                   ctx->device_ctx()->cuda_stream()>>>(box_num, buf_manager.PredBboxPtr(),
                                                       gt_boxes_ptr, buf_manager.OverlapsPtr(),
                                                       gt_max_num, gt_valid_num_ptr);
      SetMaxOverlapsAndGtIndex<<<BlocksNum4ThreadsNum(box_num), kCudaThreadsNumPerBlock, 0,
                                 ctx->device_ctx()->cuda_stream()>>>(
          box_num, gt_valid_num_ptr, gt_max_num, buf_manager.OverlapsPtr(),
          buf_manager.MaxOverapsPtr(), buf_manager.MaxOverapsGtIndicesPtr(), ignore_thresh,
          truth_thresh);
      CalcGtNearestAnchorSize<<<BlocksNum4ThreadsNum(gt_max_num), kCudaThreadsNumPerBlock, 0,
                                ctx->device_ctx()->cuda_stream()>>>(
          buf_manager.PredBboxPtr(), gt_valid_num_ptr, gt_boxes_ptr, buf_manager.AnchorBoxesTmpPtr(),
          buf_manager.BoxMaskTmpPtr(), buf_manager.MaxOverapsGtIndicesPtr(), statistics_info_ptr, anchor_boxes_size_num,
          layer_nbox, layer_height, layer_width, layer_nbox, image_height, image_width);
      int32_t* pos_inds_ptr = pos_inds->mut_dptr<int32_t>() + im_index * pos_inds->shape().Count(1);
      int32_t* neg_inds_ptr = neg_inds->mut_dptr<int32_t>() + im_index * neg_inds->shape().Count(1);
      int32_t* valid_num_ptr =
          valid_num->mut_dptr<int32_t>() + im_index * valid_num->shape().Count(1);

      SelectSamples(ctx->device_ctx()->cuda_stream(), buf_manager.MaxOverapsGtIndicesPtr(),
                    buf_manager.TempStoragePtr(), pos_inds_ptr, neg_inds_ptr, valid_num_ptr,
                    buf_manager.TempStorageBytes(), box_num);
      CalcBboxLoss<<<BlocksNum4ThreadsNum(gt_max_num), kCudaThreadsNumPerBlock, 0,
                     ctx->device_ctx()->cuda_stream()>>>(
          box_num, bbox_ptr, gt_boxes_ptr,
          gt_labels->dptr<int32_t>() + im_index * gt_labels->shape().Count(1), pos_inds_ptr,
          valid_num_ptr, buf_manager.MaxOverapsGtIndicesPtr(), buf_manager.AnchorBoxesTmpPtr(),
          buf_manager.BoxMaskTmpPtr(),
          bbox_loc_diff->mut_dptr<T>() + im_index * bbox_loc_diff->shape().Count(1),
          pos_cls_label->mut_dptr<int32_t>() + im_index * pos_cls_label->shape().Count(1),
          layer_nbox, layer_height, layer_width, image_height, image_width);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_YOLO_BOX_DIFF_GPU_KERNEL(dtype)                                                  \
  REGISTER_USER_KERNEL("yolo_box_diff")                                                           \
      .SetCreateFn<YoloBoxDiffKernel<dtype>>()                                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                             \
                       & (user_op::HobDataType("bbox", 0) == GetDataType<dtype>::value)           \
                       & (user_op::HobDataType("gt_boxes", 0) == GetDataType<dtype>::value))      \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                         \
        const Shape* bbox_shape = ctx->Shape4ArgNameAndIndex("bbox", 0);                          \
        const Shape* gt_boxes_shape = ctx->Shape4ArgNameAndIndex("gt_boxes", 0);                  \
        const int32_t box_num = bbox_shape->At(1);                                                \
        const int32_t gt_max_num = gt_boxes_shape->At(1);                                         \
        const int32_t anchor_boxes_elem_cnt =                                                     \
            ctx->Attr<std::vector<int32_t>>("anchor_boxes").size();                               \
        const int32_t layer_nbox = ctx->Attr<std::vector<int32_t>>("box_mask").size();            \
                                                                                                  \
        /* pred_bbox */                                                                           \
        const int32_t pred_bbox_aligned_bytes = GetCudaAlignedSize(box_num * 4 * sizeof(dtype));  \
        /* anchor_boxes_tmp */                                                                    \
        const int32_t anchor_boxes_tmp_aligned_bytes =                                            \
            GetCudaAlignedSize(anchor_boxes_elem_cnt * sizeof(int32_t));                          \
        /* box_mask_tmp */                                                                        \
        const int32_t box_mask_tmp_aligned_bytes =                                                \
            GetCudaAlignedSize(layer_nbox * sizeof(int32_t));                                     \
        /* overlaps */                                                                            \
        const int32_t overlaps_aligned_bytes =                                                    \
            GetCudaAlignedSize(box_num * gt_max_num * sizeof(float));                             \
        /* max_overlaps */                                                                        \
        const int32_t max_overlaps_aligned_bytes = GetCudaAlignedSize(box_num * sizeof(float));   \
        /* max_overlaps_gt_indices */                                                             \
        const int32_t max_overlaps_gt_indices_aligned_bytes =                                     \
            GetCudaAlignedSize(box_num * sizeof(int32_t));                                        \
        /* CUB Temp Storage */                                                                    \
        int32_t temp_storage_bytes = InferTempStorageForCUBYoloBoxDiff(box_num);                  \
        return pred_bbox_aligned_bytes + anchor_boxes_tmp_aligned_bytes                           \
               + box_mask_tmp_aligned_bytes + overlaps_aligned_bytes + max_overlaps_aligned_bytes \
               + max_overlaps_gt_indices_aligned_bytes + temp_storage_bytes;                      \
      });

REGISTER_YOLO_BOX_DIFF_GPU_KERNEL(float)
// REGISTER_YOLO_BOX_DIFF_GPU_KERNEL(double)
}  // namespace oneflow
