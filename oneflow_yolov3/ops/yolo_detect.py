from __future__ import absolute_import
import oneflow as flow

	
def yolo_detect(bbox, probs, origin_image_info, image_height, image_width, layer_height, layer_width, prob_thresh, num_classes, anchor_boxes, max_out_boxes=None, name=None):
    #if name is None:
    #    name = id_util.UniqueStr("YoloDetect_")
    if max_out_boxes is None:
        max_out_boxes = bbox.static_shape[1]
    assert isinstance(anchor_boxes, (list, tuple))

    op = (
        flow.user_op_builder(name)
        .Op("yolo_detect")
        .Input("bbox", [bbox])
        .Input("probs", [probs])
        .Input("origin_image_info", [origin_image_info])
        .Output("out_bbox")
        .Output("out_probs")
        .Output("valid_num")
        .Attr("image_height", image_height, "AttrTypeInt32")
        .Attr("image_width", image_width, "AttrTypeInt32")
        .Attr("layer_height", layer_height, "AttrTypeInt32")
        .Attr("layer_width", layer_width, "AttrTypeInt32")
        .Attr("prob_thresh", prob_thresh, "AttrTypeFloat")
        .Attr("num_classes", num_classes, "AttrTypeInt32")
        .Attr("max_out_boxes", max_out_boxes, "AttrTypeInt32")
        .Attr("anchor_boxes", [anchor_box for anchor_box in anchor_boxes], "AttrTypeListInt32")
        .Build().InferAndTryRun()
    )
    return op.RemoteBlobList()

def yolo_box_diff(bbox, gt_boxes, gt_labels, gt_valid_num, image_height, image_width, layer_height, layer_width, ignore_thresh, truth_thresh, box_mask, anchor_boxes_size, name=None):
    assert isinstance(anchor_boxes_size, (list, tuple))
    assert isinstance(box_mask, (list, tuple))
    
    op = (
        flow.user_op_builder(name)
        .Op("yolo_box_diff")
        .Input("bbox", [bbox])
        .Input("gt_boxes", [gt_boxes])
        .Input("gt_labels", [gt_labels])
        .Input("gt_valid_num", [gt_valid_num])
        .Output("bbox_loc_diff")
        .Output("pos_inds")
        .Output("pos_cls_label")
        .Output("neg_inds")
        .Output("valid_num")
        .Output("statistics_info")
        .Attr("image_height", image_height, "AttrTypeInt32")
        .Attr("image_width", image_width, "AttrTypeInt32")
        .Attr("layer_height", layer_height, "AttrTypeInt32")
        .Attr("layer_width", layer_width, "AttrTypeInt32")
        .Attr("ignore_thresh", ignore_thresh, "AttrTypeFloat")
        .Attr("truth_thresh", truth_thresh, "AttrTypeFloat")
        .Attr("box_mask", [mask for mask in box_mask], "AttrTypeListInt32")
        .Attr("anchor_boxes", [anchor_box for anchor_box in anchor_boxes_size], "AttrTypeListInt32")
        .Build().InferAndTryRun()
    )
    return op.RemoteBlobList()

def yolo_prob_loss(bbox_objness, bbox_clsprob, pos_inds, pos_cls_label, neg_inds, valid_num, num_classes, name=None):

    op = (
        flow.user_op_builder(name)
        .Op("yolo_prob_loss")
        .Input("bbox_objness", [bbox_objness])
        .Input("bbox_clsprob", [bbox_clsprob])
        .Input("pos_inds", [pos_inds])
        .Input("pos_cls_label", [pos_cls_label])
        .Input("neg_inds", [neg_inds])
        .Input("valid_num", [valid_num])
        .Output("bbox_objness_out")
        .Output("bbox_clsprob_out")
        .Attr("num_classes", num_classes, "AttrTypeInt32")
        .Build().InferAndTryRun()
    )
    return op.RemoteBlobList()

def logistic(x, name=None):
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Logistic_")
        )
        .Op("logistic")
        .Input("in", [x])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
