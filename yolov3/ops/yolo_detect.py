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
        .SetAttr("image_height", image_height, "AttrTypeInt32")
        .SetAttr("image_width", image_width, "AttrTypeInt32")
        .SetAttr("layer_height", layer_height, "AttrTypeInt32")
        .SetAttr("layer_width", layer_width, "AttrTypeInt32")
        .SetAttr("prob_thresh", prob_thresh, "AttrTypeFloat")
        .SetAttr("num_classes", num_classes, "AttrTypeInt32")
        .SetAttr("max_out_boxes", max_out_boxes, "AttrTypeInt32")
        .SetAttr("anchor_boxes", [anchor_box for anchor_box in anchor_boxes], "AttrTypeListInt32")
        .Build()
    )
    return op.RemoteBlobList()