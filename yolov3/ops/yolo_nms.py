from __future__ import absolute_import

import oneflow as flow


def yolo_nms(boxes, probs, iou_threshold, keep_n, batch_dims, name):
    return (
        flow.user_op_builder(name)
        .Op("yolo_nms")
        .Input("bbox", [boxes])
        .Input("probs", [probs])
        .Output("out")
        .SetAttr("iou_threshold", iou_threshold, "AttrTypeFloat")
        .SetAttr("keep_n", keep_n, "AttrTypeInt32")
        .SetAttr("batch_dims", batch_dims, "AttrTypeInt32")
        .Build()
        .RemoteBlobList()[0]
    )
