from __future__ import absolute_import

import oneflow as flow


def yolo_nms(boxes, probs, iou_threshold, keep_n, batch_dims, name):
    return (
        flow.user_op_builder(name)
        .Op("yolo_nms")
        .Input("bbox", [boxes])
        .Input("probs", [probs])
        .Output("out")
        .Attr("iou_threshold", iou_threshold)
        .Attr("keep_n", keep_n)
        .Attr("batch_dims", batch_dims)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
