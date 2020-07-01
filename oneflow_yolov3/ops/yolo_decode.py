from __future__ import absolute_import
import oneflow as flow

def yolo_predict_decoder(batch_size, image_height, image_width, image_paths, name):
    with flow.fixed_placement("cpu", "0:0"):
        return (
            flow.user_op_builder(name)
            .Op("yolo_predict_decoder")
            .Output("out")
            .Output("origin_image_info")
            .Attr("batch_size", batch_size, "AttrTypeInt32")
            .Attr("image_height", image_height, "AttrTypeInt32")
            .Attr("image_width", image_width, "AttrTypeInt32")
            .Attr("image_paths", image_paths, "AttrTypeListString")
            .Build().InferAndTryRun().RemoteBlobList()
        )
