from __future__ import absolute_import
import oneflow as flow

def yolo_predict_decoder(batch_size, image_height, image_width, image_list_path, name):
    with flow.fixed_placement("cpu", "0:0"):
        return (
            flow.user_op_builder(name)
            .Op("yolo_predict_decoder")
            .Output("out")
            .Output("origin_image_info")
            .SetAttr("batch_size", batch_size, "AttrTypeInt32")
            .SetAttr("image_height", image_height, "AttrTypeInt32")
            .SetAttr("image_width", image_width, "AttrTypeInt32")
            .SetAttr("image_list_path", image_list_path, "AttrTypeString")
            .Build().RemoteBlobList()
        )