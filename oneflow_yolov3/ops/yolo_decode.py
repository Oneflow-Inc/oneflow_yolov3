from __future__ import absolute_import
import oneflow as flow


def yolo_predict_decoder(batch_size, image_height, image_width, image_paths, name):
    with flow.scope.placement("cpu", "0:0"):
        return (
            flow.user_op_builder(name)
                .Op("yolo_predict_decoder")
                .Output("out")
                .Output("origin_image_info")
                .Attr("batch_size", batch_size)
                .Attr("image_height", image_height)
                .Attr("image_width", image_width)
                .Attr("image_paths", image_paths)
                .Build().InferAndTryRun().RemoteBlobList()
        )


def yolo_train_decoder(batch_size, image_height, image_width, classes, num_boxes, hue, jitter, saturation, exposure,
                       image_path_file, name):
    with flow.scope.placement("cpu", "0:0"):
        return (
            flow.user_op_builder(name)
                .Op("yolo_train_decoder")
                .Output("data")
                .Output("ground_truth")
                .Output("gt_valid_num")
                .Attr("batch_size", batch_size)
                .Attr("image_height", image_height)
                .Attr("image_width", image_width)
                .Attr("classes", classes)
                .Attr("num_boxes", num_boxes)
                .Attr("hue", hue)
                .Attr("jitter", jitter)
                .Attr("saturation", saturation)
                .Attr("exposure", exposure)
                .Attr("image_path_file", image_path_file)
                .Build().InferAndTryRun().RemoteBlobList()
        )
