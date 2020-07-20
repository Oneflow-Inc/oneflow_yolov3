from __future__ import absolute_import

import oneflow as flow


def upsample_nearest(x, scale, name, data_format="channels_first"):
    return (
        flow.user_op_builder(name)
        .Op("upsample_nearest")
        .Input("x", [x])
        .Output("y")
        .Attr("scale", scale, "AttrTypeInt32")
        .Attr("data_format", data_format, "AttrTypeString")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
