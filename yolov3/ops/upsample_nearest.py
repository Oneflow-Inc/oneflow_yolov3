from __future__ import absolute_import
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow as flow


def upsample_nearest(x, scale, name, data_format="channels_first"):
    return (
        flow.user_op_builder(name)
        .Op("upsample_nearest")
        .Input("x", [x])
        .Output("y")
        .SetAttr("scale", scale, "AttrTypeInt32")
        .SetAttr("data_format", data_format, "AttrTypeString")
        .Build()
        .RemoteBlobList()[0]
    )
