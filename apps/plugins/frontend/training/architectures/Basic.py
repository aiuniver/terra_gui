from typing import List
from pydantic import BaseModel

from ..extra import GroupData
from ...base import Field


class LayerFieldsData(BaseModel):
    task: Field
    loss: Field
    metrics: Field
    classes_quantity: Field


class LayerData(BaseModel):
    name: str
    alias: str
    fields: LayerFieldsData


class OutputsGroupData(GroupData):
    data: List[LayerData]


class CheckpointData(BaseModel):
    layer: Field
    indicator: Field
    type: Field
    mode: Field
    metric_name: Field
    save_best: Field
    save_weights: Field


class CheckpointGroupData(GroupData):
    data: CheckpointData


class CallbacksGroupData(GroupData):
    data: List[str]


class GroupsData(BaseModel):
    outputs: OutputsGroupData
    checkpoint: CheckpointGroupData
    callbacks: CallbacksGroupData
