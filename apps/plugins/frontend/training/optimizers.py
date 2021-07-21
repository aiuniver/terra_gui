from pydantic import BaseModel

from ..base import Field


class OptimizerSGDData(BaseModel):
    momentum: Field
    nesterov: Field


class OptimizerRMSpropData(BaseModel):
    rho: Field
    momentum: Field
    epsilon: Field
    centered: Field


class OptimizerAdamData(BaseModel):
    beta_1: Field
    beta_2: Field
    epsilon: Field
    amsgrad: Field


class OptimizerAdadeltaData(BaseModel):
    rho: Field
    epsilon: Field


class OptimizerAdagradData(BaseModel):
    initial_accumulator_value: Field
    epsilon: Field


class OptimizerAdamaxData(BaseModel):
    beta_1: Field
    beta_2: Field
    epsilon: Field


class OptimizerNadamData(BaseModel):
    beta_1: Field
    beta_2: Field
    epsilon: Field


class OptimizerFtrlData(BaseModel):
    learning_rate_power: Field
    initial_accumulator_value: Field
    l1_regularization_strength: Field
    l2_regularization_strength: Field
    l2_shrinkage_regularization_strength: Field
    beta: Field
