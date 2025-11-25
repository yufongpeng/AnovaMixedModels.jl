module AnovaMixedModels

using Statistics, StatsBase, LinearAlgebra, Distributions, Reexport, Printf, GLM
@reexport using MixedModels, AnovaBase
import StatsBase: fit!, fit
import MixedModels: FeMat, createAL, reweight!, getθ,
                     _iscomparable, 
                     deviance, dof, dof_residual, nobs
import StatsModels: RegressionModel, TableRegressionModel, vectorize, asgn, hasintercept
import AnovaBase: anova, nestedmodels, predictors, lrt_nested, _diff, subformula, dof_asgn,
                    dof, dof_residual, deviance, nobs, prednames, select_super_interaction,
                    AnovaTable, anovatable

export anova_lme, lme, glme

const GLM_MODEL = Union{TableRegressionModel{<: Union{LinearModel, GeneralizedLinearModel}}, LinearModel, GeneralizedLinearModel}

include("anova.jl")
include("fit.jl")
include("io.jl")

end