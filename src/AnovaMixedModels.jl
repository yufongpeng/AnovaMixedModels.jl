module AnovaMixedModels

using Statistics, StatsBase, LinearAlgebra, Distributions, Reexport, Printf, GLM
@reexport using MixedModels, AnovaBase
import StatsBase: fit!, fit
using MixedModels: 
    FeMat, createAL, reweight!, getθ,
    _iscomparable, dof, nobs
using StatsModels: RegressionModel, TableRegressionModel, vectorize, asgn, hasintercept
using AnovaGLM: dof_aov
using AnovaBase: FixDispDist, lrt_nested, _diff, subformula, formula_aov, dof_asgn, dof, nobs, select_super_interaction, AnovaTable
import AnovaBase: anova, nestedmodels, predictors, dof_aov, dof_residual, deviance, prednames, anovatable

export anova_lme, anova_lmm, lme, glme

@deprecate anova_lme anova_lmm 
@deprecate lme lmm 
@deprecate glme glmm 

const GLM_MODEL = Union{TableRegressionModel{<: Union{LinearModel, GeneralizedLinearModel}}, LinearModel, GeneralizedLinearModel}

include("anova.jl")
include("fit.jl")
include("io.jl")

end