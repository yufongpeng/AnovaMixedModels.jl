# ============================================================================================================
# Main API
"""
    anova(mixedmodels...; test::Type{<: GoodnessOfFit}, keyword_arguments...)
    anova(anovamodel; test::Type{<: GoodnessOfFit}, keyword_arguments...)
    anova(test::Type{<: GoodnessOfFit}, mixedmodels...; keyword_arguments...)
    anova(test::Type{<: GoodnessOfFit}, anovamodel; keyword_arguments...)

Analysis of variance.

Return `AnovaResult{M, test, N}`. See [`AnovaResult`](@ref) for details.

# Arguments
* `mixedmodels`: model objects
    1. `LinearMixedModel` fitted by `AnovaMixedModels.lme` or `fit(LinearMixedModel, ...)`
    2. `GeneralizedLinearMixedModel` fitted by `AnovaMixedModels.glme` or `fit(GeneralizedLinearMixedModel, ...)`
    If mutiple models are provided, they should be nested and the last one is the most complex. The first model can also be the corresponding `GLM` object without random effects.
* `anovamodel`: wrapped model objects; `FullModel` and `NestedModels`.
* `test`: test statistics for goodness of fit. Available tests are [`LikelihoodRatioTest`](@ref) (`LRT`) and [`FTest`](@ref). The default is based on the model type.
    1. `LinearMixedModel`: `FTest` for one model; `LRT` for nested models.
    2. `GeneralizedLinearMixedModel`: `LRT` for nested models.
    Models should not be fitted by REML for `LRT`. 

# Other keyword arguments
* When one model is provided:  
    1. `type`: type of anova (1, 2 or 3). Default value is 1.
    2. `adjust_sigma`: whether adjust σ to match that of linear mixed-effect model fitted by REML. The result will be slightly deviated from that of model fitted by REML.
* When multiple models are provided:  
    1. `check`: allows to check if models are nested. Defalut value is true. Some checkers are not implemented now.

!!! note
    For fitting new models and conducting anova at the same time, see [`anova_lme`](@ref) for `LinearMixedModel`.
!!! note
    The result with `adjust_sigma` will be slightly deviated from that of model fitted directly by REML.
!!! note
    For the computation of degrees of freedom, please see [`dof_residual_pred`](@ref).
"""
anova(::Type{<: GoodnessOfFit}, ::MixedModel)

anova(models::Vararg{M}; 
        test::Type{<: GoodnessOfFit} = length(models) > 1 ? LRT : FTest, 
        kwargs...) where {M <: MixedModel} = 
    anova(test, models...; kwargs...)

anova(aovm::FullModel{M}; 
        test::Type{<: GoodnessOfFit} = FTest, 
        kwargs...) where {M <: MixedModel} = 
    anova(test, aovm; kwargs...)

anova(aovm::NestedModels{M}; 
        test::Type{<: GoodnessOfFit} = LRT, 
        kwargs...) where {M <: MixedModel} = 
    anova(test, aovm; kwargs...)
# ==================================================================================================================
# ANOVA by F test
# Linear mixed-effect models
anova(::Type{FTest}, model::M; type::Int = 1, kwargs...) where {M <: LinearMixedModel} = 
    anova(FTest, FullModel(model, type, true, true); kwargs...)

function anova(::Type{FTest}, 
        aovm::FullModel{M};
        adjust_sigma::Bool = true) where {M <: LinearMixedModel}

    assign = asgn(predictors(aovm))
    fullpred = predictors(aovm.model)
    fullasgn = asgn(fullpred)
    df = tuple(dof_asgn(assign)...)
    dfr = dof_residual_pred(aovm)
    # calculate degree of freedom for factors and residuals
    varβ = vcov(aovm.model) 
    β = fixef(aovm.model)
    # use MMatrix/SizedMatrix ?
    offset = first(assign) + last(fullasgn) - last(assign) - 1
    adjust = 1.0
    aovm.model.optsum.REML || adjust_sigma && (adjust = (nobs(aovm.model) - length(β)) / nobs(aovm.model)) 
    if aovm.type == 1
        fs = abs2.(cholesky(Hermitian(inv(varβ))).U * β)
        # adjust σ like linear regression
        fstat = ntuple(last(fullasgn) - offset) do fix
            sum(fs[findall(==(fix + offset), fullasgn)]) / df[fix] * adjust
        end
    elseif aovm.type == 2
        fstat = ntuple(last(fullasgn) - offset) do fix
            s1 = sort!(collect(select_super_interaction(fullpred, fix + offset)))
            s2 = setdiff(s1, fix + offset)
            select1 = findall(in(s1), fullasgn)
            select2 = findall(in(s2), fullasgn)
            (β[select1]' * (varβ[select1, select1] \ β[select1]) - β[select2]' * (varβ[select2, select2] \ β[select2])) / df[fix] * adjust
        end
    else 
        # calculate block by block
        fstat = ntuple(last(fullasgn) - offset) do fix
            select = findall(==(fix + offset), fullasgn)
            β[select]' * (varβ[select, select] \ β[select]) / df[fix] * adjust
        end
    end

    pvalue = ntuple(lastindex(fstat)) do id
        ccdf(FDist(df[id], dfr[id]), abs(fstat[id]))
    end
    AnovaResult(aovm, FTest, df, ntuple(x->NaN, length(fstat)), fstat, pvalue, (dof_residual = dfr,))
end

#=
function anova(::Type{FTest}, 
        model::GeneralizedLinearMixedModel; 
        type::Int = 1, 
        adjust_sigma::Bool = true,
        kwargs...) 
        # use refit! and deviance as in GLM?
end
=#
# ==================================================================================================================
# ANOVA by Likehood-ratio test 
# Linear mixed-effect models

function anova(::Type{LRT}, model::M) where {M <: LinearMixedModel}
    # check if fitted by ML 
    # nested random effects for REML ?
    model.optsum.REML && throw(
        ArgumentError("""Likelihood-ratio tests for REML-fitted models are only valid when the fixed-effects specifications are identical"""))
    @warn "Fit all submodels"
    models = nestedmodels(model; null = isnullable(model))
    anova(LRT, models)
end

function anova(::Type{LRT}, aovm::FullModel{M}) where {M <: LinearMixedModel}
    # check if fitted by ML 
    # nested random effects for REML ?
    aovm.model.optsum.REML && throw(
        ArgumentError("""Likelihood-ratio tests for REML-fitted models are only valid when the fixed-effects specifications are identical"""))
    @warn "Fit all submodels"
    models = nestedmodels(aovm.model; null = isnullable(aovm.model))
    anova(LRT, models)
end

# =================================================================================================================
# Nested models 

function anova(::Type{LikelihoodRatioTest}, 
                models::Vararg{M}; 
                check::Bool = true) where {M <: MixedModel}

    check && (_iscomparable(models...) || throw(
        ArgumentError("""Models are not comparable: are the objectives, data and, where appropriate, the link and family the same?""")))
    # _isnested (by QR) or isnested (by formula) are not part of _iscomparable:  
    # isnested = true  
    df = dof.(models)
    ord = sortperm(collect(df))
    df = df[ord]
    models = models[ord]
    lrt_nested(NestedModels(models), df, deviance_mixedmodel.(models), 1)
end

anova(::Type{LikelihoodRatioTest}, aovm::NestedModels{M}) where {M <: MixedModel} = 
    lrt_nested(aovm, dof.(aovm.model), deviance_mixedmodel.(aovm.model), 1)

#=
function anova(::Type{LikelihoodRatioTest}, 
                models::Vararg{<: GeneralizedLinearMixedModel}; 
                check::Bool = true,
                kwargs...)
    # need new nestedmodels
end
=#

# Compare to GLM
anova(m0::M, m1::T, ms::Vararg{T}; type::Int = 1, kwargs...) where {M <: GLM_MODEL, T <: MixedModel} = anova(LRT, m0, m1, ms...; kwargs...)
function anova(
                ::Type{LikelihoodRatioTest}, 
                m0::M,
                m::T,
                ms::Vararg{T};
                check::Bool = true
            ) where {M <: GLM_MODEL, T <: MixedModel}
    # Contain _isnested (by QR) and test on formula
    check && (_iscomparable(m0, m) || throw(
        ArgumentError("""Models are not comparable: are the objectives, data and, where appropriate, the link and family the same?""")))
    check && (_iscomparable(m, ms...) || throw(
        ArgumentError("""Models are not comparable: are the objectives, data and, where appropriate, the link and family the same?""")))
    m = [m, ms...]
    df = dof.(m)
    ord = sortperm(df)
    df = (dof(m0), df[ord]...)
    models = (m0, m[ord]...)
    # isnested is not part of _iscomparable:  
    # isnested = true 
    # dev = (_criterion(m0), deviance.(models[2:end])...)
    dev = deviance_mixedmodel.(models)
    lrt_nested(MixedAovModels{Union{M, T}, length(models)}(models), df, dev, 1)
end

anova(::Type{LikelihoodRatioTest}, aovm::MixedAovModels{M}) where {M <:  Union{GLM_MODEL, MixedModel}} = 
    lrt_nested(aovm, df = dof.(aovm.model), deviance_mixedmodel.(aovm.model), 1)

# =================================================================================================================================
# Fit new models

"""
    anova_lme(f::FormulaTerm, tbl; test::Type{<: GoodnessOfFit} = FTest, keyword_arguments...)

    anova_lme(test::Type{<: GoodnessOfFit}, f::FormulaTerm, tbl; keyword_arguments...)

    anova(test::Type{<: GoodnessOfFit}, ::Type{<: LinearMixedModel}, f::FormulaTerm, tbl; keyword_arguments...)

ANOVA for linear mixed-effect models.

# Arguments
* `f`: a `Formula`.
* `tbl`: a `Tables.jl` compatible data.
* `test`: `GoodnessOfFit`. The default is `FTest`.

# Keyword arguments
* `test`: `GoodnessOfFit`. The default is `FTest`.
* `type`: type of anova (1, 2 or 3). Default value is 1.
* `adjust_sigma`: whether adjust σ to match that of linear mixed-effect model fitted by REML. The result will be slightly deviated from that of model fitted by REML.

# Other keyword arguments
* `wts = []`
* `contrasts = Dict{Symbol,Any}()`
* `progress::Bool = true`
* `REML::Bool = true`

`anova_lme` generate a `LinearMixedModel` fitted with REML if applying [`FTest`](@ref); otherwise, a model fitted with ML.
!!! note
    The result with `adjust_sigma` will be slightly deviated from that of model fitted directly by REML.
"""
anova_lme(f::FormulaTerm, tbl; 
        test::Type{<: GoodnessOfFit} = FTest,
        kwargs...) = 
    anova(test, LinearMixedModel, f, tbl; kwargs...)

anova_lme(test::Type{<: GoodnessOfFit}, f::FormulaTerm, tbl; 
        kwargs...) = 
    anova(test, LinearMixedModel, f, tbl; kwargs...)

function anova(test::Type{<: GoodnessOfFit}, ::Type{<: LinearMixedModel}, f::FormulaTerm, tbl; 
        wts = [], 
        contrasts = Dict{Symbol,Any}(), 
        progress::Bool = true, 
        REML::Bool = test == FTest ? true : false, 
        kwargs...)
    model = lme(f, tbl; wts, contrasts, progress, REML)
    anova(test, model; kwargs...)
end

"""
    lme(f::FormulaTerm, tbl; wts, contrasts, progress, REML)

An alias for `fit(LinearMixedModel, f, tbl; wts, contrasts, progress, REML)`.
"""
lme(f::FormulaTerm, tbl; 
    wts = [], 
    contrasts = Dict{Symbol, Any}(), 
    progress::Bool = true, 
    REML::Bool = false) = 
    fit(LinearMixedModel, f, tbl; 
        wts, contrasts, progress, REML)

"""
    glme(f::FormulaTerm, tbl, d::Distribution, l::Link; kwargs...)

An alias for `fit(GeneralizedLinearMixedModel, f, tbl, d, l; kwargs...)`
"""
glme(f::FormulaTerm, tbl, d::Distribution = Normal(), l::Link = canonicallink(d);
    kwargs...) = 
    fit(GeneralizedLinearMixedModel, f, tbl, d, l; kwargs...)
