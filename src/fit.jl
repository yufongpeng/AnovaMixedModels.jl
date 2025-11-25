# ==========================================================================================================
predictors(model::T) where {T <: MixedModel} = first(formula(model).rhs).terms

deviance_mixedmodel(model) = -2 * loglikelihood(model)

lhs_no_intercept(lhs::String) = Set([lhs])
lhs_no_intercept(lhs) = Set(filter!(!=("(Intercept)"), lhs))

"""
    dof_residual_pred(model::LinearMixedModel)
    dof_residual_pred(aovm::FullModel{LinearMixedModel})

Compute degrees of freedom (DOF) of residuals for each predictors in a linear mixed effect models.

DOF of residuals are estimated by between-within method. For details, please refer to the documentation or [GLMM FAQ](https://bbolker.github.io/mixedmodels-misc/glmmFAQ.html#why-doesnt-lme4-display-denominator-degrees-of-freedomp-values-what-other-options-do-i-have) for details.

To be noticed, my implementation is a little different from the reference one. 
When intercept is not in fix effects, the reference algorithm regards the first term as intercept; however, there is no such replacement here. 
"""
function dof_residual_pred(model::LinearMixedModel)
    randoms = collect(formula(model).rhs) # ranef formula terms
    fixs = popfirst!(randoms) # fixef formula terms
    isempty(randoms) && (return repeat([nobs(model) - length(fixs.terms)], length(fixs.terms)))
    reterms = reverse(model.reterms) # Vector of ReMat
    fixname = collect(model.feterm.cnames) # fixef name
    n0 = hasintercept(fixs) ? (popfirst!(fixname); 1) : 0

    # Determine affected fix effects for each random effects
    affectfixname = Dict{String, Set{String}}()
    for pair in randoms
        union!(get!(affectfixname, prednames(pair.rhs), Set{String}()), lhs_no_intercept(coefnames(pair.lhs)))
    end 

    # Determines if fix effects vary at each random effects group
    within = mapreduce(hcat, reterms) do remat
        levels = unique(remat.refs)
        map(eachindex(fixname)) do fixid
            all(levels) do level
                vals = view(model.Xymat.wtxy, findall(==(level), remat.refs), fixid + n0)
                all(==(first(vals)), vals)
            end
        end
    end
    within = [within repeat([true], length(fixname))] # observation level

    # Turn affected fix effects into true
    randname = prednames.(getproperty.(reterms, :trm))
    for (key, value) in affectfixname
        id = findfirst(in(value), fixname)
        isnothing(id) || (within[id, findfirst(==(key), randname)] = true)
    end 
    # Find first true for each fix effect, aka level
    level = mapslices(findfirst, within, dims = 2)
    # Number of fix effects per level
    nfixperlv = zeros(Int, length(randname) + 1)
    for i in level
        nfixperlv[i] += 1
    end 

    # dfᵢ = ngroupᵢ - ngroupᵢᵢ₋₁ - nfixᵢ
    ngroupperlv = (n0, length.(getproperty.(reterms, :levels))..., nobs(model)) # number of random effects observation per level
    dfperlv = _diff(ngroupperlv) .- nfixperlv

    # Assign df to each factors based on the level
    assign = asgn(fixs)
    hasintercept(fixs) && popfirst!(assign)
    offset = first(assign) - 1
    dfr = ntuple(last(assign) - offset) do i
        dfperlv[level[findfirst(==(i + offset), assign)]]
    end
    (last(dfperlv), dfr...)
end

dof_residual_pred(aovm::FullModel{<: LinearMixedModel}) = getindex.(Ref(dof_residual_pred(aovm.model)), aovm.pred_id)

"""
    nestedmodels(model::LinearMixedModel; null::Bool = true, <keyword arguments>)

    nestedmodels(::Type{LinearMixedModel}, f::FormulaTerm, tbl; null::Bool = true, wts = [], contrasts = Dict{Symbol, Any}(), verbose::Bool = false, REML::Bool = false)

Generate nested models from a model or modeltype, formula and data.
The null model will be an empty model if the keyword argument `null` is true (default).
"""
function nestedmodels(model::M; null::Bool = true, kwargs...) where {M <: LinearMixedModel}
    f = formula(model)
    range = null ? (0:length(f.rhs[1].terms) - 1) : (1:length(f.rhs[1].terms) - 1)
    assign = asgn(first(f.rhs))
    REML = model.optsum.REML
    pivot = model.feterm.piv
    X = copy(model.X)
    y = copy(model.y)
    T = promote_type(Float64, eltype(y))
    size(X, 2) > 0 && (X = pivot == collect(1:size(X, 2)) ? X : X[:, pivot])
    reterms = deepcopy(model.reterms)
    sqrtwts = copy(model.sqrtwts)
    σ = model.optsum.sigma
    models = map(range) do id
        subf = subformula(f.lhs, f.rhs, id)
        cnames = coefnames(first(subf.rhs))
        # modify X by assign
        select = findall(x->x <= id, assign)
        subX = X[:, select]
        feterms = MixedModels.FeTerm{T}[]
        push!(feterms, MixedModels.FeTerm(subX, isa(cnames, String) ? [cnames] : collect(cnames)))
        feterm = only(feterms)
        Xy = MixedModels.FeMat(feterm, vec(y))
        reweight!(Xy, sqrtwts)
        A, L = createAL(reterms, Xy)
        θ = foldl(vcat, getθ(c) for c in reterms)
        optsum = OptSummary(θ)
        optsum.sigma = isnothing(σ) ? nothing : T(σ)
        lmm = LinearMixedModel(
            subf,
            reterms,
            Xy,
            feterm,
            sqrtwts,
            model.parmap,
            (n = length(y), p = feterm.rank, nretrms = length(reterms)),
            A,
            L,
            optsum,
            )
        fit!(lmm; REML, progress = true)
        lmm
    end
    NestedModels(models..., model)
end

nestedmodels(::Type{<: LinearMixedModel}, f::FormulaTerm, tbl; 
                null::Bool = true, 
                wts = [], 
                contrasts = Dict{Symbol, Any}(), 
                progress::Bool = true, 
                REML::Bool = false) = 
    nestedmodels(fit(LinearMixedModel, f, tbl; wts, contrasts, progress, REML); null)

# For nestedmodels
isnullable(::LinearMixedModel) = true

# Specialized dof_residual
dof_residual(aov::AnovaResult{<: FullModel{<: MixedModel}, FTest}) = aov.otherstat.dof_residual