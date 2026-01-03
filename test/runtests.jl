using AnovaMixedModels, CSV, DataFrames, CategoricalArrays
using AnovaGLM: glm, lm 
using AnovaMixedModels: dof_residual_pred
using Test
import Base.isapprox

test_show(x) = show(IOBuffer(), x)
macro test_error(x)
    return quote
        try 
            $x
            false
        catch e
            @error e
            true
        end
    end
end

const anova_datadir = joinpath(dirname(@__FILE__), "..", "data")

"Examples from https://m-clark.github.io/mixed-models-with-R/"
gpa = CSV.read(joinpath(anova_datadir, "gpa.csv"), DataFrame)
transform!(gpa, 
    7 => x->replace(x, "yes" => true, "no" => false, "NA" => missing), 
    4 => x->categorical(x, levels = ["1 hour", "2 hours", "3 hours"], ordered = true),
    renamecols = false)
transform!(gpa, [1, 2, 5, 7] .=> categorical, renamecols = false)

nurses = CSV.read(joinpath(anova_datadir, "nurses.csv"), DataFrame)
transform!(nurses, 
    [1, 2, 3, 4, 6, 9, 11] .=> categorical, 
    10 => x->categorical(x, levels = ["small", "medium", "large"], ordered = true), 
    renamecols = false)

"Data from R package datarium"
anxiety = CSV.read(joinpath(anova_datadir, "anxiety.csv"), DataFrame)
transform!(anxiety, :id => categorical, renamecols = false)

"Data from R package HSAUR2"
toenail = CSV.read(joinpath(anova_datadir, "toenail.csv"), DataFrame)
transform!(toenail, [1, 3] .=> categorical, renamecols = false)
transform!(toenail, 2 => ByRow(x -> x == "none or mild"), renamecols = false)

# custimized approx
isapprox(x::NTuple{N, Float64}, y::NTuple{N, Float64}, atol::NTuple{N, Float64} = x ./ 1000) where N = 
    all(map((a, b, c)->isapprox(a, b, atol = c > eps(Float64) ? c : eps(Float64)), x, y, atol))

@testset "AnovaMixedModels.jl" begin
    @testset "LinearMixedModel" begin
        @testset "One random effect on intercept" begin
            lmm1 = lmm(@formula(gpa ~ occasion + sex + job + (1|student)), gpa)
            lms = nestedmodels(LinearMixedModel, @formula(gpa ~ occasion + sex + job + (1|student)), gpa)
            lmf = FullModel(lmm1, 1, true, true)
            lmm2 = lmm(@formula(gpa ~ occasion * sex + job + (1|student)), gpa)
            lmm3 = lmm(@formula(gpa ~ occasion * sex * job + (1|student)), gpa)
            aovlmm3 = anova_lmm(@formula(score ~ group * time + (1|id)), anxiety, type = 3)
            aovlmm2 = anova_lmm(@formula(gpa ~ occasion * sex + job + (1|student)), gpa, type = 2, REML = true)
            aovlmm1 = anova_lmm(LRT, @formula(score ~ group * time + (1|id)), anxiety)
            global aovf = anova(lmm1)
            global aovlr = anova(lmm1, lmm2, lmm3)
            global aovn = anova(lms)
            global aovff = anova(lmf)
            global aovfl = anova(lmf; test = LRT)
            global aov3 = anova(lmm1, type = 3)
            lr = MixedModels.likelihoodratiotest(lmm1, lmm2, lmm3)
            @test !(@test_error test_show(aovf))
            @test !(@test_error test_show(aovlr))
            @test !(@test_error test_show(aov3))
            @test length(aovfl.anovamodel.model) == 5
            @test anova_type(aov3) == 3
            @test first(nobs(aovlr)) == nobs(lmm1)
            @test dof(aovf) == dof(aovff)
            @test dof_residual(aovf) == (997, 997, 198, 997)
            @test isapprox(teststat(aovlmm2), (28731.976842989556, 635.1750307060438, 17.210623194472724, 46.16338718571525, 14.321381335479893))
            @test isapprox(teststat(aovf), (28899.81231026455, 714.3647106859062, 19.19804098862722, 45.425989531937134))
            @test isapprox(deviance(aovlr), tuple(lr.deviance...))
            @test isapprox(filter(!isnan, pval(aovlr)), tuple(lr.pvalues...)[2:end])
            @test aovlmm3.anovamodel.model.optsum.REML 
            @test !last(aovlmm1.anovamodel.model).optsum.REML
            @test prednames(lmm2) == ["(Intercept)",  "occasion", "sex", "job", "occasion & sex"]
        end 
        @testset "Cross random effect on slope and intercept" begin
            lmm1 = lmm(@formula(gpa ~ occasion + (1|student)), gpa)
            lmm2 = lmm(@formula(gpa ~ occasion + (occasion|student)), gpa)
            global aovlr = anova(lmm1, lmm2)
            global aovf = anova(lmm2)
            lr = MixedModels.likelihoodratiotest(lmm1, lmm2)
            @test !(@test_error test_show(aovlr))
            @test !(@test_error test_show(aovf))
            @test first(nobs(aovlr)) == nobs(lmm1)
            @test dof(aovlr) == tuple(lr.dof...) .- 1
            @test dof_residual(aovf) == (1000, 198)
            @test isapprox(deviance(aovlr), tuple(lr.deviance...))
            @test isapprox(filter(!isnan, pval(aovlr)), tuple(lr.pvalues...)[2:end])
        end
        @testset "Nested random effects" begin
            lmm1 = lmm(@formula(stress ~ age  + sex + experience + treatment + wardtype + hospsize + (1|hospital)), nurses)
            lmm2 = lmm(@formula(stress ~ age  + sex + experience + treatment + wardtype + hospsize + (1|hospital/wardid)), nurses)
            global aov = anova(lmm1, lmm2)
            lr = MixedModels.likelihoodratiotest(lmm1, lmm2)
            @test !(@test_error test_show(aov))
            @test first(nobs(aov)) == nobs(lmm1)
            @test dof(aov) == tuple(lr.dof...) .- 1
            @test dof_residual_pred(lmm2) == (897, 897, 897, 897, 73, 73, 22)
            @test isapprox(deviance(aov), tuple(lr.deviance...))
            @test isapprox(filter(!isnan, pval(aov)), tuple(lr.pvalues...)[2:end])
        end
        #=
        @testset "Random effects on slope and intercept" begin
            aov1 = anova_lmm(@formula(extro ~ open + agree + social + (1|school) + (1|class)), school, REML = true)
            aov2 = anova_lmm(@formula(extro ~ open + agree + social + (open|school) + (open|class)), school, REML = true)
            @test aov1.stats.type == 1
            @test aov2.stats.nobs == 1200
            @test all(aov1.stats.dof .== (1, 1, 1, 1))
            @test all(aov2.stats.resdof .== (1192, 2, 1192, 1192))
            @test isapprox(aov1.stats.fstat, (208.372624411485, 1.673863453540288, 0.3223490172035519, 0.32161077338130784))
            @test isapprox(aov2.stats.pval, (1.5264108749941817e-43, 0.32443726673109696, 0.5622508640865989, 0.5681019984771887))
        end
    
        @testset "Nested random effects on intercept" begin
            aov = anova_lmm(@formula(extro ~ open + agree + social + (1|school) + (1|school&class)), school)
            @test aov.stats.type == 1
            @test aov.stats.nobs == 1200
            @test all(aov.stats.dof .== (1, 1, 1, 1))
            @test all(aov.stats.resdof .== (1173, 1173, 1173, 1173))
            @test isapprox(aov.stats.fstat, (227.24904612916632, 1.4797389900418527, 1.8026931823189165, 0.08511022388038622))
            @test isapprox(aov.stats.pval, (4.508653887500674e-47, 0.22406007872324163, 0.1796470040402073, 0.7705396333261272))
        end
    
        @testset "Nested random effects on slope" begin
            aov = anova_lmm(@formula(extro ~ open + agree + social + (open|school) + (open|school&class)), school)
            @test aov.stats.type == 1
            @test aov.stats.nobs == 1200
            @test all(aov.stats.dof .== (1, 1, 1, 1))
            @test all(aov.stats.resdof .== (1174, 4, 1174, 1174))
            @test isapprox(aov.stats.fstat, (250.00542522864583, 1.2322678515772565, 2.1135395635863543, 0.10258998684862923))
            @test isapprox(aov.stats.pval, (3.364459604379112e-51, 0.3292014599294774, 0.1462685388135273, 0.748800408618393))
        end
        =#
    end
    
    @testset "LinearModel and LinearMixedModel" begin
        lm1 = lm(@formula(score ~ group * time), anxiety)
        lmm1 = lmm(@formula(score ~ group * time + (1|id)), anxiety)
        lmm2 = lmm(@formula(score ~ group * time + (group|id)), anxiety)
        global aov = anova(lm1, lmm1, lmm2)
        lr = MixedModels.likelihoodratiotest(lm1, lmm1, lmm2)
        @test !(@test_error test_show(aov))
        @test first(nobs(aov)) == nobs(lmm1)
        @test dof(aov) == tuple(lr.dof...) .- 1
        @test isapprox(deviance(aov), -2 .* tuple(lr.loglikelihood...))
        @test isapprox(pval(aov)[2:end], tuple(lr.pvalues...)[2:end])
    end
    
    @testset "GeneralizedLinearModel and GeneralizedLinearMixedModel" begin
        glm0 = glm(@formula(outcome ~ visit), toenail, Binomial(), LogitLink())
        glm1 = glm(@formula(outcome ~ visit + treatment), toenail, Binomial(), LogitLink())
        glmm1 = glmm(@formula(outcome ~ visit + treatment + (1|patientID)), toenail, Binomial(), LogitLink(), nAGQ=20, wts = ones(Float64, size(toenail, 1)))
        glmm2 = glmm(@formula(outcome ~ visit * treatment + (1|patientID)), toenail, Binomial(), LogitLink(), nAGQ=20, wts = ones(Float64, size(toenail, 1)))
        global aov = anova(glm1, glmm1, glmm2)
        global aovg = anova(glm0, glm1)
        lr = MixedModels.likelihoodratiotest(glm1, glmm1, glmm2)
        @test !(@test_error test_show(aov))
        @test !(@test_error test_show(aovg))
        @test first(nobs(aov)) == nobs(glmm1)
        @test dof(aov) == tuple(lr.dof...)
        @test isapprox(deviance(aov), -2 .* tuple(lr.loglikelihood...))
        @test isapprox(pval(aov)[2:end], tuple(lr.pvalues...)[2:end])
    end    
end
