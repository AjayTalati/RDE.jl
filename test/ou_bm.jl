using Base.Test
using Brownian, RDE

# Specify Brownian motion
t = 0:1/2^2:5;
bm = BrownianMotion(t);

# Specify OU process driven by the Brownian motion
λ = 3.;
σ = 0.1;
ou = OU(λ, σ, bm);

# Generate data from the OU process via exact simulation using Gillespie's algorithm
data = rand(ou, 2.);

# Compare exact loglikelihood of OU based on the explicit formula and on the logpdf from Distributions
x = rand(bm.n-1);
y = loglikelihood(ou, x; accuracy=:exact, method=:explicit);
z = loglikelihood(ou, x; accuracy=:exact, method=:pdf);
@test_approx_eq_eps y z 1e-8

# Compute exact MLE estimator of of (λ, σ) parameters of OU
fit_mle(OU(bm), data, accuracy=:exact);

# Compute approximate MLE estimator of of (λ, σ) parameters of OU
fit_mle(OU(bm), data, accuracy=:approx);

# Compare exact MLE estimate of λ via BM with approximate MLE estimate of λ via FBM
x = fit_mle(OU(bm), data, parameter=:drift, accuracy=:exact);
y = fit_mle(FOU(bm), data, parameter=:drift);
@test_approx_eq_eps x y 1e-8

# Compare approximate MLE estimate of σ via BM with approximate MLE estimate of σ via FBM
x = fit_mle(OU(bm), data, parameter=:diffusion, accuracy=:approx);
y = fit_mle(FOU(bm), data, parameter=:diffusion);
@test_approx_eq_eps x y 1e-8
