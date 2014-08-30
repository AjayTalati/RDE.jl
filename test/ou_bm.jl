using Base.Test
using Brownian, RDE

### Simulation 1: comparing exact and approx MLE estimators of OU-BM parameters

# Specify Brownian motion
t = 0:1/2^2:5;
bm = BrownianMotion(t);

# Specify OU process driven by the Brownian motion
λ = 3.;
σ = 0.1;
ou = OU(λ, σ, bm);

# Generate data from the OU process via exact simulation using Gillespie's algorithm
data = rand(ou, 2.);

# Compute exact MLE estimator of of (λ, σ) parameters of OU-BM
exact_mle_ou(bm, data);

# Compute approximate MLE estimator of of (λ, σ) parameters of OU-BM
approx_mle_ou(bm, data);

# Compare exact MLE estimate of λ via BM with approximate MLE estimate of λ via FBM
@test_approx_eq_eps exact_mle_ou_drift(bm, data) approx_mle_ou_drift(convert(FBM, bm), data) 1e-8

# Compare approximate MLE estimate of σ via BM with approximate MLE estimate of σ via FBM
@test_approx_eq_eps approx_mle_ou_diffusion(bm, data) approx_mle_ou_diffusion(convert(FBM, bm), data) 1e-8
