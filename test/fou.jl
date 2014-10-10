using Base.Test
using Brownian, RDE

# Specify Brownian motion
t = 0:1/2^2:5;
h = 0.75;
fbm = FBM(t, h);

# Specify FOU process
λ = 3.;
σ = 0.1;
fou = FOU(λ, σ, fbm);

# Generate data from the OU process via exact simulation using Gillespie's algorithm
data = rand(fou);
