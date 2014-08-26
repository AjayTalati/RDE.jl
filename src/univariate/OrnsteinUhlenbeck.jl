### Ornstein-Uhlenbeck process: dy_t = -λ*y_t*dt+σ*dx_t, y_0 = 0
immutable OrnsteinUhlenbeck <: ContinuousUnivariateStochasticProcess
  λ::Float64 # drift parameter (also known as inverse relaxation time)
  σ::Float64 # diffusion parameter
  x::Union(BrownianMotion, FBM) # rough path, which is either Brownian motion or fractional Brownian motion

  function OrnsteinUhlenbeck(λ::Float64, σ::Float64, x::Union(BrownianMotion, FBM))
    λ > 0. || error("Parameter λ must be positive.")
    σ > 0. || error("Parameter σ must be positive.")
    new(λ, σ, x)
  end
end

OrnsteinUhlenbeck(λ::Float64, x::Union(BrownianMotion, FBM)) = OrnsteinUhlenbeck(λ, 1., x)
OrnsteinUhlenbeck(x::Union(BrownianMotion, FBM)) = OrnsteinUhlenbeck(1., 1., x)

const OU = OrnsteinUhlenbeck

### Routine for the exact simulation of OU process driven by Brownian motion started at 0.
### D.T. Gillespie, Exact Numerical Simulation of the Ornstein-Uhlenbeck Process and its Integral, Physical Review E,
### 54 (2), 1996, pp. 2084-2091.
function rand!(y::Vector{Float64}, p::OrnsteinUhlenbeck, p0::Float64)
  d = Normal()
  for i = 1:p.x.n-1
    λt = p.λ*p.x.t[i+1]
    y[i] = exp(-λt)*p0+p.σ*sqrt(0.5*(1-exp(-2*λt))/p.λ)*rand(d)
  end

  y
end

rand(p::OrnsteinUhlenbeck, p0::Float64) = rand!(Array(Float64, p.x.n-1), p, p0)

### Ito map from rough path increment dx to the next iteration of the solution given the previous iteration y of the
### solution
function ito(p::OrnsteinUhlenbeck, y::Float64, dx::Float64)
  λδ::Float64 = p.λ*p.x.t[end]/(p.x.n-1)
  expmλδ::Float64 = exp(-λδ)
  y*expmλδ+(p.σ)*dx*(1-expmλδ)/λδ  
end

### Ito map which takes a vector of increments dx as input, i.e. a vector of fractional Guassian noise samples
function ito!(y::Vector{Float64}, p::OrnsteinUhlenbeck, dx::Vector{Float64})
  y[1] = ito(p, 0., dx[1])
  for i = 2:p.x.n-1
    y[i] = ito(p, y[i-1], dx[i])
  end
  y
end

ito(p::OrnsteinUhlenbeck, dx::Vector{Float64}) = ito!(Array(Float64, p.x.n-1), p, dx)

### Inverse Ito map takes two successive values of the solution vector and returns a rough path increment dx
function invito(p::OrnsteinUhlenbeck, y0::Float64, y1::Float64)
  λδ::Float64 = p.λ*p.x.t[end]/(p.x.n-1)
  expmλδ::Float64 = exp(-λδ)
  λδ*(y1-y0*expmλδ)/(p.σ*(1-expmλδ))
end

function invito!(dx::Vector{Float64}, p::OrnsteinUhlenbeck, y::Vector{Float64})
  pnmone::Int64 = p.x.n-1
  λδ::Float64 = p.λ*p.x.t[end]/pnmone
  expmλδ::Float64 = exp(-λδ)

  dx[1] = λδ*y[1]/(p.σ*(1-expmλδ))
  for i = 2:pnmone
    dx[i] = λδ*(y[i]-y[i-1]*expmλδ)/(p.σ*(1-expmλδ))
  end

  dx
end

invito(p::OrnsteinUhlenbeck, y::Vector{Float64}) = invito!(Array(Float64, p.x.n-1), p, y)

### Auxiliary function that computes the quadratic form of log-pdf of increments of linearly interpolated rough path x
function quad_ou(x::Union(BrownianMotion, FBM), λ::Float64, yPy::Float64, lPl::Float64, yPl::Float64)
  expmλδ::Float64 = exp(-λ*x.t[end]/(x.n-1))  
  yPy+abs2(expmλδ)*lPl-2*expmλδ*yPl
end

### Approximate log-likelihood based on increments of linearly interpolated rough path x
function loglikelihood(p::OrnsteinUhlenbeck, q::Float64, logdetC::Float64)
  pnmone::Int64 = p.x.n-1
  λδ::Float64 = p.λ*p.x.t[end]/pnmone
  invφ::Float64 = abs2(λδ/(p.σ*(1-exp(-λδ))))
  0.5*(-invφ*q-logdetC+pnmone*(log(invφ)-log(2*pi)))
end

loglikelihood(p::OrnsteinUhlenbeck, yPy::Float64, lPl::Float64, yPl::Float64, logdetC::Float64) =
  loglikelihood(p, quad_ou(p.x, p.λ, yPy, lPl, yPl), logdetC)

function loglikelihood(p::OrnsteinUhlenbeck, y::Vector{Float64}, C::Matrix{Float64}=autocov(convert(FGN, p.x), p.x.n-1))
  pnmone::Int64 = p.x.n-1
  λδ::Float64 = p.λ*p.x.t[end]/pnmone
  logpdf(MvNormal(C), invito(y, p))+pnmone*log(λδ/(p.σ*(1-exp(-λδ))))
end

### Approximate MLE estimator of drift parameter of OU process with FBM noise
approx_mle_ou_drift(x::FBM, lPl::Float64, yPl::Float64) = log(lPl/yPl)*(x.n-1)/x.t[end]

function approx_mle_ou_drift(x::FBM, y::Vector{Float64}, y0::Float64=0.)
  l::Vector{Float64} = [y0, y[1:end-1]]
  Pl::Vector{Float64} = inv(autocov(convert(FGN, x), x.n-1))*l
  approx_mle_ou_drift(x, dot(l, Pl), dot(y, Pl))
end

### Exact MLE estimator of drift parameter of OU process with BM noise
exact_mle_ou_drift(x::BrownianMotion, ll::Float64, yl::Float64) = log(ll/yl)*(x.n-1)/x.t[end]

function exact_mle_ou_drift(x::BrownianMotion, y::Vector{Float64}, y0::Float64=0.)
  l::Vector{Float64} = [y0, y[1:end-1]]
  exact_mle_ou_drift(x, dot(l, l), dot(l, y))
end

### Approximate MLE estimator of diffusion parameter of OU process with FBM noise
function approx_mle_ou_diffusion(x::Union(BrownianMotion, FBM), λ::Float64, q::Float64)
  pnmone::Int64 = x.n-1
  λδ::Float64 = λ*x.t[end]/pnmone
  λδ*sqrt(q/pnmone)/(1-exp(-λδ))
end

approx_mle_ou_diffusion(x::FBM, λ::Float64, yPy::Float64, lPl::Float64, yPl::Float64) =
  approx_mle_ou_diffusion(x, λ, quad_ou(x, λ, yPy, lPl, yPl))

function approx_mle_ou_diffusion(x::FBM, yPy::Float64, lPl::Float64, yPl::Float64)
  λ::Float64 = approx_mle_ou_drift(x, lPl, yPl)
  approx_mle_ou_diffusion(x, λ, yPy, lPl, yPl)
end

function approx_mle_ou_diffusion(x::FBM, y::Vector{Float64}, y0::Float64=0.)
  l::Vector{Float64} = [y0, y[1:end-1]]
  P::Matrix{Float64} = inv(autocov(convert(FGN, x), x.n-1))
  Pl::Vector{Float64} = P*l
  approx_mle_ou_diffusion(x, dot(y, P*y), dot(l, Pl), dot(y, Pl))
end

### Exact MLE estimator of diffusion parameter of OU process with BM noise
function exact_mle_ou_diffusion(x::BrownianMotion, λ::Float64, q::Float64)
  pnmone::Int64 = x.n-1
  sqrt(2*λ*q/(pnmone*(1-exp(-2*λ*x.t[end]/pnmone))))
end

exact_mle_ou_diffusion(x::BrownianMotion, λ::Float64, yy::Float64, ll::Float64, yl::Float64) =
  exact_mle_ou_diffusion(x, λ, quad_ou(x, λ, yy, ll, yl))

function exact_mle_ou_diffusion(x::BrownianMotion, yy::Float64, ll::Float64, yl::Float64)
  λ::Float64 = exact_mle_ou_drift(x, ll, yl)
  exact_mle_ou_diffusion(x, λ, yy, ll, yl)
end

function exact_mle_ou_diffusion(x::BrownianMotion, y::Vector{Float64}, y0::Float64=0.)
  l::Vector{Float64} = [y0, y[1:end-1]]
  exact_mle_ou_diffusion(x, dot(y, y), dot(l, l), dot(l, y))
end

### Approximate MLE estimator of diffusion parameter of OU process with BM noise
function approx_mle_ou_diffusion(x::BrownianMotion, λ::Float64, yy::Float64, ll::Float64, yl::Float64)
  pnmone::Int64 = x.n-1
  δ::Float64 = x.t[end]/pnmone
  λ*sqrt(δ*quad_ou(x, λ, yy, ll, yl)/pnmone)/(1-exp(-λ*δ))
end

function approx_mle_ou_diffusion(x::BrownianMotion, yy::Float64, ll::Float64, yl::Float64)
  λ::Float64 = exact_mle_ou_drift(x, ll, yl)
  approx_mle_ou_diffusion(x, λ, yy, ll, yl)
end

function approx_mle_ou_diffusion(x::BrownianMotion, y::Vector{Float64}, y0::Float64=0.)
  l::Vector{Float64} = [y0, y[1:end-1]]
  approx_mle_ou_diffusion(x, dot(y, y), dot(l, l), dot(l, y))
end

### Approximate MLE estimator of drift and diffusion parameters of OU process with FBM noise
function approx_mle_ou(x::FBM, yPy::Float64, lPl::Float64, yPl::Float64)
  λ::Float64 = approx_mle_ou_drift(x, lPl, yPl)
  [λ, approx_mle_ou_diffusion(x, λ, quad_ou(x, λ, yPy, lPl, yPl))]
end

function approx_mle_ou(x::FBM, y::Vector{Float64}, y0::Float64=0.)
  l::Vector{Float64} = [y0, y[1:end-1]]
  P::Matrix{Float64} = inv(autocov(convert(FGN, x), x.n-1))
  Pl::Vector{Float64} = P*l
  lPl::Float64 = dot(l, Pl)
  yPl::Float64 = dot(y, Pl)
  [approx_mle_ou_drift(x, lPl, yPl), approx_mle_ou_diffusion(x, dot(y, P*y), lPl, yPl)]
end

### Exact MLE estimator of diffusion parameter of OU process with BM noise
exact_mle_ou(x::BrownianMotion, yy, ll, yl) = [exact_mle_ou_drift(x, ll, yl), exact_mle_ou_diffusion(x, yy, ll, yl)]

function exact_mle_ou(x::BrownianMotion, y::Vector{Float64}, y0::Float64=0.)
  l::Vector{Float64} = [y0, y[1:end-1]]
  ll::Float64 = dot(l, l)
  yl::Float64 = dot(l, y)
  [exact_mle_ou_drift(x, ll, yl), exact_mle_ou_diffusion(x, dot(y, y), ll, yl)]
end

### Approximate MLE estimator of diffusion parameter of OU process with BM noise
approx_mle_ou(x::BrownianMotion, yy, ll, yl) = [exact_mle_ou_drift(x, ll, yl), approx_mle_ou_diffusion(x, yy, ll, yl)]

function approx_mle_ou(x::BrownianMotion, y::Vector{Float64}, y0::Float64=0.)
  l::Vector{Float64} = [y0, y[1:end-1]]
  ll::Float64 = dot(l, l)
  yl::Float64 = dot(l, y)
  [exact_mle_ou_drift(x, ll, yl), approx_mle_ou_diffusion(x, dot(y, y), ll, yl)]
end
