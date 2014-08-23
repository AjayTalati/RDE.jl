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

# Auxiliary functions that compute the quadratic form of log-pdf of increments of linearly interpolated rough path x
function quad_ou(p::OrnsteinUhlenbeck, yPy::Float64, lPl::Float64, yPl::Float64)
  expmλδ::Float64 = exp(-p.λ*p.x.t[end]/(p.x.n-1))  
  yPy+abs2(expmλδ)*lPl-2*expmλδ*yPl
end

# Log-pdf of increments of linearly interpolated rough path x
function logpdf(p::OrnsteinUhlenbeck, q::Float64, logdetC::Float64)
  δ::Float64 = p.x.t[end]/(p.x.n-1)
  λδ::Float64 = p.λ*δ
  invφ::Float64 = abs2(λδ/(p.σ*(1-exp(-λδ))))
  0.5*(-δ*(invφ*q+logdetC)-log(2*pi)+log(invφ))
end

logpdf(p::OrnsteinUhlenbeck, yPy::Float64, lPl::Float64, yPl::Float64, logdetC::Float64) =
  logpdf(p, quad_ou(p, yPy, lPl, yPl), logdetC)

function logpdf(p::OrnsteinUhlenbeck, y::Vector{Float64}, C::Matrix{Float64}=autocov(convert(FGN, p.x), p.x.n-1))
  δ::Float64 = p.x.t[end]/(p.x.n-1)
  λδ::Float64 = p.λ*δ
  δ*logpdf(MvNormal(C), invito(y, p))+log(λδ/(p.σ*(1-exp(-λδ))))
end

# Approximate MLE estimator of drift parameter of OU process with BM noise
approx_mle_ou_drift(x::BrownianMotion, ll::Float64, yl::Float64) = log(ll/yl)*(x.n-1)/x.t[end]

function approx_mle_ou_drift(x::BrownianMotion, y::Vector{Float64}, y0::Float64=0.)
  l::Vector{Float64} = [y0, y[1:end-1]]
  approx_mle_ou_drift(x, dot(l, l), dot(l, y))
end

# Approximate MLE estimator of drift parameter of OU process with FBM noise
approx_mle_ou_drift(x::FBM, lPl::Float64, yPl::Float64) = log(lPl/yPl)*(x.n-1)/x.t[end]

function approx_mle_ou_drift(x::FBM, y::Vector{Float64}, y0::Float64=0.)
  l::Vector{Float64} = [y0, y[1:end-1]]
  Pl::Vector{Float64} = inv(autocov(convert(FGN, x), x.n-1))*l
  approx_mle_ou_drift(x, dot(l, Pl), dot(y, Pl))
end

# Approximate MLE estimator of diffusion parameter of OU process with BM noise
#function approx_mle_ou_diffusion(y::Vector{Float64}, x::BrownianMotion, l::Float64, y0::Float64=0.)
#end

# Approximate MLE estimator of diffusion parameter of OU process with FBM noise
#function approx_mle_ou_diffusion(y::Vector{Float64}, x::FBM, l::Float64, y0::Float64=0.)
#end

#approx_mle_ou
