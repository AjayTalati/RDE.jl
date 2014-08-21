### Ornstein-Uhlenbeck process: dy_t = -λ*y_t*dt+σ*dx_t, y_0 = 0
immutable OrnsteinUhlenbeck <: ContinuousUnivariateStochasticProcess
  λ::Float64 # drift parameter
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
function ito(y::Float64, dx::Float64, p::OrnsteinUhlenbeck)
  λδ::Float64 = p.λ/(p.x.n-1)
  expmλδ::Float64 = exp(-λδ)
  y*expmλδ+(p.σ)*dx*(1-expmλδ)/λδ  
end

### Ito map which takes a vector of increments dx as input, i.e. a vector of fractional Guassian noise samples
function ito!(y::Vector{Float64}, dx::Vector{Float64}, p::OrnsteinUhlenbeck)
  y[1] = ito(0., dx[1], p)
  for i = 2:p.x.n-1
    y[i] = ito(y[i-1], dx[i], p)
  end
  y
end

ito(dx::Vector{Float64}, p::OrnsteinUhlenbeck) = ito!(Array(Float64, p.x.n-1), dx, p)

### Inverse Ito map takes two successive values of the solution vector and returns a rough path increment dx
function invito(y0::Float64, y1::Float64, p::OrnsteinUhlenbeck)
  λδ::Float64 = p.λ/(p.x.n-1)
  expmλδ::Float64 = exp(-λδ)
  λδ*(y1-y0*expmλδ)/(p.σ*(1-expmλδ))
end

function invito!(dx::Vector{Float64}, y::Vector{Float64}, p::OrnsteinUhlenbeck)
  pnmone::Int64 = p.x.n-1
  λδ::Float64 = p.λ/pnmone
  expmλδ::Float64 = exp(-λδ)

  dx[1] = λδ*y[1]/(p.σ*(1-expmλδ))
  for i = 2:pnmone
    dx[i] = λδ*(y[i]-y[i-1]*expmλδ)/(p.σ*(1-expmλδ))
  end

  dx
end

invito(y::Vector{Float64}, p::OrnsteinUhlenbeck) = invito!(Array(Float64, p.x.n-1), y, p)

# Log-pdf of increments of linearly interpolated rough path x
function logpdf(y::Vector{Float64}, p::OrnsteinUhlenbeck; C::Matrix{Float64}=autocov(convert(FGN, p.x), p.x.n-1))
  δ::Float64 = 1/(p.x.n-1)
  λδ::Float64 = p.λ*δ
  δ*logpdf(MvNormal(C), invito(y, p))+log(λδ/(p.σ*(1-exp(-λδ))))
end

function logpdf(y::Vector{Float64}, p::OrnsteinUhlenbeck, logdetC::Float64, yPy::Float64, lPl::Float64, yPl::Float64)
  δ::Float64 = 1/(p.x.n-1)
  λδ::Float64 = p.λ*δ
  expmλδ::Float64 = exp(-λδ)
  phiinv::Float64 = (λδ/(p.σ*(1-expmλδ)))^2
  0.5*(-δ*(phiinv*(yPy+exp(-2*λδ)*lPl-2*expmλδ*yPl)+logdetC)-log(2*pi)+log(phiinv))
end

# Approximate MLE estimator of drift parameter of OU process with FBM noise
function approx_mle_ou_drift(y::Vector{Float64}, x::Union(BrownianMotion, FBM))
  ly::Vector{Float64} = [0, y[1:end-1]]
  pnmone::Int64 = x.n-1
  P::Matrix{Float64} = inv(autocov(convert(FGN, x), pnmone))
  pnmone*log((ly'*P*ly)/(ly'*P*y))
end

#approx_mle_ou_diffusion

#approx_mle_ou
