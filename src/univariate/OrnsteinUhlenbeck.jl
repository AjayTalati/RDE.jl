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

### Ito map from rough path increment dx to the next iteration of the solution given the previous iteration y of the
### solution
function ito(y::Float64, dx::Float64, p::OrnsteinUhlenbeck)
  λδ::Float64 = p.λ/(p.x.n-1)
  expmld::Float64 = exp(-λδ)
  y*expmld+(p.σ)*dx*(1-expmld)/λδ  
end

### Ito map which takes a vector of increments dx as input, i.e. a vector of fractional Guassian noise samples
function ito!(y::Vector{Float64}, dx::Vector{Float64}, p::OrnsteinUhlenbeck)
  y[1] = 0.
  for i = 1:p.x.n-1
    y[i+1] = ito(y[i], dx[i], p)
  end
  y
end

ito(dx::Vector{Float64}, p::OrnsteinUhlenbeck) = ito!(Array(Float64, p.x.n), dx, p)

### Inverse Ito map takes two successive values of the solution vecto and returns a rough path increment dx
function invito(y0::Float64, y1::Float64, p::OrnsteinUhlenbeck)
  λδ::Float64 = p.λ/(p.x.n-1)
  expmld::Float64 = exp(-λδ)
  λδ*(y1-y0*expmld)/(p.σ*(1-expmld))
end

function invito!(dx::Vector{Float64}, y::Vector{Float64}, p::OrnsteinUhlenbeck)
  pnmone::Int64 = p.x.n-1
  λδ::Float64 = p.λ/(pnmone)
  expmld::Float64 = exp(-λδ)

  for i = 1:pnmone
    dx[i] = λδ*(y[i+1]-y[i]*expmld)/(p.σ*(1-expmld))
  end

  dx
end

invito(y::Vector{Float64}, p::OrnsteinUhlenbeck) = invito!(Array(Float64, p.x.n-1), y, p)

# Log-pdf of increments of linearly interpolated rough path x
function logpdf(y::Vector{Float64}, p::OrnsteinUhlenbeck)
  pnmone::Int64 = p.x.n-1
  λδ::Float64 = p.λ/pnmone
  logpdf(MvNormal(autocov(convert(FGN, p.x), pnmone)), invito(y, p))/pnmone+log(λδ/(p.σ*(1-exp(-λδ))))
end
