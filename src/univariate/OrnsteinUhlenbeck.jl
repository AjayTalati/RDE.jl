immutable OrnsteinUhlenbeck <: ContinuousUnivariateStochasticProcess
  l::Float64 # drift parameter
  σ::Float64 # diffusion parameter
  x::Union(BrownianMotion, FBM) # rough path, which is either Brownian motion or fractional Brownian motion

  function OrnsteinUhlenbeck(l::Float64, σ::Float64, x::Union(BrownianMotion, FBM))
    l > 0. || error("Parameter l must be positive.")
    σ > 0. || error("Parameter σ must be positive.")
    new(l, σ, x)
  end
end

OrnsteinUhlenbeck(l::Float64, x::Union(BrownianMotion, FBM)) = OrnsteinUhlenbeck(l, 1., x)
OrnsteinUhlenbeck(x::Union(BrownianMotion, FBM)) = OrnsteinUhlenbeck(1., 1., x)

### Ito map from rough path increment dx to 
function ito(y::Float64, dx::Float64, p::OrnsteinUhlenbeck)
  pnmone::Int64 = p.x.n-1
  n::Int64 = 2^floor(log2(pnmone))

  ld = l/n
  expmld = exp(-ld)
  y*expmld+(p.σ)*dx*(1-expmld)/ld  
end
