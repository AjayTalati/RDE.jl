### Ornstein-Uhlenbeck process: dy_t = -l*y_t*dt+σ*dx_t, y_0 = 0
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

### Ito map from rough path increment dx to the next iteration of the solution given the previous iteration y of the
### solution
function ito(y::Float64, dx::Float64, p::OrnsteinUhlenbeck)
  ld::Float64 = l/(p.x.n-1)
  expmld::Float64 = exp(-ld)
  y*expmld+(p.σ)*dx*(1-expmld)/ld  
end

### Ito map which takes a vector of increments dx as input, i.e. a vector of fractional Guassian noise samples
function ito!(y::Vector{Float64}, dx::Vector{Float64}, p::OrnsteinUhlenbeck)
  y[1] = 0.
  for i = 1:p.x.n-1
    y[i+i] = ito(y[i], dx[i], p)
  end
  y
end

ito(dx::Vector{Float64}, p::OrnsteinUhlenbeck) = ito!(Array(Float64, p.x.n), dx, p)

### Inverse Ito map takes two successive values of the solution vecto and returns a rough path increment dx
function invito(y0::Float64, y1::Float64, p::OrnsteinUhlenbeck)
  ld::Float64 = l/(p.x.n-1)
  expmld::Float64 = exp(-ld)
  ld*(y1-y0*expmld)/(p.σ*(1-expmld))
end

function invito!(dx::Vector{Float64}, y::Vector{Float64}, p::OrnsteinUhlenbeck)
  pnmone::Int64 = p.n-1
  ld::Float64 = l/(pnmone)
  expmld::Float64 = exp(-ld)

  for i = 1:pnmone
    dx[i] = ld*()
    # TODO: complete it
  end

  dx
end

UniOUFBMAllInverseIto[y_, l_, s_] := Module[{ld, expmld},
  ld = l/(Length[y]-1);
  expmld = Exp[-ld];
  ld*(Rest[y]-Most[y]*expmld)/(s*(1-expmld))
]