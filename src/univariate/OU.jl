### Ornstein-Uhlenbeck process dy_t = -λ*y_t*dt+σ*dx_t, y_0 = 0 driveny by Brownian motion x_t
immutable OrnsteinUhlenbeck <: ContinuousUnivariateStochasticProcess
  λ::Float64 # drift parameter (also known as inverse relaxation time)
  σ::Float64 # diffusion parameter
  x::BrownianMotion # rough path, which is Brownian motion

  function OrnsteinUhlenbeck(λ::Float64, σ::Float64, x::BrownianMotion)
    isnan(λ) || λ > 0. || error("Parameter λ must be positive.")
    isnan(σ) || σ > 0. || error("Parameter σ must be positive.")
    new(λ, σ, x)
  end
end

OrnsteinUhlenbeck(x::BrownianMotion) = OrnsteinUhlenbeck(NaN, NaN, x)

const OU = OrnsteinUhlenbeck

### Ornstein-Uhlenbeck process dy_t = -λ*y_t*dt+σ*dx_t, y_0 = 0 driveny by fractional Brownian motion x_t
immutable FractionalOrnsteinUhlenbeck <: ContinuousUnivariateStochasticProcess
  λ::Float64 # drift parameter (also known as inverse relaxation time)
  σ::Float64 # diffusion parameter
  x::FBM # rough path, which is fractional Brownian motion

  function FractionalOrnsteinUhlenbeck(λ::Float64, σ::Float64, x::FBM)
    isnan(λ) || λ > 0. || error("Parameter λ must be positive.")
    isnan(σ) || σ > 0. || error("Parameter σ must be positive.")
    new(λ, σ, x)
  end
end

FractionalOrnsteinUhlenbeck(x::FBM) = FractionalOrnsteinUhlenbeck(NaN, NaN, x)
FractionalOrnsteinUhlenbeck(x::BrownianMotion) = FractionalOrnsteinUhlenbeck(NaN, NaN, convert(FBM, x))

const FOU = FractionalOrnsteinUhlenbeck

typealias BMOrFBM Union(BrownianMotion, FBM)
typealias OUOrFOU Union(OU, FOU)

convert(::Type{FOU}, p::OU) = FOU(p.λ, p.σ, convert(FBM, p.x))

### Auxiliary function for calculating the hypergeometric 1F2 series in the autocovariance of stationary FOU
function ou_fbm_autocov_hypergeom_term_with_product(h::Float64, λ::Float64, s::Float64;
  maxnevals::Int64=0, reltol::Float64=1e-8, abstol::Float64=1e-8)
  local λtimess::Float64 = λ*s
  local series::Float64 = 0.
  local siter::Float64 = 1.
  local k::Int64 = 1
  local pterm::Float64
  local nevals::Int64 = (maxnevals == 0 ? 0 : k)
  local err::Float64 = 1.

  while (0 <= nevals <= maxnevals) && err > reltol*abs(siter) && err > abstol
    pterm = 1.
    for j = 0:(k-1)
      pterm *= (h+0.5+j)*(h+1+j)
    end
    siter += (λtimess^(2*k))/((4^k)*pterm)

    k += 1
    nevals = (maxnevals == 0 ? 0 : k)
    err = abs(series-siter)

    series = siter
  end

  series
end

function ou_fbm_autocov_hypergeom_term_with_gamma(h::Float64, λ::Float64, s::Float64;
  maxnevals::Int64=0, reltol::Float64=1e-8, abstol::Float64=1e-8)
  local twohplusone::Float64 = 2*h+1
  local λtimess::Float64 = λ*s
  local series::Float64 = 1.
  local siter::Float64 = 0.
  local k::Int64 = 1
  local twok::Float64
  local nevals::Int64 = (maxnevals == 0 ? 0 : k)
  local err::Float64 = 1.

  while (0 <= nevals <= maxnevals) && err > reltol*abs(siter) && err > abstol
    twok = 2*k
    siter += (λtimess^twok)/gamma(twohplusone+twok)

    k += 1
    nevals = (maxnevals == 0 ? 0 : k)
    err = abs(series-siter)

    series = siter
  end

  1+gamma(twohplusone)*series
end

function ou_fbm_autocov_hypergeom_term(h::Float64, λ::Float64, s::Float64;
  maxnevals::Int64=0, reltol::Float64=1e-8, abstol::Float64=1e-8, method::Symbol=:product)
  if method == :product
    ou_fbm_autocov_hypergeom_term_with_product(h, λ, s; maxnevals=maxnevals, reltol=reltol, abstol=abstol)
  elseif method == :gamma
    ou_fbm_autocov_hypergeom_term_with_gamma(h, λ, s; maxnevals=maxnevals, reltol=reltol, abstol=abstol)
  else
    error("Accepted methods are :product and :gamma.")
  end
end

autocov_hypergeom_term(p::FOU, s::Float64;
  maxnevals::Int64=0, reltol::Float64=1e-8, abstol::Float64=1e-8, method::Symbol=:product) =
  ou_fbm_autocov_hypergeom_term(p.h, p.λ, s; maxnevals=maxnevals, reltol=reltol, abstol=abstol, method=method)

### Autocovariance of stationary FOU
function ou_fbm_autocov_with_single_sum(h::Float64, λ::Float64, σ::Float64, s::Float64;
  maxnevals::Int64=0, reltol::Float64=1e-8, abstol::Float64=1e-8)
  local twoh::Float64 = 2*h
  local λtimess::Float64 = λ*s
  local series::Float64 = 1.
  local siter::Float64 = 0.
  local k::Int64 = 0
  local twok::Float64
  local twokplustwoh::Float64
  local nevals::Int64 = (maxnevals == 0 ? 0 : 1)
  local err::Float64 = 1.

  while (0 <= nevals <= maxnevals) && err > reltol*abs(siter) && err > abstol
    twok = 2*k
    twokplustwoh = twoh+twok
    siter += (1/(gamma(twok+1)*(λ^twoh))-(s^twoh)/(twokplustwoh*gamma(twokplustwoh)))*(λtimess^twok)

    k += 1
    nevals = (maxnevals == 0 ? 0 : nevals+1)
    err = abs(series-siter)

    series = siter
  end

  h*gamma(twoh)*abs2(σ)*series
end

function ou_fbm_autocov_with_hypergeom(h::Float64, λ::Float64, σ::Float64, s::Float64;
  maxnevals::Int64=0, reltol::Float64=1e-8, abstol::Float64=1e-8, hmethod::Symbol=:product)
  local twoh::Float64 = 2*h
  local λtimess::Float64 = λ*s

  0.5*abs2(σ)*(gamma(twoh+1)*cosh(λtimess)/(λ^twoh)
    -(s^twoh)*ou_fbm_autocov_hypergeom_term(h, λ, s; maxnevals=maxnevals, reltol=reltol, abstol=abstol, method=hmethod))
end

function ou_fbm_autocov(h::Float64, λ::Float64, σ::Float64, s::Float64;
  maxnevals::Int64=0, reltol::Float64=1e-8, abstol::Float64=1e-8, method::Symbol=:sum)
  if method == :sum
    ou_fbm_autocov_with_single_sum(h, λ, σ, s; maxnevals=maxnevals, reltol=reltol, abstol=abstol)
  elseif method == :hypergeom
    ou_fbm_autocov_with_hypergeom(h, λ, σ, s; maxnevals=maxnevals, reltol=reltol, abstol=abstol)
  else
    error("Accepted methods are :sum and :hypergeom.")
  end
end

autocov(p::FOU, s::Float64;
  maxnevals::Int64=0, reltol::Float64=1e-8, abstol::Float64=1e-8, method::Symbol=:sum) =
  ou_fbm_autocov(p.x.h, p.λ, p.σ, s; maxnevals=maxnevals, reltol=reltol, abstol=abstol, method=method)

function autocov!(c::Matrix{Float64}, p::FOU)
  n::Int64 = p.x.n-1

  for i = 1:n
    for j = 1:i
      c[i, j] = autocov(p, abs(p.x.t[j]-p.x.t[i]))
    end
  end

  for i = 1:n
    for j = (i+1):n
      c[i, j] = c[j, i]
    end
  end

  c
end

function autocov(p::FOU)
  n::Int64 = p.x.n-1
  autocov!(Array(Float64, n, n), p)
end

### Routine for the exact simulation of OU process driven by Brownian motion started at 0.
### D.T. Gillespie, Exact Numerical Simulation of the Ornstein-Uhlenbeck Process and its Integral, Physical Review E,
### 54 (2), 1996, pp. 2084-2091.
function rand!(y::Vector{Float64}, p::OU, y0::Float64)
  for i = 1:p.x.n-1
    λt = p.λ*p.x.t[i+1]
    y[i] = exp(-λt)*y0+p.σ*sqrt(0.5*(1-exp(-2*λt))/p.λ)*randn()
  end

  y
end

rand(p::OU, y0::Float64) = rand!(Array(Float64, p.x.n-1), p, y0)

### Routine for the exact simulation of stationary FOU proccess.
### The rand_gp method is based on sampling from FOU when seen as a Gaussian process (GP)
rand_gp(p::FOU) = rand(MvNormal(autocov(p)))

function rand_gp(p::Vector{FOU})
  np = length(p)
  y = Array(Float64, p.x[1].n-1, np)

  for i = 1:np
    y[:, i] = rand_gp(p[i])
  end

  y
end

### Routine for the exact simulation of stationary FOU proccess.
### The rand_chol method is based on Cholesky decomposition.
### G. Schoechtel, Motion of Inertial Particles in Gaussian Fields Driven by an Infinite-Dimensional Fractional Brownian
### Motion, PhD thessis, 2013.
### The complexity of the algorithm is O(n^3), where n is the number of FBM samples.
function rand_chol(p::FOU)
  chol(autocov(p), :L)*randn(p.x.n-1)
end

function rand_chol(p::Vector{FOU})
  np = length(p)
  y = Array(Float64, p.x[1].n-1, np)

  for i = 1:np
    y[:, i] = rand_chol(p[i])
  end

  y
end

### Interface for sampling FOU.
### For the time being, only the method based on GP and on Cholesky decomposition have been implemented.
### Other methods include the Durbin-Levinson, the circular embedding and spectral approximate methods.
function rand(p::FOU; rtype::Symbol=:gp)
  if rtype == :gp
    rand_gp(p)
  elseif rtype == :chol
    rand_chol(p)
  else
    error("Accepted methods are :gp and :chol.")
  end
end

function rand(p::Vector{FOU}; rtype::Symbol=:gp)
  if rtype == :gp
    rand_gp(p)
  elseif rtype == :chol
    rand_chol(p)
  else
    error("Accepted methods are :gp and :chol.")
  end
end

### Ito map from rough path increment dx to the next iteration of the solution given the previous iteration y
function ito(p::OUOrFOU, y::Float64, dx::Float64)
  λδ::Float64 = p.λ*p.x.t[end]/(p.x.n-1)
  expmλδ::Float64 = exp(-λδ)
  y*expmλδ+(p.σ)*dx*(1-expmλδ)/λδ  
end

### Ito map which takes a vector of increments dx as input, i.e. a vector of fractional Guassian noise samples
function ito!(y::Vector{Float64}, p::OUOrFOU, dx::Vector{Float64})
  y[1] = ito(p, 0., dx[1])
  for i = 2:p.x.n-1
    y[i] = ito(p, y[i-1], dx[i])
  end
  y
end

ito(p::OUOrFOU, dx::Vector{Float64}) = ito!(Array(Float64, p.x.n-1), p, dx)

### Inverse Ito map takes two successive values of the solution vector and returns a rough path increment dx
function invito(p::OUOrFOU, y0::Float64, y1::Float64)
  λδ::Float64 = p.λ*p.x.t[end]/(p.x.n-1)
  expmλδ::Float64 = exp(-λδ)
  λδ*(y1-y0*expmλδ)/(p.σ*(1-expmλδ))
end

function invito!(dx::Vector{Float64}, p::OUOrFOU, y::Vector{Float64})
  pnmone::Int64 = p.x.n-1
  λδ::Float64 = p.λ*p.x.t[end]/pnmone
  expmλδ::Float64 = exp(-λδ)

  dx[1] = λδ*y[1]/(p.σ*(1-expmλδ))
  for i = 2:pnmone
    dx[i] = λδ*(y[i]-y[i-1]*expmλδ)/(p.σ*(1-expmλδ))
  end

  dx
end

invito(p::OUOrFOU, y::Vector{Float64}) = invito!(Array(Float64, p.x.n-1), p, y)

### Auxiliary function that computes the quadratic form of log-pdf of increments of linearly interpolated rough path x
function quad_ou(x::BMOrFBM, λ::Float64, yPy::Float64, lPl::Float64, yPl::Float64)
  expmλδ::Float64 = exp(-λ*x.t[end]/(x.n-1))  
  yPy+abs2(expmλδ)*lPl-2*expmλδ*yPl
end

quad(p::OUOrFOU, yPy::Float64, lPl::Float64, yPl::Float64) = quad_ou(p.x, p.λ, yPy, lPl, yPl)

### Exact log-likelihood of OU process
function exact_loglik(p::OU, q::Float64)
  pnmone::Int64 = p.x.n-1
  φ::Float64 = 2*p.λ/(abs2(p.σ)*(1-exp(-2*p.λ*p.x.t[end]/pnmone)))
  0.5*(-φ*q+pnmone*(log(φ)-log(2*pi)))
end

exact_loglik(p::OU, yy::Float64, ll::Float64, yl::Float64) = exact_loglik(p, quad(p, yy, ll, yl))

function exact_loglik_explicit(p::OU, y::Vector{Float64}, y0::Float64=0.)
  l::Vector{Float64} = [y0, y[1:end-1]]
  exact_loglik(p, dot(y, y), dot(l, l), dot(l, y))
end

function exact_loglik_with_pdf(p::OU, y::Vector{Float64}, y0::Float64=0.)
  expmλδ::Float64 = exp(-p.λ*p.x.t[end]/(p.x.n-1))
  logpdf(IsoNormal([y0, y[1:end-1]]*expmλδ, p.σ*sqrt(0.5*(1-abs2(expmλδ))/p.λ)), y)
end

function exact_loglik(p::OU, y::Vector{Float64}, y0::Float64=0.; method::Symbol=:explicit)
  if method == :explicit
    exact_loglik_explicit(p, y, y0)
  elseif method == :pdf
    exact_loglik_with_pdf(p, y, y0)
  else
    error("Accepted methods are :explicit and :pdf.")
  end
end

### Approximate log-likelihood of OU and FOU processes based on increments of linearly interpolated rough path x
function approx_loglik(p::OUOrFOU, q::Float64, logdetC::Float64)
  pnmone::Int64 = p.x.n-1
  λδ::Float64 = p.λ*p.x.t[end]/pnmone
  invξ::Float64 = abs2(λδ/(p.σ*(1-exp(-λδ))))
  0.5*(-invξ*q-logdetC+pnmone*(log(invξ)-log(2*pi)))
end

approx_loglik(p::OU, yPy::Float64, lPl::Float64, yPl::Float64, logdetC::Float64=(p.x.t[end]/(p.x.n-1))^(p.x.n-1)) =
  approx_loglik(p, quad(p, yPy, lPl, yPl), logdetC)

approx_loglik(p::FOU, yPy::Float64, lPl::Float64, yPl::Float64,
  logdetC::Float64=logdet(autocov(convert(FGN, p.x), p.x.n-1))) =
  approx_loglik(p, quad(p, yPy, lPl, yPl), logdetC)

function approx_loglik(p::OUOrFOU, y::Vector{Float64}, C::Matrix{Float64}=autocov(convert(FGN, p.x), p.x.n-1))
  pnmone::Int64 = p.x.n-1
  λδ::Float64 = p.λ*p.x.t[end]/pnmone
  pnew
  logpdf(MvNormal(C), invito(typeof(p)(p.λ, p.σ, p.x), y))+pnmone*log(λδ/(p.σ*(1-exp(-λδ))))
end

### Interface for loglikelihood evaluation of OU process
function loglikelihood(p::OU, mainargs...; accuracy::Symbol=:exact, keyargs...)
  if accuracy == :exact
    exact_loglik(p, mainargs...; keyargs...)
  elseif accuracy == :approx
    approx_loglik(p, mainargs...)
  else
    error("accuracy keyword arg can be either :exact or :approx.")
  end
end

### Interface for loglikelihood evaluation of FOU process
loglikelihood(p::FOU, mainargs...) = approx_loglik(p, mainargs...)

### Exact MLE estimator of drift parameter of OU process
exact_mle_ou_drift(x::BrownianMotion, ll::Float64, yl::Float64) = log(ll/yl)*(x.n-1)/x.t[end]

function exact_mle_ou_drift(x::BrownianMotion, y::Vector{Float64}, y0::Float64=0.)
  l::Vector{Float64} = [y0, y[1:end-1]]
  exact_mle_ou_drift(x, dot(l, l), dot(l, y))
end

### Approximate MLE estimator of drift parameter of OU and FOU processes
approx_mle_ou_drift(x::BMOrFBM, lPl::Float64, yPl::Float64) = log(lPl/yPl)*(x.n-1)/x.t[end]

function approx_mle_ou_drift(x::BMOrFBM, y::Vector{Float64}, y0::Float64=0.)
  l::Vector{Float64} = [y0, y[1:end-1]]
  Pl::Vector{Float64} = inv(autocov(convert(FGN, x), x.n-1))*l
  approx_mle_ou_drift(x, dot(l, Pl), dot(y, Pl))
end

### Interface for MLE evaluation of drift term of OU process
function mle_drift(p::OU, mainargs...; accuracy::Symbol=:exact)
  if accuracy == :exact
    exact_mle_ou_drift(p.x, mainargs...)
  elseif accuracy == :approx
    approx_mle_ou_drift(p.x, mainargs...)
  else
    error("accuracy keyword arg can be either :exact or :approx.")
  end
end

### Interface for MLE evaluation of drift term of FOU process
mle_drift(p::FOU, mainargs...) = approx_mle_ou_drift(p.x, mainargs...)

### Exact MLE estimator of diffusion parameter of OU process
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

### Approximate MLE estimator of diffusion parameter of OU process
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

### Approximate MLE estimator of diffusion parameter of OU and FOU processes
function approx_mle_ou_diffusion(x::BMOrFBM, λ::Float64, q::Float64)
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

### Interface for MLE evaluation of diffusion term of OU process
function mle_diffusion(p::OU, mainargs...; accuracy::Symbol=:exact)
  if accuracy == :exact
    exact_mle_ou_diffusion(p.x, mainargs...)
  elseif accuracy == :approx
    approx_mle_ou_diffusion(p.x, mainargs...)
  else
    error("accuracy keyword arg can be either :exact or :approx.")
  end
end

### Interface for MLE evaluation of diffusion term of FOU process
mle_diffusion(p::FOU, mainargs...) = approx_mle_ou_diffusion(p.x, mainargs...)

### Exact MLE estimator of diffusion parameter of OU process
exact_mle_ou(x::BrownianMotion, yy::Float64, ll::Float64, yl::Float64) =
  [exact_mle_ou_drift(x, ll, yl), exact_mle_ou_diffusion(x, yy, ll, yl)]

function exact_mle_ou(x::BrownianMotion, y::Vector{Float64}, y0::Float64=0.)
  l::Vector{Float64} = [y0, y[1:end-1]]
  ll::Float64 = dot(l, l)
  yl::Float64 = dot(l, y)
  [exact_mle_ou_drift(x, ll, yl), exact_mle_ou_diffusion(x, dot(y, y), ll, yl)]
end

### Approximate MLE estimator of diffusion parameter of OU and FOU processes
approx_mle_ou(x::BrownianMotion, yy::Float64, ll::Float64, yl::Float64) =
  [exact_mle_ou_drift(x, ll, yl), approx_mle_ou_diffusion(x, yy, ll, yl)]

function approx_mle_ou(x::BrownianMotion, y::Vector{Float64}, y0::Float64=0.)
  l::Vector{Float64} = [y0, y[1:end-1]]
  ll::Float64 = dot(l, l)
  yl::Float64 = dot(l, y)
  [exact_mle_ou_drift(x, ll, yl), approx_mle_ou_diffusion(x, dot(y, y), ll, yl)]
end

### Interface for MLE evaluation of drift and diffusion terms of OU process
function mle(p::OU, mainargs...; accuracy::Symbol=:exact)
  if accuracy == :exact
    exact_mle_ou(p.x, mainargs...)
  elseif accuracy == :approx
    approx_mle_ou(p.x, mainargs...)
  else
    error("accuracy keyword arg can be either :exact or :approx.")
  end
end

### Approximate MLE estimator of drift and diffusion parameters of FOU process
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

### Interface for MLE evaluation of drift and diffusion terms of FOU process
mle(p::FOU, mainargs...) = approx_mle_ou(p.x, mainargs...)

### Interface for MLE evaluation of OU process
function fit_mle(p::OU, mainargs...; parameter::Symbol=:both, keyargs...)
  if parameter == :both
    mle(p, mainargs...; keyargs...)
  elseif parameter == :drift
    mle_drift(p, mainargs...; keyargs...)
  elseif parameter == :diffusion
    mle_diffusion(p, mainargs...; keyargs...)
  else
    error("parameter keyword arg can be :both or :drift or :diffusion.")
  end
end

### Interface for MLE evaluation of FOU process
function fit_mle(p::FOU, mainargs...; parameter::Symbol=:both)
  if parameter == :both
    mle(p, mainargs...)
  elseif parameter == :drift
    mle_drift(p, mainargs...)
  elseif parameter == :diffusion
    mle_diffusion(p, mainargs...)
  else
    error("parameter keyword arg can be :both or :drift or :diffusion.")
  end
end
