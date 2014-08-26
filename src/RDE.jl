module RDE

using Brownian
using Distributions

import Base: convert, rand!, rand
import Distributions: loglikelihood

export
  OrnsteinUhlenbeck,
  OU,
  rand!,
  rand,
  ito!,
  ito,
  invito!,
  invito,
  loglikelihood,
  approx_mle_ou_drift,
  approx_mle_ou_diffusion,
  approx_mle_ou,
  exact_mle_ou_drift,
  exact_mle_ou_diffusion,
  exact_mle_ou

include(joinpath("univariate", "OrnsteinUhlenbeck.jl"))

end
