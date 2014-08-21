module RDE

using Brownian
using Distributions

import Base: convert, rand!, rand
import Distributions: logpdf

export
  OrnsteinUhlenbeck,
  OU,
  rand!,
  rand,
  ito!,
  ito,
  invito!,
  invito,
  logpdf,
  approx_mle_ou_drift

include(joinpath("univariate", "OrnsteinUhlenbeck.jl"))

end
