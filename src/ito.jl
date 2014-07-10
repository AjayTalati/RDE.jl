UniOUFBMIto[y_, dx_, n_, l_, s_] := Module[{ld, expmld},
  ld = l/n;
  expmld = Exp[-ld];
  y*expmld+s*dx*(1-expmld)/ld
]

### Ito map from rough path increment dx to 
function ito(y::Float64, dx::Float64, p::OrnsteinUhlenbeck)
  pnmone::Int64 = p.x.n-1
  n::Int64 = 2^floor(log2(pnmone))

  ld = l/n
  expmld = exp(-ld)
  y*expmld+(p.σ)*dx*(1-expmld)/ld  
end
