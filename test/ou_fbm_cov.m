#!/opt/bin/mathematica/10.0.1/MathematicaScript -script

(* Mathematica script for testing the covariance of stationary fOU *)

Print["Integral appearing in the covariance of stationary fOU:"]
x = Integrate[Cos[s*u]*(u^(1 - 2*h)/(q^2 + u^2)), {u, 0, Infinity},
  Assumptions -> Element[s | h | q, Reals] && 0 < h < 1]
Print[x, "\n"]

Print["Coefficient of the hypergeometric series appearing in the covariance of stationary fOU:"]
x = FullSimplify[(Gamma[k + 1]*Gamma[h + 1/2]*Gamma[h+1])/(Gamma[k+h+1/2]*Gamma[k+h+1]), 
  Assumptions -> Element[k, Integers] && k > 0]
Print[x, "\n"]

Print["First for terms of the hypergeometric series appearing in the covariance of stationary fOU:"]
x =Series[HypergeometricPFQ[{1}, {1/2 + h, 1 + h}, z], {z, 0, 4}]
Print[x, "\n"]

OUFBMCovHSeries[h_, b_, t_] := HypergeometricPFQ[{1}, {1/2+h, 1+h}, (b^2 t^2)/4]
x = OUFBMCovHSeries[0.75, 3, 5]
Print["Hypergeometric series in the ovariance of stationary fOU with drift λ=3 and Hurst index h=0.75 for t=5.: ", x,
  "\n"]

OUFBMCov[h_, λ_, σ_, t_] :=
  0.5*σ^2*(Gamma[2*h+1]*Cosh[λ*t]/(λ^(2*h))-t^(2*h)*HypergeometricPFQ[{1}, {1/2+h, 1+h}, (λ^2 t^2)/4])
x = OUFBMCov[0.75, 3, 0.1, 5]
Print["Covariance of stationary fOU with drift λ=3, diffusion constant σ=0.1 and Hurst index h=0.75 for t=5.: ", x]
