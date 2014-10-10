tests = ["ou", "fou"]

for t in tests
  tfile = "$t.jl"
  println("*** $tfile")
  include(tfile)
end
