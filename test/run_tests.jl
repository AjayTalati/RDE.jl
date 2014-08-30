tests = ["ou_bm"]

for t in tests
  tfile = "$t.jl"
  println("*** $tfile")
  include(tfile)
end
