function integration2d_julia(n::Int)
# interval size
  h = π/n
# cummulative variable
  mysum = 0.0
# regular integration in the X axis
  for i in 0:n-1
    x = h*(i+0.5)
#   regular integration in the Y axis
    for j in 0:n-1
       y = h*(j + 0.5)
       mysum = mysum + sin(x+y)
    end
  end
  return mysum
end
