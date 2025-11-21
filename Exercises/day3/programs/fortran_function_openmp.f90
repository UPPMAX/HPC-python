function integration2d_fortran_openmp(n) result(integral)
    !$ use omp_lib
    implicit none
    integer, parameter :: dp=selected_real_kind(15,9)
    real(kind=dp), parameter   :: pi=3.14159265358979323
    integer, intent(in)        :: n
    real(kind=dp)              :: integral

    integer                    :: i,j
!   interval size
    real(kind=dp)              :: h
!   x and y variables
    real(kind=dp)              :: x,y
!   cummulative variable
    real(kind=dp)              :: mysum

    h = pi/(1.0_dp * n)
    mysum = 0.0_dp
!   regular integration in the X axis
!$omp parallel do reduction(+:mysum) private(x,y,j)
    do i = 0, n-1
       x = h * (i + 0.5_dp)
!      regular integration in the Y axis
       do j = 0, n-1
           y = h * (j + 0.5_dp)
           mysum = mysum + sin(x + y)
       enddo
    enddo
!$omp end parallel do

    integral = h*h*mysum

end function integration2d_fortran_openmp

