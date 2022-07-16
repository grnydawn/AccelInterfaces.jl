using AccelInterfaces

const kernel_text = """

[fortran]

INTEGER i

DO i=LBOUND(x, 1), UBOUND(x, 1)
z(i) = x(i) + y(i)
END DO

[fortran_openacc]

INTEGER i

!\$acc parallel loop present(x, y, z)
DO i=LBOUND(x, 1), UBOUND(x, 1)
z(i) = x(i) + y(i)
END DO
!\$acc end parallel loop

"""

const N = 10
const x = fill(1, N)
const y = fill(2, N)
const z = fill(0, N)
const answer = fill(3, N)

#@jaccel myaccel1 framework(fortran) compile("gfortran -fPIC -shared")
#
#@jkernel mykernel myaccel1 kernel_text
#
#@jlaunch(mykernel, x, y; output=(z,))
#
#@assert z == answer

fill!(z, 0)

@jaccel myaccel2 framework(fortran_openacc) compile("gfortran -fopenacc -fPIC -shared")

@jkernel mykernel2 myaccel2 kernel_text

@jenterdata myaccel2 allocate(x, y, z) update(x, y)

@jlaunch(mykernel2, x, y; output=(z,))

@jexitdata myaccel2 update(z) deallocate(x, y, z)

@jdecel myaccel2

@assert z == answer
