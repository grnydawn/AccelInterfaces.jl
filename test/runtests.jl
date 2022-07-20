using AccelInterfaces
using Test

import Profile

const fort_compile = "ftn -fPIC -shared -g"
#const fort_compile = "gfortran -fPIC -shared -g"
const acc_compile = "ftn -shared -fPIC -h acc,noomp"
const omp_compile = "ftn -shared -fPIC -h omp,noacc"

const TEST1 = 100
const TEST2 = (1, 2)

const N = 3
const x = fill(1, N)
const y = fill(2, N)

const res = fill(3, N)

function fortran_test_string()

    kernel_text = """
[fortran]
INTEGER i

DO i=LBOUND(x, 1), UBOUND(x, 1)
    z(i) = x(i) + y(i)
END DO
"""

    z = fill(0, N)

    @jaccel myaccel framework(fortran) compile("gfortran -fPIC -shared")

    @jkernel mykernel myaccel kernel_text

    @jlaunch(mykernel, x, y; output=(z,))

    @test z == res

end

function fortran_test_file()

    z = fill(0, N)

    @jaccel myaccel framework(fortran) constant(TEST1, TEST2
            ) compile(fort_compile) set(debugdir=".jaitmp")

    @jkernel mykernel myaccel "ex1.knl"

    @jlaunch(mykernel, x, y; output=(z,))

    @test z == res


end

function fortran_openacc_tests()

    z = fill(0, N)

    ismaster = true

    @jaccel myaccel framework(fortran_openacc) constant(TEST1, TEST2
                    ) compile(acc_compile) set(master=ismaster,
                    debugdir=".jaitmp")


    @jkernel mykernel myaccel "ex1.knl"

    @jenterdata myaccel allocate(x, y, z) update(x, y)

    @jlaunch(mykernel, x, y; output=(z,))

    @jexitdata myaccel update(z) deallocate(x, y, z)

    @jdecel myaccel

    @test z == res

end

function fortran_omptarget_tests()

    z = fill(0, N)

    ismaster = true

    @jaccel myaccel framework(fortran_omptarget) constant(TEST1, TEST2
                    ) compile(omp_compile) set(master=ismaster,
                    debugdir=".jaitmp")


    @jkernel mykernel myaccel "ex1.knl"

    @jenterdata myaccel allocate(x, y, z) update(x, y)

    @jlaunch(mykernel, x, y; output=(z,))

    @jexitdata myaccel update(z) deallocate(x, y, z)

    @jdecel myaccel

    @test z == res

end

@testset "AccelInterfaces.jl" begin

    fortran_test_string()
    fortran_test_file()
    fortran_openacc_tests()
    fortran_omptarget_tests()

end

