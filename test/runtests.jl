using AccelInterfaces
using Test

import Profile

if Sys.islinux()
    const fort_compile = "ftn -fPIC -shared -g"
    const acc_compile  = "ftn -shared -fPIC -h acc,noomp"
    const omp_compile  = "ftn -shared -fPIC -h omp,noacc"
    const cpp_compile  = "ftn -fPIC -shared -g"

elseif Sys.isapple()
    const fort_compile = "gfortran -fPIC -shared -g"
    const cpp_compile  = "g++ -fPIC -shared -g"

else
    error("Current OS is not supported yet.")

end

const TEST1 = 100
const TEST2 = (1, 2)

const SHAPE = (2,3,4)
const X = rand(Float64, SHAPE)
const Y = rand(Float64, SHAPE)

const ANS = X .+ Y

function fortran_test_string()

    kernel_text = """
[fortran]
INTEGER i, j, k

DO k=LBOUND(X, 3), UBOUND(X, 3)
    DO j=LBOUND(X, 2), UBOUND(X, 2)
        DO i=LBOUND(X, 1), UBOUND(X, 1)
            Z(i, j, k) = X(i, j, k) + Y(i, j, k)
        END DO
    END DO
END DO
"""

    Z = fill(0.::Float64, SHAPE)

    @jaccel myaccel framework(fortran) compile(fort_compile)

    @jkernel mykernel myaccel kernel_text

    @jlaunch(mykernel, X, Y; output=(Z,))

    @test Z == ANS

end

function fortran_test_file()

    Z = fill(0.::Float64, SHAPE)

    @jaccel myaccel framework(fortran) constant(TEST1, TEST2
            ) compile(fort_compile) set(debugdir=".jaitmp")

    @jkernel mykernel myaccel "ex1.knl"

    @jlaunch(mykernel, X, Y; output=(Z,))

    @test Z == ANS


end

function fortran_openacc_tests()

    Z = fill(0.::Float64, SHAPE)

    ismaster = true

    @jaccel myaccel framework(fortran_openacc) constant(TEST1, TEST2
                    ) compile(acc_compile) set(master=ismaster,
                    debugdir=".jaitmp")


    @jkernel mykernel myaccel "ex1.knl"

    @jenterdata myaccel allocate(X, Y, Z) update(X, Y)

    @jlaunch(mykernel, X, Y; output=(Z,))

    @jexitdata myaccel update(Z) deallocate(X, Y, Z)

    @jdecel myaccel

    @test Z == ANS

end

function fortran_omptarget_tests()

    Z = fill(0.::Float64, SHAPE)

    ismaster = true

    @jaccel myaccel framework(fortran_omptarget) constant(TEST1, TEST2
                    ) compile(omp_compile) set(master=ismaster,
                    debugdir=".jaitmp")


    @jkernel mykernel myaccel "ex1.knl"

    @jenterdata myaccel allocate(X, Y, Z) update(X, Y)

    @jlaunch(mykernel, X, Y; output=(Z,))

    @jexitdata myaccel update(Z) deallocate(X, Y, Z)

    @jdecel myaccel

    @test Z == ANS

end

function cpp_test_string()

    kernel_text = """
[cpp]
int i;

for(int i=0; i < LENGTH(X); i++) {
    Z[i] = X[i] + Y[i]
}
"""

    Z = fill(0.::Float64, SHAPE)

    @jaccel myaccel framework(cpp) compile(cpp_compile)

    @jkernel mykernel myaccel kernel_text

    @jlaunch(mykernel, X, Y; output=(Z,))

    @test Z == ANS

end

@testset "AccelInterfaces.jl" begin

    if Sys.islinux()
        fortran_test_string()
        fortran_test_file()
        fortran_openacc_tests()
        fortran_omptarget_tests()
        cpp_test_string()

    elseif Sys.isapple()
        fortran_test_string()
        fortran_test_file()
        cpp_test_string()

    else
        error("Current OS is not supported yet.")

    end

end

