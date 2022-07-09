using AccelInterfaces
using Test

import Profile

const fort_compile = "ftn -fPIC -shared -g"
const acc_compile = "ftn -shared -fPIC -h acc,noomp"
const constvars = (100,(1,2))
const constnames = ("TEST", "TEST2")
const TEST1 = 100
const TEST2 = (1, 2)

const N = 3
const x = fill(1, N)
const y = fill(2, N)

const res = fill(3, N)

innames = ("x", "y")
outnames = ("z",)

function fortran_tests()

    z = fill(0, N)

    accel = AccelInfo(JAI_FORTRAN, constnames=constnames, constvars=constvars, compile=fort_compile)
    @test accel isa AccelInfo

    kernel = KernelInfo(accel, "ex1.knl")
    #knl = @jkernel(accel, "ex1.knl")

    @test kernel isa KernelInfo

    @jenterdata accel allocate(x, y, z)

    @jlaunch(kernel, x, y; output=(z,))

    @test z == res

    @jexitdata accel deallocate(x, y, z)

end

function fortran_openacc_tests()

    z = fill(0, N)


    accel = AccelInfo(JAI_FORTRAN_OPENACC, constnames=constnames,
                    constvars=constvars, compile=acc_compile)

    #accel = jaccel(JAI_FORTRAN_OPENACC, const=(TEST1, TEST2), compile=acc_compile)


    kernel = KernelInfo(accel, "ex1.knl")

    @jenterdata accel allocate(x, y, z) update(x, y)

    @jlaunch(kernel, x, y; output=(z,))

    @jexitdata accel update(z) deallocate(x, y, z)

    @test z == res

end

@testset "AccelInterfaces.jl" begin

    # testing AccelInterfaces module loading
    @test JAI_FORTRAN isa AccelType

    #fortran_tests()
    fortran_openacc_tests()

end

