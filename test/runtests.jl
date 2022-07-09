using AccelInterfaces
using Test

import Profile

const constvars = (100,(1,2))
const constnames = ("TEST", "TEST2")

const N = 3
const x = fill(1, N)
const y = fill(2, N)

const res = fill(3, N)

innames = ("x", "y")
outnames = ("z",)

function fortran_tests()

    z = fill(0, N)

    accel = AccelInfo(JAI_FORTRAN, constnames=constnames, constvars=constvars)
    @test accel isa AccelInfo

    kernel = KernelInfo(accel, "ex1.knl")
    @test kernel isa KernelInfo

    compile = "ftn -fPIC -shared -g"

    @enterdata accel allocate(x, y, z)

    @launch(kernel, x, y; outvars=(z,), compile=compile)

    @test z == res

    @exitdata accel deallocate(x, y, z)

end

function fortran_openacc_tests()

    z = fill(0, N)

    compile = "ftn -shared -fPIC -h acc,noomp"

    accel = AccelInfo(JAI_FORTRAN_OPENACC, constnames=constnames,
                    constvars=constvars, compile=compile)

    kernel = KernelInfo(accel, "ex1.knl")

# TODO:
# - @enterdata accel allocate(x, y, z) update(x, y)
# - @exitdata accel update(z) deallocate(x, y, z) 
# - any order of allocate, update, deallocate
# - check allowed directives at enter and exit

    @enterdata accel allocate(x, y, z) update(x, y)
    #@enterdata accel allocate(x, y, z, names=(innames..., outnames...)) update(x, y, names=innames)

    #@launch(kernel, x, y, outvars=(z,), innames=innames, outnames=outnames, compile=compile)
    @launch(kernel, x, y; outvars=(z,), compile=compile)

    #@exitdata accel update(z, names=outnames) deallocate(x, y, z, names=(innames..., outnames...))
    @exitdata accel update(z) deallocate(x, y, z)

    @test z == res

end

@testset "AccelInterfaces.jl" begin

    # testing AccelInterfaces module loading
    @test JAI_FORTRAN isa AccelType

    fortran_tests()
    fortran_openacc_tests()

end

