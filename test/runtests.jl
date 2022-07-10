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

#function fortran_tests()
#
#    z = fill(0, N)
#
#    accel = AccelInfo(JAI_FORTRAN, const_names=constnames, const_vars=constvars, compile=fort_compile)
#    @test accel isa AccelInfo
#
#    kernel = KernelInfo(accel, "ex1.knl")
#    #knl = @jkernel(accel, "ex1.knl")
#
#    @test kernel isa KernelInfo
#
#    @jenterdata accel allocate(x, y, z)
#
#    @jlaunch(kernel, x, y; output=(z,))
#
#    @test z == res
#
#    @jexitdata accel deallocate(x, y, z)
#
#end
#
function fortran_openacc_tests()

    z = fill(0, N)

    ismaster = true

    @jaccel myaccel framework(fortran_openacc) constant(TEST1, TEST2) compile(acc_compile) set(master=ismaster)


    @jkernel mykernel myaccel "ex1.knl"

    @jenterdata myaccel allocate(x, y, z) update(x, y)

    @jlaunch(mykernel, x, y; output=(z,))

    @jexitdata myaccel update(z) deallocate(x, y, z)

    @jdecel myaccel

    @test z == res

end

@testset "AccelInterfaces.jl" begin

    # testing AccelInterfaces module loading
    @test JAI_FORTRAN isa AccelType

    #fortran_tests()
    fortran_openacc_tests()

end

