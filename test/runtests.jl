using AccelInterfaces
using Test

function basic_tests()

    constvars = (100,)
    constnames = ("TEST",)

    accel = AccelInfo(JAI_FORTRAN, constnames=constnames, constvars=constvars)
    @test accel isa AccelInfo

    kernel = KernelInfo(accel, "ex1.knl")
    @test kernel isa KernelInfo

    x = [1,2,3]
    y = [2,3,4]
    z = [0,0,0]

    innames = ("x", "y", "z")
    outnames = ("z",)
    compile = "gfortran -fPIC -shared -g"

    launch!(kernel, x, y, z, outvars=(z,), innames=innames,
            outnames=outnames, compile=compile)
    @test z == [3,5,7]

end

@testset "AccelInterfaces.jl" begin

    # testing AccelInterfaces module loading
    @test JAI_FORTRAN isa AccelType

    basic_tests()

end

