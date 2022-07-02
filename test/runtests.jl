using AccelInterfaces
using Test

function launch_test()
    return true
end


@testset "AccelInterfaces.jl" begin

    @test FLANG isa AccelType

    accel = AccelInfo(FLANG)
    @test accel isa AccelInfo

    kernel = KernelInfo(accel, "ex1.knl")
    @test kernel isa KernelInfo

    x = [1,2,3]
    y = [2,3,4]
    z = [0,0,0]

    innames = ("x", "y")
    outnames = ("z",)
    compile = "gfortran -fPIC -shared -g"

    launch!(kernel, x, y, outvars=(z,), innames=innames,
            outnames=outnames, compile=compile)
    @test z == [3,5,7]
end

