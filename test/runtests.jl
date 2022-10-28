using AccelInterfaces
#using AccelInterfaces: @jaccel, @jkernel, @jenterdata, @jlaunch, @jwait, @jexitdata, @jdecel

using Test

#import Profile

if occursin("crusher", Sys.BINDIR)
    const fort_compile = "ftn -fPIC -shared -g"
    const acc_compile  = "ftn -shared -fPIC -h acc,noomp"
    const omp_compile  = "ftn -shared -fPIC -h omp,noacc"
    #const acc_compile  = "ftn -shared -fPIC -fopenacc"
    #const omp_compile  = "ftn -shared -fPIC -fopenmp"

    const cpp_compile  = "CC -fPIC -shared -g"
    const workdir = "/gpfs/alpine/cli133/proj-shared/grnydawn/temp/jaiwork"

#elseif Sys.isapple()
else
    const fort_compile = "gfortran -fPIC -shared -g"
    const cpp_compile  = "g++ -fPIC -shared -g"
    const workdir = ".jaitmp"

end

const TEST1 = 100
const TEST2 = (1, 2)

const SHAPE = (2,3,4)
const X = rand(Float64, SHAPE)
const Y = rand(Float64, SHAPE)

const ANS = X .+ Y

function fortran_test_string()


    kernel_text = """

test = 1

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

    @jaccel framework(fortran=fort_compile) set(debugdir=workdir, workdir=workdir)

    @jenterdata allocate(X, Y, Z)

    @jkernel mykernel kernel_text

    @jlaunch(mykernel, X, Y; output=(Z,))

    @jexitdata deallocate(X, Y, Z)

    @jdecel

    @test Z == ANS

end

function fortran_test_file()

    Z = fill(0.::Float64, SHAPE)

    @jaccel framework(fortran=(compile=fort_compile,)) constant(TEST1, TEST2
            ) set(debugdir=workdir, workdir=workdir)

    @jkernel mykernel "ex1.knl"

    @jlaunch(mykernel, X, Y; output=(Z,))

    @test Z == ANS


end

function fortran_openacc_tests()

    Z = fill(0.::Float64, SHAPE)

    ismaster = true

    @jaccel framework(fortran_openacc=acc_compile) constant(TEST1, TEST2
                    ) compile(acc_compile) device(1) set(debugdir=workdir, master=ismaster,
                    workdir=workdir)


    @jkernel mykernel "ex1.knl"

    @jenterdata allocate(X, Y, Z) updateto(X, Y)

    @jlaunch(mykernel, X, Y; output=(Z,))

    @jexitdata updatefrom(Z) deallocate(X, Y, Z) async
    @test Z == ANS
    #println("TTTTT", Z)

    @jwait

    @jdecel


end

function fortran_omptarget_tests()

    Z = fill(0.::Float64, SHAPE)

    ismaster = true

    @jaccel myaccel framework(fortran_omptarget=omp_compile) constant(TEST1, TEST2
                    ) device(1) set(master=ismaster,
                    debugdir=workdir, workdir=workdir)


    @jkernel mykernel "ex1.knl" myaccel

    @jenterdata myaccel allocate(X, Y, Z) updateto(X, Y)

    @jlaunch(mykernel, X, Y; output=(Z,))

    @jexitdata myaccel updatefrom(Z) deallocate(X, Y, Z)

    @jdecel myaccel

    @test Z == ANS

end

function cpp_test_string()

    kernel_text = """

[cpp]
//for(int k=0; k<JSHAPE(X, 0); k++) {
    int k = 0;
    for(int j=0; j<JSHAPE(X, 1); j++) {
        for(int i=0; i<JSHAPE(X, 2); i++) {
            Z[k][j][i] = X[k][j][i] + Y[k][j][i];
        }
    }
//}
"""

    Z = fill(0.::Float64, SHAPE)

    ENV["CXX"] = "g++"
    ENV["CXXFLAGS"] = "-fPIC -shared -g"

    @jaccel myaccel framework(cpp) set(debugdir=workdir, workdir=workdir)

    @jkernel mykernel kernel_text myaccel

    #Profile.@profile @jlaunch(mykernel, X, Y; output=(Z,))
    #@time for i in range(1, stop=10)
        @jlaunch(mykernel, X, Y; output=(Z,))
    #end

    @test Z[:,:,1] == ANS[:,:,1]

    #open(".jaitmp/profile.txt", "w") do s
    #    Profile.print(s, format=:flat, sortedby=:count)
    #end

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

