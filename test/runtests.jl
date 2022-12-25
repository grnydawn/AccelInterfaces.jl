using AccelInterfaces

using Test

if occursin("crusher", Sys.BINDIR)
    const SYSNAME = "Crusher"

elseif occursin("summit", Sys.BINDIR)
    const SYSNAME = "Summit"

elseif Sys.isapple()
    const SYSNAME = "MacOS"

elseif Sys.islinux()
    const SYSNAME = "Linux"

else
    error("Not supported system.")
end

if SYSNAME == "Crusher" 
    const fort_compile = "ftn -fPIC -shared -g"
    const acc_compile  = "ftn -shared -fPIC -h acc,noomp"
    const omp_compile  = "ftn -shared -fPIC -h omp,noacc"
    #const acc_compile  = "ftn -shared -fPIC -fopenacc"
    #const omp_compile  = "ftn -shared -fPIC -fopenmp"

    const cpp_compile  = "CC -fPIC -shared -g"
    const hip_compile  = "hipcc -shared -fPIC -lamdhip64 -g"
    const workdir = "/gpfs/alpine/cli133/proj-shared/grnydawn/temp/jaiwork"

elseif SYSNAME == "Summit" 
    const fort_compile = "pgfortran -fPIC -shared -g"
    const acc_compile  = "pgfortran -shared -fPIC -acc -ta=tesla:nordc"

    const cpp_compile  = "pgc++ -fPIC -shared -g"
    const cuda_compile  = "nvcc --linker-options=\"-fPIC\" --shared -g"

    #const workdir = "/gpfs/alpine/scratch/grnydawn/cli137/jaiwork"
    const workdir = "/ccs/home/grnydawn/temp/jaiwork"

else
    const fort_compile = "gfortran -fPIC -shared -g"
    const cpp_compile  = "g++ -fPIC -shared -g"
    const workdir = ".jaitmp"

end

const TEST1 = 100
const TEST2 = (1, 2)
const SHAPE = (2,3,4)

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

    X = rand(Float64, SHAPE)
    Y = rand(Float64, SHAPE)
    Z = fill(0.::Float64, SHAPE)
    ANS = X .+ Y

    @jaccel fortacc framework(fortran=fort_compile) set(debugdir=workdir, workdir=workdir)

    @jenterdata fortacc allocate(X, Y, Z)

    @jkernel fortacc mykernel kernel_text

    @jlaunch fortacc mykernel input(X, Y) output(Z,)

    @jexitdata fortacc deallocate(X, Y, Z)

    @jwait fortacc

    @jdecel fortacc

    @test Z == ANS

end

function fortran_test_file()

    X = rand(Float64, SHAPE)
    Y = rand(Float64, SHAPE)
    Z = fill(0.::Float64, SHAPE)
    ANS = X .+ Y


    @jaccel fortfileacc framework(fortran=(compile=fort_compile,)) constant(TEST1, TEST2
            ) set(debugdir=workdir, workdir=workdir)

    @jkernel fortfileacc mykernel "ex1.knl"

    @jlaunch fortfileacc mykernel input(X, Y) output(Z,)

    @test Z == ANS

end

function fortran_openacc_tests()

    X = rand(Float64, SHAPE)
    Y = rand(Float64, SHAPE)
    Z = fill(0.::Float64, SHAPE)
    ANS = X .+ Y

    ismaster = true

    @jaccel accacc framework(fortran_openacc=acc_compile) constant(TEST1, TEST2
                    ) compile(acc_compile) device(1) set(debugdir=workdir, master=ismaster,
                    workdir=workdir)


    @jkernel accacc mykernel "ex1.knl"

    @jenterdata accacc allocate(X, Y, Z) updateto(X, Y)

    @jlaunch accacc mykernel input(X, Y) output(Z,)

    @jexitdata accacc updatefrom(Z) deallocate(X, Y, Z) async

    @jwait accacc

    @test Z == ANS

    @jdecel accacc


end

function fortran_omptarget_tests()

    X = rand(Float64, SHAPE)
    Y = rand(Float64, SHAPE)
    Z = fill(0.::Float64, SHAPE)
    ANS = X .+ Y

    ismaster = true

    @jaccel omptacc framework(fortran_omptarget=omp_compile) constant(TEST1, TEST2
                    ) device(1) set(master=ismaster,
                    debugdir=workdir, workdir=workdir)


    @jkernel omptacc mykernel "ex1.knl"

    @jenterdata omptacc allocate(X, Y, Z) updateto(X, Y)

    @jlaunch omptacc mykernel input(X, Y) output(Z,)

    @jexitdata omptacc updatefrom(Z) deallocate(X, Y, Z)

    @jdecel omptacc

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
    X = rand(Float64, SHAPE)
    Y = rand(Float64, SHAPE)
    Z = fill(0.::Float64, SHAPE)
    ANS = X .+ Y

    ENV["CXX"] = "g++"
    ENV["CXXFLAGS"] = "-fPIC -shared -g"

    @jaccel cppacc framework(cpp) set(debugdir=workdir, workdir=workdir)

    @jkernel cppacc mykernel kernel_text

    #Profile.@profile @jlaunch(mykernel, X, Y; output=(Z,))
    #@time for i in range(1, stop=10)
        @jlaunch cppacc mykernel input(X, Y) output(Z,)
    #end

    @test Z[:,:,1] == ANS[:,:,1]

    #open(".jaitmp/profile.txt", "w") do s
    #    Profile.print(s, format=:flat, sortedby=:count)
    #end

    @jdecel cppacc
end

function cuda_test_string()

    kernel_text = """

[cuda]
for(int k=0; k<JSHAPE(X, 0); k++) {
    for(int j=0; j<JSHAPE(X, 1); j++) {
        for(int i=0; i<JSHAPE(X, 2); i++) {
            Z[k][j][i] = X[k][j][i] + Y[k][j][i];
        }
    }
}
"""

    X = rand(Float64, SHAPE)
    Y = rand(Float64, SHAPE)
    Z = fill(0.::Float64, SHAPE)
    ANS = X .+ Y

    @jaccel cudaacc framework(cuda=cuda_compile) set(debugdir=workdir, workdir=workdir)

    @jenterdata cudaacc allocate(X, Y, Z) updateto(X, Y) async

    @jkernel cudaacc mykernel kernel_text

    @jlaunch cudaacc mykernel input(X, Y) output(Z,) cuda("chevron"=>(1,1))

    @jexitdata cudaacc updatefrom(Z) deallocate(X, Y, Z) async

    @jwait cudaacc

    @test Z == ANS

    @jdecel cudaacc


end

function hip_test_string()
    kernel_text = """

[hip]
/*
for(int k=0; k<JSHAPE(X, 0); k++) {
    for(int j=0; j<JSHAPE(X, 1); j++) {
        for(int i=0; i<JSHAPE(X, 2); i++) {
            Z[k][j][i] = X[k][j][i] + Y[k][j][i];
        }
    }
}
*/
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;

    Z[i][j][k] = X[i][j][k] + Y[i][j][k];
"""

    X = rand(Float64, SHAPE)
    Y = rand(Float64, SHAPE)
    Z = fill(0.::Float64, SHAPE)
    ANS = X .+ Y

    @jaccel hipacc framework(hip=hip_compile) set(debugdir=workdir, workdir=workdir)

    @jenterdata hipacc allocate(X, Y, Z) updateto(X, Y)

    @jkernel hipacc mykernel kernel_text

    tt = ((4,3,2),1)
    #@jlaunch hipacc mykernel input(X, Y) output(Z,) hip("chevron"=>((4,3,2),1), "test"=>3)
    @jlaunch hipacc mykernel input(X, Y) output(Z,) hip("chevron"=>tt, "test"=>3)

    @jexitdata hipacc updatefrom(Z,) deallocate(X, Y, Z) async

    @jwait hipacc

    @test Z == ANS

    @jdecel hipacc

end

function fortran_openacc_hip_test_string()
    kernel_text = """

[hip]

/*
for(int k=0; k<JSHAPE(X, 0); k++) {
    for(int j=0; j<JSHAPE(X, 1); j++) {
        for(int i=0; i<JSHAPE(X, 2); i++) {
            Z[k][j][i] = X[k][j][i] + Y[k][j][i];
        }
    }
}
*/
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;

    Z1[i][j][k] = X1[i][j][k] + Y1[i][j][k];

"""

    X1 = rand(Float64, SHAPE)
    Y1 = rand(Float64, SHAPE)
    Z1 = fill(0.::Float64, SHAPE)
    ANS = X1 .+ Y1

    @jaccel acchipacc framework(fortran_openacc=acc_compile) set(debugdir=workdir, workdir=workdir)

    @jenterdata acchipacc allocate(X1, Y1, Z1) updateto(X1, Y1)

    @jkernel acchipacc mykernel kernel_text framework(hip=hip_compile)

    tt = ((4,3,2),1)
    @jlaunch acchipacc mykernel input(X1, Y1) output(Z1) hip("chevron"=>tt)

    @jexitdata acchipacc updatefrom(Z1) deallocate(X1, Y1, Z1) async

    @jwait acchipacc

    @test Z1 == ANS

    @jdecel acchipacc

end

function fortran_openacc_cuda_test_string()
    kernel_text = """

[cuda]

for(int k=0; k<JSHAPE(X, 0); k++) {
    for(int j=0; j<JSHAPE(X, 1); j++) {
        for(int i=0; i<JSHAPE(X, 2); i++) {
            Z[k][j][i] = X[k][j][i] + Y[k][j][i];
        }
    }
}
"""

    X = rand(Float64, SHAPE)
    Y = rand(Float64, SHAPE)
    Z = fill(0.::Float64, SHAPE)
    ANS = X .+ Y

    @jaccel acccudaacc framework(fortran_openacc=acc_compile) set(debugdir=workdir, workdir=workdir)

    @jenterdata acccudaacc allocate(X, Y, Z) updateto(X, Y)

    @jkernel acccudaacc mykernel kernel_text framework(cuda=cuda_compile)

    @jlaunch acccudaacc mykernel input(X, Y) output(Z) cuda("chevron"=>(1,1))

    @jexitdata acccudaacc updatefrom(Z) deallocate(X, Y, Z) async

    @jwait acccudaacc

    @test Z == ANS

    @jdecel acccudaacc

end

@testset "AccelInterfaces.jl" begin

    if SYSNAME == "Crusher"
        fortran_test_string()
        fortran_test_file()
        fortran_openacc_tests()
        fortran_omptarget_tests()
        cpp_test_string()
        hip_test_string()
        fortran_openacc_hip_test_string()

    elseif SYSNAME == "Summit"
        #fortran_test_string()
        #fortran_test_file()
        #fortran_openacc_tests()
        #cpp_test_string()
        #cuda_test_string()
        fortran_openacc_cuda_test_string()

    elseif SYSNAME == "Linux"
        fortran_test_string()
        fortran_test_file()
        fortran_openacc_tests()
        cpp_test_string()

    elseif SYSNAME == "MacOS"
        fortran_test_string()
        fortran_test_file()
        cpp_test_string()

    else
        error("Current OS is not supported yet.")

    end
end

