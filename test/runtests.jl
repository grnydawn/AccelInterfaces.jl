using AccelInterfaces

using Test

if occursin("crusher", Sys.BINDIR)
    const SYSNAME = "Crusher"

elseif get!(ENV, "NERSC_HOST", "") == "perlmutter"
    const SYSNAME = "Perlmutter"

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

    const workdir = "/gpfs/alpine/scratch/grnydawn/cli137/jaiwork"
    #const workdir = "/ccs/home/grnydawn/temp/jaiwork"

    if get!(ENV, "LMOD_FAMILY_COMPILER", "") == "pgi"

        const fort_compile = "pgfortran -fPIC -shared -g"
        const acc_compile  = "pgfortran -shared -fPIC -acc -ta=tesla:nordc"

        const cpp_compile  = "pgc++ -fPIC -shared -g"
        const cuda_compile  = "nvcc --linker-options=\"-fPIC\" --compiler-options=\"-fPIC\" --shared -g"

    elseif get!(ENV, "LMOD_FAMILY_COMPILER", "") == "xl"

        const fort_compile = "xlf_r -qpic -qmkshrobj -g"
        const omp_compile  = "xlf_r -qpic -qmkshrobj -qoffload -qsmp -g"

        const cpp_compile  = "xlc++_r -qpic -qmkshrobj -g"
        const cuda_compile  = "nvcc --linker-options=\"-fPIC\"  --compiler-options=\"-fPIC\" --shared -g"

    end

elseif SYSNAME == "Perlmutter" 

    const fort_compile = "ftn -fPIC -shared -g"
    const acc_compile  = "ftn -shared -fPIC -h acc,noomp"
    #const omp_compile  = "ftn -shared -fPIC -h omp,noacc"
    const omp_compile  = "ftn -shared -fPIC -fopenmp"
    #const acc_compile  = "ftn -shared -fPIC -fopenacc"
    #const omp_compile  = "ftn -shared -fPIC -fopenmp"

    const cpp_compile  = "CC -fPIC -shared -g"
    const cuda_compile  = "nvcc --linker-options=\"-fPIC\" --compiler-options=\"-fPIC\" --shared -g"
    const workdir = "/pscratch/sd/y/youngsun/jaiwork"

else
    const fort_compile = "gfortran -fPIC -shared -g"
    const cpp_compile  = "g++ -fPIC -shared -g"
    const workdir = ".jworkdir"

end

const TEST1 = 100
const TEST2 = (1, 2)
const SHAPE = (2,3,4)
#const SHAPE = (200,30,40)

@jconfig constant(TEST1, TEST2) set(workdir=workdir)

function fortran_test_string()


    kernel_text = """
[kernel: x, y=1, z=":e:"]
X = 1
[fortran: t=X, y=3]
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

    fcompile = Dict("compile" => fort_compile)

    @jaccel fortacc framework(fortran=fcompile) set(debug=true)

    @jenterdata fortacc alloc(X, Y, Z)

    @jkernel kernel_text mykernel fortacc framework(fortran=fort_compile)

    @jlaunch mykernel fortacc input(X, Y) output(Z,) fortran(test="1", tt="2")

    @jexitdata fortacc delete(X, Y, Z)

    @jwait fortacc

    @jdecel fortacc

    @test Z == ANS

end

function fortran_test_file()

    X = rand(Float64, SHAPE)
    Y = rand(Float64, SHAPE)
    Z = fill(0.::Float64, SHAPE)
    ANS = X .+ Y

    gfortran = Dict(
                    :path => "gfortran",
                    :opt_shared=>"-fPIC -shared",
                    :opt_frameworks=>((:fortran, ""),)
                    )

    @jaccel compiler(gfortran=gfortran)

    @jkernel "ex1.knl"

    @jlaunch input(X, Y) output(Z)

    @jdecel

    @test Z == ANS

end

function fortran_openacc_tests()

    X = rand(Float64, SHAPE)
    Y = rand(Float64, SHAPE)
    Z = fill(0.::Float64, SHAPE)
    ANS = X .+ Y

    ismaster = true

    #@jconfig test compiler(gnu="testeg")

    @jaccel accacc framework(fortran_openacc=acc_compile) constant(TEST1, TEST2
                    ) compiler(acc_compile) device(1) set(debugdir=workdir, master=ismaster,
                    workdir=workdir)


    @jkernel "ex1.knl" mykernel accacc

    @jenterdata accacc alloc(X, Y, Z) updateto(X, Y)

    @jlaunch mykernel accacc input(X, Y) output(Z,)

    @jexitdata accacc updatefrom(Z) delete(X, Y, Z) async

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


    @jkernel "ex1.knl" mykernel omptacc 

    @jenterdata omptacc alloc(X, Y, Z) updateto(X, Y)

    @jlaunch mykernel omptacc input(X, Y) output(Z,)

    @jexitdata omptacc updatefrom(Z) delete(X, Y, Z)

    @jdecel omptacc

    @test Z == ANS

end

function fortran_omptarget_cuda_tests()

    X = rand(Float64, SHAPE)
    Y = rand(Float64, SHAPE)
    Z = fill(0.::Float64, SHAPE)
    ANS = X .+ Y

    ismaster = true

    @jaccel ompcuda framework(fortran_omptarget=omp_compile, cuda=cuda_compile) constant(TEST1, TEST2
                    ) device(1, 2) set(master=ismaster,
                    debugdir=workdir, workdir=workdir)


    @jkernel  "ex1.knl" mykernel ompcuda

    @jenterdata ompcuda alloc(X, Y, Z) updateto(X, Y)

    @jlaunch mykernel ompcuda input(X, Y) output(Z,) cuda(chevron=(SHAPE,1))

    @jexitdata ompcuda updatefrom(Z) delete(X, Y, Z)

    @jdecel ompcuda

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

    @jkernel kernel_text mykernel cppacc

    #Profile.@profile @jlaunch(mykernel, X, Y; output=(Z,))
    #@time for i in range(1, stop=10)
        @jlaunch mykernel cppacc input(X, Y) output(Z,)
    #end

    @test Z[:,:,1] == ANS[:,:,1]

    #open(".jworkdir/profile.txt", "w") do s
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

    @jenterdata cudaacc alloc(X, Y, Z) updateto(X, Y) async

    @jkernel kernel_text mykernel cudaacc

    @jlaunch mykernel cudaacc input(X, Y) output(Z,) cuda(chevron=(1,1))

    @jexitdata cudaacc updatefrom(Z) delete(X, Y, Z) async

    @jwait cudaacc

    @test Z == ANS

    @jdecel cudaacc


end

function hip_fortran_test_string()

    kernel_hiptext = """

[hip]
for(int k=0; k<JSHAPE(X, 0); k++) {
    for(int j=0; j<JSHAPE(X, 1); j++) {
        for(int i=0; i<JSHAPE(X, 2); i++) {
            Z[k][j][i] = X[k][j][i] + Y[k][j][i];
        }
    }
}
"""

    kernel_fortrand = """
[fortran]
CALL RANDOM_NUMBER(X)
CALL RANDOM_NUMBER(Y)
CALL RANDOM_NUMBER(Z)
"""

    X = Array{Float64}(undef, SHAPE)
    Y = Array{Float64}(undef, SHAPE)
    Z = Array{Float64}(undef, SHAPE)

    #println("#######")
    #println(Z)

    @jaccel hipacc framework(hip=hip_compile) set(debugdir=workdir, workdir=workdir)

    @jkernel kernel_fortrand initarrays hipacc framework(fortran=fort_compile)

    @jlaunch initarrays hipacc output(X, Y, Z) fortran(test=2)

    #println("#######")
    #println(Z)

    ANS = X .+ Y

    @jenterdata hipacc alloc(X, Y, Z) updateto(X, Y) async

    @jkernel kernel_hiptext mykernel hipacc

    @jlaunch mykernel hipacc input(X, Y) output(Z,) hip(chevron=(1,1))

    @jexitdata hipacc updatefrom(Z) delete(X, Y, Z) async

    @jwait hipacc

    #println("#######")
    #println(Z)

    @test Z == ANS

    @jdecel hipacc


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

    @jenterdata hipacc alloc(X, Y, Z) updateto(X, Y)

    @jkernel kernel_text mykernel hipacc

    tt = ((4,3,2),1)
    @jlaunch mykernel hipacc input(X, Y) output(Z,) hip(chevron=tt, test=3)

    @jexitdata hipacc updatefrom(Z,) delete(X, Y, Z) async

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

    @jenterdata acchipacc alloc(X1, Y1, Z1) updateto(X1, Y1)

    @jkernel kernel_text mykernel acchipacc framework(hip=hip_compile)

    tt = ((4,3,2),1)
    @jlaunch mykernel acchipacc input(X1, Y1) output(Z1) hip(chevron=tt)

    @jexitdata acchipacc updatefrom(Z1) delete(X1, Y1, Z1) async

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

    @jenterdata acccudaacc alloc(X, Y, Z) updateto(X, Y)

    @jkernel kernel_text mykernel acccudaacc framework(cuda=cuda_compile)

    @jlaunch mykernel acccudaacc input(X, Y) output(Z) cuda(chevron=(1,1))

    @jexitdata acccudaacc updatefrom(Z) delete(X, Y, Z) async

    @jwait acccudaacc

    @test Z == ANS

    @jdecel acccudaacc

end

@testset "AccelInterfaces.jl" begin

    if SYSNAME == "Crusher"
        fortran_test_string()
        fortran_test_file()
        #fortran_openacc_tests()
        #fortran_omptarget_tests()
        #cpp_test_string()
        #hip_test_string()
        #hip_fortran_test_string()
        #fortran_openacc_hip_test_string()

    elseif SYSNAME == "Perlmutter"
        #fortran_test_string()
        #fortran_test_file()
        #fortran_omptarget_tests()
        fortran_omptarget_cuda_tests()
        #cpp_test_string()
        #hip_test_string()
        #hip_fortran_test_string()
        #fortran_openacc_hip_test_string()

    elseif SYSNAME == "Summit"
        fortran_test_string()
        #fortran_test_file()
        #fortran_openacc_tests()
        #cpp_test_string()
        #cuda_test_string()
        #fortran_openacc_cuda_test_string()

    elseif SYSNAME == "Linux"
        fortran_test_string()
        fortran_test_file()
        fortran_openacc_tests()
        cpp_test_string()

    elseif SYSNAME == "MacOS"
        fortran_test_string()
        fortran_test_file()
        #cpp_test_string()

    else
        error("Current OS is not supported yet.")

    end
end

