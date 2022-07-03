module AccelInterfaces

import Libdl.dlopen,
       Libdl.RTLD_LAZY,
       Libdl.RTLD_DEEPBIND,
       Libdl.RTLD_GLOBAL,
       Libdl.dlsym

import SHA.sha1

import OffsetArrays.OffsetArray,
       OffsetArrays.OffsetVector

export AccelType, JAI_VERSION, JAI_FORTRAN, JAI_CPP, JAI_FORTRAN_OPENACC,
        JAI_ANYACCEL, AccelInfo,
        KernelInfo, get_accel!,get_kernel!, allocate!, deallocate!, copyin!,
        copyout!, launch!

const TIMEOUT = 60
const JAI_VERSION = "0.0.1"

@enum AccelType JAI_FORTRAN JAI_CPP JAI_ANYACCEL JAI_FORTRAN_OPENACC JAI_HEADER
@enum BuildType JAI_LAUNCH JAI_ALLOCATE JAI_DEALLOCATE JAI_COPYIN JAI_COPYOUT

struct AccelInfo

    accelid::String
    acceltype::AccelType
    ismaster::Bool
    constvars::Tuple
    constnames::NTuple
    compile::Union{String, Nothing}
    sharedlibs::Dict
    workdir::Union{String, Nothing}

    function AccelInfo(acceltype::AccelType; ismaster::Bool=true,
                    constvars::Tuple=(), compile::Union{String, Nothing}=nothing,
                    constnames::NTuple=(), workdir::Union{String, Nothing}=nothing)

        # TODO: check if acceltype is supported in this system(h/w, compiler, ...)
        #     : detect available acceltypes according to h/w, compiler, flags, ...

        accelid = bytes2hex(sha1(string(Sys.STDLIB, JAI_VERSION,
                        acceltype, constvars, constnames, compile))[1:4])

        new(accelid, acceltype, ismaster, constvars, constnames, compile, 
            Dict(), workdir)
    end
end

include("./kernel.jl")
include("./fortran.jl")
include("./cpp.jl")

function timeout(duration)

    tstart = now()
    while true
        if isfile(libpath)
            break

        elseif now() - tstart > duration
            error("Timeout occured while waiting for shared library")

        else
            sleep(0.1)
        end
    end
end

function get_accel!(acceltype::AccelType; ismaster::Bool=true, constvars::Tuple=(),
                    compile::Union{String, Nothing}=nothing,
                    constnames::NTuple=())

    return AccelInfo(acceltype, ismaster=ismaster, constvars=constvars,
                    compile=compile, constnames=constnames)
end

function get_kernel!(accel::AccelInfo, path::String)

    return KernelInfo(accel, path)
end

function allocate!(accel::AccelInfo, invars...; innames::NTuple=())

    inargs, indtypes, insizes = argsdtypes(accel, invars)

    launchid = bytes2hex(sha1(string(JAI_ALLOCATE, accel.accelid, dtypes, sizes))[1:4])

    # load shared lib
    if haskey(kinfo.accel.sharedlibs, launchid)
        dlib = kinfo.accel.sharedlibs[launchid]

    elseif kinfo.accel.ismaster
        libpath = build!(accel, JAI_ALLOCATE, launchid, inargs, innames)
        dlib = dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        accel.sharedlibs[launchid] = dlib

    else
        timeout(TIMEOUT)

        dlib = dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        accel.sharedlibs[launchid] = dlib
    end


    dfunc = dlsym(dlib, :allocate)
    argtypes = Meta.parse(string(((indtypes...),)))
    ccallexpr = :(ccall($kfunc, Int64, $argtypes, $(inargs...)))

    @eval return $ccallexpr

end

function allocate!(kernel::KernelInfo, data...)
    return allocate!(kernel.accel, data...)
end

function deallocate!(accel::AccelInfo, data...)
end

function deallocate!(kernel::KernelInfo, data...)
    return deallocate!(kernel.accel, data...)
end

function copyin!(accel::AccelInfo, data...)
end

function copyin!(kernel::KernelInfo, data...)
    return copyin!(kernel.accel, data...)
end


function copyout!(accel::AccelInfo, data...)
end

function copyout!(kernel::KernelInfo, data...)
    return copyout!(kernel.accel, data...)
end

function argsdtypes(ainfo::AccelInfo, data)

    args = []
    dtypes = []
    sizes = []

    for arg in data
        if typeof(arg) <: AbstractArray
            push!(args, arg)
            push!(dtypes, Ptr{typeof(args[end])})

        elseif ainfo.acceltype == JAI_CPP
            push!(args, arg)
            push!(dtypes, typeof(args[end]))

        elseif ainfo.acceltype == JAI_FORTRAN
            push!(args, arg)
            push!(dtypes, Ref{typeof(args[end])})

        end

        push!(sizes, size(args[end]))
    end

    args, dtypes, sizes
end

# kernel launch
function launch!(kinfo::KernelInfo, invars...;
                 innames::NTuple=(), outnames=NTuple=(),
                 outvars::Union{Tuple, Vector}=(),
                 compile::Union{String, Nothing}=nothing,
                 workdir::Union{String, Nothing}=nothing)

    inargs, indtypes, insizes = argsdtypes(kinfo.accel, invars)
    outargs, outdtypes, outsizes = argsdtypes(kinfo.accel, outvars)

    args = vcat(inargs, outargs)
    dtypes = vcat(indtypes, outdtypes)

    launchid = bytes2hex(sha1(string(JAI_LAUNCH, kinfo.kernelid, indtypes, insizes,
                            outdtypes, outsizes))[1:4])

    # load shared lib
    if haskey(kinfo.accel.sharedlibs, launchid)
        dlib = kinfo.accel.sharedlibs[launchid]

    elseif kinfo.accel.ismaster
        libpath = build!(kinfo, launchid, inargs, outargs, innames,
                            outnames, compile, workdir)
        dlib = dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        kinfo.accel.sharedlibs[launchid] = dlib

    else
        timeout(TIMEOUT)

        dlib = dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        kinfo.accel.sharedlibs[launchid] = dlib
    end


    kfunc = dlsym(dlib, :launch)
    argtypes = Meta.parse(string(((dtypes...),)))
    ccallexpr = :(ccall($kfunc, Int64, $argtypes, $(args...)))

    @eval return $ccallexpr

end

function setup_build(acceltype, compile, workdir, launchid)

    if workdir == nothing
        workdir = joinpath(pwd(), ".jaitmp")
    end

    if !isdir(workdir)
        mkdir(workdir)
    end

    if acceltype == JAI_FORTRAN
        srcpath = joinpath(workdir, "F$(launchid).F90")
        if compile == nothing
            compile = "gfortran -fPIC -shared -g"
        end

    elseif  kinfo.accel.acceltype == JAI_CPP
        srcpath = joinpath(workdir, "C$(launchid).cpp")
        if compile == nothing
            compile = "g++ -fPIC -shared -g"

        else 
            compile = accel.compile
        end
    end

    (workdir, srcpath, compile)
end


# kernel build
function build!(kinfo::KernelInfo, launchid::String, inargs::Vector, outargs::Vector,
                innames::NTuple, outnames::NTuple, compile::Union{String, Nothing},
                workdir::Union{String, Nothing})

    if compile == nothing
        compile = kinfo.accel.compile
    end

    if workdir == nothing
        workdir = kinfo.accel.workdir
    end

    workdir, srcpath, compile = setup_build(kinfo.accel.acceltype, compile, workdir, launchid)

    # generate source code
    if !isfile(srcpath)

        generate!(kinfo, srcpath, launchid, inargs, outargs, innames, outnames)

    end

    fname, ext = splitext(basename(srcpath))
    outpath = joinpath(workdir, "S$(launchid).so")

    compilelog = read(run(`$(split(compile)) -o $outpath $(srcpath)`), String)
    #println("COMPIE CMD\n", compile)
    #println("COMPIE LOG\n", compilelog)

    outpath
end


# allocate! build
function build!(ainfo::AccelInfo, buildtype::BuildType, launchid::String, inargs::Vector,
                innames::NTuple)

    if compile == nothing
        compile = accel.compile
    end

    workdir, srcpath, compile = setup_build(accel.acceltype, compile, accel.workdir, launchid)

    # generate source code
    if !isfile(srcpath)
        generate!(accel, buildtype, srcpath, launchid, inargs, innames)

    end

    fname, ext = splitext(basename(srcpath))
    outpath = joinpath(workdir, "S$(launchid).so")

    compilelog = read(run(`$(split(compile)) -o $outpath $(srcpath)`), String)

    outpath
end

# kernel generate
function generate!(kinfo::KernelInfo, srcpath::String, launchid::String, inargs::Vector,
                outargs::Vector, innames::NTuple, outnames::NTuple)

    body = kinfo.kerneldef.body

    if kinfo.accel.acceltype == JAI_FORTRAN
        code = gencode_fortran_kernel(kinfo, launchid, body, inargs, outargs, innames, outnames)

    elseif kinfo.accel.acceltype == JAI_CPP
        code = gencode_cpp_kernel(kinfo, launchid, body, inargs, outargs, innames, outnames)

    elseif kinfo.accel.acceltype == JAI_FORTRAN_OPENACC
        code = gencode_fortran_kernel(kinfo, launchid, body, inargs, outargs, innames, outnames)

    end

    open(srcpath, "w") do io
           write(io, code)
    end

end

# accel generate
function generate!(ainfo::AccelInfo, buildtype::BuildType, srcpath::String,
                    launchid::String, inargs::Vector, innames::NTuple)

    if kinfo.accel.acceltype == JAI_FORTRAN
        code = gencode_fortran_allocate(ainfo, launchid, inargs, innames)

    elseif kinfo.accel.acceltype == JAI_CPP
        code = gencode_cpp_allocate(ainfo, launchid, inargs, innames)

    elseif kinfo.accel.acceltype == JAI_FORTRAN_OPENACC
        code = gencode_fortran_allocate(ainfo, launchid, inargs, innames)

    end

    open(srcpath, "w") do io
           write(io, code)
    end

end

end
