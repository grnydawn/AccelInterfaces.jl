module AccelInterfaces

import Libdl.dlopen,
       Libdl.RTLD_LAZY,
       Libdl.RTLD_DEEPBIND,
       Libdl.RTLD_GLOBAL,
       Libdl.dlsym

import SHA.sha1
import Dates.now

import OffsetArrays.OffsetArray,
       OffsetArrays.OffsetVector

export AccelType, JAI_VERSION, JAI_FORTRAN, JAI_CPP, JAI_FORTRAN_OPENACC,
       JAI_ANYACCEL, JAI_CPP_OPENACC, AccelInfo,
       KernelInfo, get_accel!,get_kernel!, allocate!, deallocate!, copyin!,
       copyout!, launch!

const TIMEOUT = 60

const JAI_VERSION = "0.0.1"

@enum AccelType begin
        JAI_FORTRAN
        JAI_FORTRAN_OPENACC
        JAI_FORTRAN_OMPTARGET
        JAI_CPP
        JAI_CPP_OPENACC
        JAI_CPP_OMPTARGET
        JAI_ANYACCEL
        JAI_HEADER
end


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

# NOTE: keep the order of includes
include("./kernel.jl")
include("./fortran.jl")
include("./fortran_openacc.jl")
include("./cpp.jl")

function timeout(libpath::String, duration::Integer)

    tstart = now()
    while true
        if isfile(libpath)
            break

        elseif convert(Dates.Second, (now() - tstart)) > duration
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

function accel_method(buildtype::BuildType, accel::AccelInfo, data...; names::NTuple=())

    if accel.acceltype in (JAI_FORTRAN, JAI_CPP)
        return
    end

    args, dtypes, sizes = argsdtypes(accel, data)

    launchid = bytes2hex(sha1(string(buildtype, accel.accelid, dtypes, sizes))[1:4])

    workdir = accel.workdir == nothing ? joinpath(pwd(), ".jaitmp") : accel.workdir

    if !isdir(workdir)
        mkdir(workdir)
    end

    libpath = joinpath(workdir, "SL$(launchid).so")

    # load shared lib
    if haskey(accel.sharedlibs, launchid)
        dlib = accel.sharedlibs[launchid]

    elseif accel.ismaster
        build!(accel, buildtype, launchid, libpath, workdir, args, names)
        dlib = dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        accel.sharedlibs[launchid] = dlib

    else
        timeout(libpath, TIMEOUT)

        dlib = dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        accel.sharedlibs[launchid] = dlib
    end

    if buildtype == JAI_ALLOCATE
        dfunc = dlsym(dlib, :jai_allocate)

    elseif buildtype == JAI_COPYIN
        dfunc = dlsym(dlib, :jai_copyin)

    elseif buildtype == JAI_COPYOUT
        dfunc = dlsym(dlib, :jai_copyout)

    elseif buildtype == JAI_DEALLOCATE
        dfunc = dlsym(dlib, :jai_deallocate)

    else
        error(string(buildtype) * " is not supported.")

    end

    argtypes = Meta.parse(string(((dtypes...),)))
    ccallexpr = :(ccall($dfunc, Int64, $argtypes, $(args...)))

    @eval return $ccallexpr

end

function allocate!(accel::AccelInfo, data...; names::NTuple=())
    accel_method(JAI_ALLOCATE, accel, data...; names=names)
end

function allocate!(kernel::KernelInfo, data...; names::NTuple=())
    return allocate!(kernel.accel, data...; names=names)
end

function deallocate!(accel::AccelInfo, data...; names::NTuple=())
    accel_method(JAI_DEALLOCATE, accel, data...; names=names)
end

function deallocate!(kernel::KernelInfo, data...; names::NTuple=())
    return deallocate!(kernel.accel, data...; names=names)
end

function copyin!(accel::AccelInfo, data...; names::NTuple=())
    accel_method(JAI_COPYIN, accel, data...; names=names)
end

function copyin!(kernel::KernelInfo, data...; names::NTuple=())
    return copyin!(kernel.accel, data...; names=names)
end

function copyout!(accel::AccelInfo, data...; names::NTuple=())
    accel_method(JAI_COPYOUT, accel, data...; names=names)
end

function copyout!(kernel::KernelInfo, data...; names::NTuple=())
    return copyout!(kernel.accel, data...; names=names)
end

function argsdtypes(ainfo::AccelInfo, data)

    args = []
    dtypes = []
    sizes = []

    for arg in data
        if typeof(arg) <: AbstractArray
            push!(args, arg)
            push!(dtypes, Ptr{typeof(args[end])})

        elseif ainfo.acceltype in (JAI_CPP, JAI_CPP_OPENACC)
            push!(args, arg)
            push!(dtypes, typeof(args[end]))

        elseif ainfo.acceltype in (JAI_FORTRAN, JAI_FORTRAN_OEPNACC)
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

    workdir = kinfo.accel.workdir == nothing ? joinpath(pwd(), ".jaitmp") : kinfo.accel.workdir

    if !isdir(workdir)
        mkdir(workdir)
    end

    libpath = joinpath(workdir, "SL$(launchid).so")

    # load shared lib
    if haskey(kinfo.accel.sharedlibs, launchid)
        dlib = kinfo.accel.sharedlibs[launchid]

    elseif kinfo.accel.ismaster
        build!(kinfo, launchid, libpath, workdir, inargs, outargs,
                innames, outnames, compile)
        dlib = dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        kinfo.accel.sharedlibs[launchid] = dlib

    else
        timeout(libpath, TIMEOUT)

        dlib = dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        kinfo.accel.sharedlibs[launchid] = dlib
    end


    kfunc = dlsym(dlib, :jai_launch)
    argtypes = Meta.parse(string(((dtypes...),)))
    ccallexpr = :(ccall($kfunc, Int64, $argtypes, $(args...)))

    @eval return $ccallexpr

end

function setup_build(acceltype::AccelType, buildtype::BuildType, launchid::String,
                workdir::String, compile::Union{String, Nothing})

    prefix = ACCEL_CODE[acceltype] * BUILD_CODE[buildtype]
 
    if acceltype == JAI_FORTRAN
        srcpath = joinpath(workdir, "$(prefix)F$(launchid).F90")
        if compile == nothing
            compile = "gfortran -fPIC -shared -g"
        end

    elseif  acceltype == JAI_CPP
        srcpath = joinpath(workdir, "$(prefix)$(launchid).cpp")
        if compile == nothing
            compile = "g++ -fPIC -shared -g"
        end

    elseif  acceltype == JAI_FORTRAN_OPENACC
        srcpath = joinpath(workdir, "$(prefix)$(launchid).F90")
        if compile == nothing
            compile = "gfortran -fPIC -shared -fopenacc -g"
        end

    else
        error(string(acceltype) * " is not supported yet.")

    end

    (srcpath, compile)
end


# kernel build
function build!(kinfo::KernelInfo, launchid::String, outpath::String, workdir::String,
                inargs::Vector, outargs::Vector, innames::NTuple, outnames::NTuple,
                compile::Union{String, Nothing})

    if compile == nothing
        compile = kinfo.accel.compile
    end

    srcpath, compile = setup_build(kinfo.accel.acceltype, JAI_LAUNCH, launchid,
                                    workdir, compile)

    # generate source code
    if !isfile(srcpath)

        generate!(kinfo, srcpath, launchid, inargs, outargs, innames, outnames)

    end

    fname, ext = splitext(basename(srcpath))

    compilelog = read(run(`$(split(compile)) -o $outpath $(srcpath)`), String)
    #println("COMPIE CMD\n", compile)
    #println("COMPIE LOG\n", compilelog)

    outpath
end


# non-kernel build
function build!(ainfo::AccelInfo, buildtype::BuildType, launchid::String,
                outpath::String, workdir::String, inargs::Vector, innames::NTuple)

    srcpath, compile = setup_build(ainfo.acceltype, buildtype,
                launchid, workdir, ainfo.compile)

    # generate source code
    if !isfile(srcpath)
        generate!(ainfo, buildtype, srcpath, launchid, inargs, innames)

    end

    fname, ext = splitext(basename(srcpath))

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

    if ainfo.acceltype == JAI_FORTRAN_OPENACC
        code = gencode_fortran_openacc(ainfo, buildtype, launchid, inargs, innames)

    else
        error(string(ainfo.acceltype) * " is not supported for allocation.")

    end

    open(srcpath, "w") do io
           write(io, code)
    end

end

end
