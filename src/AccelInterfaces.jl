module AccelInterfaces

import Libdl.dlopen,
       Libdl.RTLD_LAZY,
       Libdl.RTLD_DEEPBIND,
       Libdl.RTLD_GLOBAL,
       Libdl.dlsym

import SHA.sha1

import OffsetArrays.OffsetArray,
       OffsetArrays.OffsetVector

export AccelType, JAI_VERSION, JAI_FORTRAN, JAI_CPP, JAI_ANYACCEL, AccelInfo,
        KernelInfo, get_accel!,get_kernel!, allocate!, deallocate!, copyin!,
        copyout!, launch!

const TIMEOUT = 60
const JAI_VERSION = "0.0.1"

@enum AccelType JAI_FORTRAN JAI_CPP JAI_ANYACCEL JAI_HEADER

struct AccelInfo

    accelid::String
    acceltype::AccelType
    ismaster::Bool
    constvars::Tuple
    constnames::NTuple
    compile::Union{String, Nothing}
    sharedlibs::Dict

    function AccelInfo(acceltype::AccelType; ismaster::Bool=true,
                    constvars::Tuple=(), compile::Union{String, Nothing}=nothing,
                    constnames::NTuple=())

        # TODO: check if acceltype is supported in this system(h/w, compiler, ...)
        #     : detect available acceltypes according to h/w, compiler, flags, ...

        accelid = bytes2hex(sha1(string(Sys.STDLIB, JAI_VERSION,
                        acceltype, constvars, constnames, compile))[1:4])

        new(accelid, acceltype, ismaster, constvars, constnames, compile, Dict())
    end
end

include("./kernel.jl")
include("./fortran.jl")
include("./cpp.jl")


function get_accel!(acceltype::AccelType; ismaster::Bool=true, constvars::Tuple=(),
                    compile::Union{String, Nothing}=nothing,
                    constnames::NTuple=())

    return AccelInfo(acceltype, ismaster=ismaster, constvars=constvars,
                    compile=compile, constnames=constnames)
end

function get_kernel!(accel::AccelInfo, path::String)

    return KernelInfo(accel, path)
end

function allocate!(accel::AccelInfo, data...)
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
        if typeof(arg) <: OffsetArray
            push!(args, arg.parent)
            push!(dtypes, Ptr{typeof(args[end])})

        elseif typeof(arg) <: AbstractArray
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

function launch!(kinfo::KernelInfo, invars...;
                 innames::NTuple=(), outnames=NTuple=(),
                 outvars::Union{Tuple, Vector}=(),
                 compile::Union{String, Nothing}=nothing,
                 workdir::Union{String, Nothing}=nothing)


    inargs, indtypes, insizes = argsdtypes(kinfo.accel, invars)
    outargs, outdtypes, outsizes = argsdtypes(kinfo.accel, outvars)

    args = vcat(inargs, outargs)
    dtypes = vcat(indtypes, outdtypes)

    launchid = bytes2hex(sha1(string(kinfo.kernelid, indtypes, insizes,
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
        tstart = now()
        while true
            if isfile(libpath)
                break

            elseif now() - tstart > TIMEOUT
                error("Timeout occured while waiting for shared library")

            else
                sleep(0.1)
            end
        end

        dlib = dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        kinfo.accel.sharedlibs[launchid] = dlib
    end


    kfunc = dlsym(dlib, :launch)
    argtypes = Meta.parse(string(((dtypes...),)))
    ccallexpr = :(ccall($kfunc, Int64, $argtypes, $(args...)))

    @eval return $ccallexpr

end

function build!(kinfo::KernelInfo, launchid::String, inargs::Vector, outargs::Vector,
                innames::NTuple, outnames::NTuple, compile::Union{String, Nothing},
                workdir::Union{String, Nothing})

    if workdir == nothing
        workdir = joinpath(pwd(), ".jaitmp")
    end

    if !isdir(workdir)
        mkdir(workdir)
    end

    if kinfo.accel.acceltype == JAI_FORTRAN
        srcpath = joinpath(workdir, "F$(launchid).F90")
        compile = (compile == nothing ? "gfortran -fPIC -shared" : compile)

    elseif  kinfo.accel.acceltype == JAI_CPP
        srcpath = joinpath(workdir, "C$(launchid).cpp")
        compile = compile == nothing ? "g++ -fPIC -shared -g" : compile

    end

    # generate source code
    if !isfile(srcpath)

        generate!(kinfo, srcpath, launchid, inargs, outargs, innames, outnames)

    end

    fname, ext = splitext(basename(srcpath))
    outpath = joinpath(workdir, "S$(launchid).so")

    run(`$(split(compile)) -o $outpath $(srcpath)`)

    outpath
end


function generate!(kinfo::KernelInfo, srcpath::String, launchid::String, inargs::Vector,
                outargs::Vector, innames::NTuple, outnames::NTuple)

    body = kinfo.kerneldef.body

    if kinfo.accel.acceltype == JAI_FORTRAN
        code = gencode_fortran(kinfo, launchid, body, inargs, outargs, innames, outnames)

    elseif kinfo.accel.acceltype == JAI_CPP
        code = gencode_cpp(kinfo, launchid, body, inargs, outargs, innames, outnames)

    end

    open(srcpath, "w") do io
           write(io, code)
    end

end

end
