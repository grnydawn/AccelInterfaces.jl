module AccelInterfaces

import Libdl.dlopen,
       Libdl.RTLD_LAZY,
       Libdl.RTLD_DEEPBIND,
       Libdl.RTLD_GLOBAL,
       Libdl.dlsym

import OffsetArrays.OffsetArray,
       OffsetArrays.OffsetVector

export AccelType, FLANG, CLANG, ANYACCEL, AccelInfo, KernelInfo,
        get_accel!,get_kernel!, allocate!, deallocate!, copyin!, copyout!, launch!

@enum AccelType FLANG CLANG ANYACCEL

struct AccelInfo

    acceltype::AccelType
    ismaster::Bool
    constvars::Tuple
    constnames::NTuple
    compile::Union{String, Nothing}
    sharedlibs::Dict
    constants::Dict

    function AccelInfo(acceltype::AccelType; ismaster::Bool=true, constvars::Tuple,
                    compile::Union{String, Nothing}=nothing,
                    constnames::NTuple=())

        new(acceltype, ismaster, constvars, constnames, compile, Dict(), Dict())
    end
end

struct KernelInfo

    accel::AccelInfo
    kernelpath::String
    sharedlibs::Dict

    function KernelInfo(accel::AccelInfo, path::String)

        new(accel, path, Dict())
    end
end

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

    for arg in data
        if typeof(arg) <: OffsetArray
            push!(args, arg.parent)
            push!(dtypes, Ptr{typeof(args[end])})

        elseif typeof(arg) <: AbstractArray
            push!(args, arg)
            push!(dtypes, Ptr{typeof(args[end])})

        elseif ainfo.acceltype == CLANG
            push!(args, arg)
            push!(dtypes, typeof(args[end]))

        elseif ainfo.acceltype == FLANG
            push!(args, arg)
            push!(dtypes, Ref{typeof(args[end])})

        end
    end

    args, dtypes
end
function launch!(kinfo::KernelInfo, invars...;
                 innames::NTuple=(), outnames=NTuple=(),
                 outvars::Union{Tuple, Vector}=(),
                 compile::Union{String, Nothing}=nothing,
                 workdir::Union{String, Nothing}=nothing)

    #println("TTT", length(kernel.accel.constnames))


    inargs, indtypes = argsdtypes(kinfo.accel, invars)
    inoutargs, inoutdtypes = argsdtypes(kinfo.accel, [inv for invar in invars if invar in outvars])
    outargs, outdtypes = argsdtypes(kinfo.accel, outvars)

    args = vcat(inargs, outargs)
    dtypes = vcat(indtypes, outdtypes)

    # generate hash for exitdata

    hashid = hash(("launch!", kinfo.accel.acceltype, indtypes, inoutdtypes, outdtypes))

    # load shared lib
    if haskey(kinfo.sharedlibs, hashid)
        dlib = kinfo.sharedlibs[hashid]

    else
        libpath = build!(kinfo, hashid, inargs, indtypes, inoutargs, inoutdtypes,
                        outargs, outdtypes, innames, outnames, compile, workdir)
        dlib = dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)

    end


    kfunc = dlsym(dlib, :launch)
    argtypes = Meta.parse(string(((dtypes...),)))
    #ccallexpr = :(ccall($kfunc, Int64, $argtypes, $(args...)))
    ccallexpr = :(ccall($kfunc, Int64, ()))

    @eval return $ccallexpr

end

function build!(kinfo::KernelInfo, hashid::UInt64, inargs::Vector, indtypes::Vector,
                inoutargs::Vector, inoutdtypes::Vector, outargs::Vector, outdtypes::Vector,
                innames::NTuple, outnames::NTuple, compile::Union{String, Nothing},
                workdir::Union{String, Nothing})

    if workdir == nothing
        workdir = joinpath(pwd(), ".jaitmp")
    end

    if !isdir(workdir)
        mkdir(workdir)
    end

    if kinfo.accel.acceltype == FLANG
        srcpath = joinpath(workdir, "F$(hashid).F90")
        compile = (compile == nothing ? "gfortran -fPIC -shared" : compile)

    elseif  kinfo.accel.acceltype == CLANG
        srcpath = joinpath(workdir, "C$(hashid).cpp")
        compile = compile == nothing ? "g++ -fPIC -shared" : compile

    end

    # generate source code
    if !isfile(srcpath)

        generate!(kinfo, srcpath, hashid, inargs, indtypes, inoutargs, inoutdtypes,
                        outargs, outdtypes, innames, outnames)

    end

    fname, ext = splitext(basename(srcpath))
    outpath = joinpath(workdir, "S$(hashid).so")

    run(`$(split(compile)) -o $outpath $(srcpath)`)

    outpath
end


function generate!(kinfo::KernelInfo, srcpath::String, hashid::UInt64, inargs::Vector, indtypes::Vector,
                inoutargs::Vector, inoutdtypes::Vector, outargs::Vector, outdtypes::Vector,
                innames::NTuple, outnames::NTuple)


    if kinfo.accel.acceltype == FLANG
        code = Fortran.gencode(kinfo, hashid, inargs, indtypes, inoutargs, inoutdtypes,
                                outargs, outdtypes, innames, outnames)

    elseif kinfo.accel.acceltype == CLANG
        code = CPP.gencode(kinfo, hashid, inargs, indtypes, inoutargs, inoutdtypes,
                                outargs, outdtypes, innames, outnames)

    end

    open(srcpath, "w") do io
           write(io, code)
    end

end

end
