module AccelInterfaces

import Libdl.dlopen,
       Libdl.RTLD_LAZY,
       Libdl.RTLD_DEEPBIND,
       Libdl.RTLD_GLOBAL,
       Libdl.dlsym

import SHA.sha1
import Dates.now,
       Dates.Millisecond

import OffsetArrays.OffsetArray,
       OffsetArrays.OffsetVector

export AccelType, JAI_VERSION, JAI_FORTRAN, JAI_CPP, JAI_FORTRAN_OPENACC,
       JAI_ANYACCEL, JAI_CPP_OPENACC, JAI_HOST, JAI_DEVICE, AccelInfo,
       KernelInfo, get_accel!,get_kernel!, allocate!, deallocate!, update!,
       launch!


const JAI_VERSION = "0.0.1"
const TIMEOUT = 10

@enum DeviceType JAI_HOST JAI_DEVICE

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

JaiConstType = Union{Number, NTuple{N, T}, AbstractArray{T, N}} where {N, T<:Number}
JaiDataType = JaiConstType

struct AccelInfo

    accelid::String
    acceltype::AccelType
    ismaster::Bool
    constvars::NTuple{N,JaiConstType} where {N}
    constnames::NTuple{N, String} where {N}
    compile::Union{String, Nothing}
    sharedlibs::Dict{String, Ptr{Nothing}}
    workdir::Union{String, Nothing}

    function AccelInfo(acceltype::AccelType; ismaster::Bool=true,
                    constvars::NTuple{N,JaiConstType} where {N}=(),
                    compile::Union{String, Nothing}=nothing,
                    constnames::NTuple{N, String} where {N}=(),
                    workdir::Union{String, Nothing}=nothing)

        # TODO: check if acceltype is supported in this system(h/w, compiler, ...)
        #     : detect available acceltypes according to h/w, compiler, flags, ...

        accelid = bytes2hex(sha1(string(Sys.STDLIB, JAI_VERSION,
                        acceltype, constvars, constnames, compile))[1:4])

        if workdir == nothing
            workdir = joinpath(pwd(), ".jaitmp")
        end

        if ismaster && !isdir(workdir)
            mkdir(workdir)
        end

        new(accelid, acceltype, ismaster, constvars, constnames, compile, 
            Dict{String, Ptr{Nothing}}(), workdir)
    end
end

# NOTE: keep the order of includes
include("./kernel.jl")
include("./fortran.jl")
include("./fortran_openacc.jl")
include("./cpp.jl")

function timeout(libpath::String, duration::Real; waittoexist::Bool=true) :: Nothing

    local tstart = now()

    while true
        local check = waittoexist ? ispath(libpath) : ~ispath(libpath)

        if check
            break

        elseif ((now() - tstart)/ Millisecond(1000)) > duration
            error("Timeout occured while waiting for shared library")

        else
            sleep(0.1)
        end
    end
end

function get_accel!(acceltype::AccelType; ismaster::Bool=true,
                    constvars::NTuple{N,JaiConstType} where {N}=(),
                    compile::Union{String, Nothing}=nothing,
                    constnames::NTuple{N, String} where {N}=()) :: AccelInfo

    return AccelInfo(acceltype, ismaster=ismaster, constvars=constvars,
                    compile=compile, constnames=constnames)
end

function get_kernel!(accel::AccelInfo, path::String) :: KernelInfo
    return KernelInfo(accel, path)
end

function accel_method(buildtype::BuildType, accel::AccelInfo,
            data::Vararg{JaiDataType, N} where {N};
            names::NTuple{N, String} where {N}=()) :: Int64

    if accel.acceltype in (JAI_FORTRAN, JAI_CPP)
        return 0::Int64
    end

    dtypes, sizes = argsdtypes(accel, data...)

    args = data

    launchid = bytes2hex(sha1(string(buildtype, accel.accelid, dtypes, sizes))[1:4])

    local libpath = joinpath(accel.workdir, "SL$(launchid).so")

    # load shared lib
    if haskey(accel.sharedlibs, launchid)
        dlib = accel.sharedlibs[launchid]

    else
        build!(accel, buildtype, launchid, libpath, args, names)
        dlib = dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        accel.sharedlibs[launchid] = dlib
    end

    if buildtype == JAI_ALLOCATE
        local dfunc = dlsym(dlib, :jai_allocate)

    elseif buildtype == JAI_COPYIN
        local dfunc = dlsym(dlib, :jai_copyin)

    elseif buildtype == JAI_COPYOUT
        local dfunc = dlsym(dlib, :jai_copyout)

    elseif buildtype == JAI_DEALLOCATE
        local dfunc = dlsym(dlib, :jai_deallocate)

    else
        error(string(buildtype) * " is not supported.")

    end

    local argtypes = Meta.parse(string(((dtypes...),)))
    local ccallexpr = :(ccall($dfunc, Int64, $argtypes, $(args...)))

    @eval return $ccallexpr

end

function allocate!(accel::AccelInfo,
            data::Vararg{JaiDataType, N} where {N};
            names::NTuple{N, String} where {N}=()) :: Int64
    return accel_method(JAI_ALLOCATE, accel, data...; names=names)
end

function allocate!(kernel::KernelInfo,
            data::Vararg{JaiDataType, N} where {N};
            names::NTuple{N, String} where {N}=()) :: Int64
    return allocate!(kernel.accel, data...; names=names)
end

function deallocate!(accel::AccelInfo,
            data::Vararg{JaiDataType, N} where {N};
            names::NTuple{N, String} where {N}=()) :: Int64
    return accel_method(JAI_DEALLOCATE, accel, data...; names=names)
end

function deallocate!(kernel::KernelInfo,
            data::Vararg{JaiDataType, N} where {N};
            names::NTuple{N, String} where {N}=()) :: Int64
    return deallocate!(kernel.accel, data...; names=names)
end

function update!(devtype::DeviceType, accel::AccelInfo,
            data::Vararg{JaiDataType, N} where {N};
            names::NTuple{N, String} where {N}=()) :: Int64

	if devtype == JAI_HOST
		return accel_method(JAI_COPYOUT, accel, data...; names=names)

	elseif devtype == JAI_DEVICE
		return accel_method(JAI_COPYIN, accel, data...; names=names)

	else
		error(string(devtype) * " is not supported.")
	end
end

function update!(devtype::DeviceType, kernel::KernelInfo,
            data::Vararg{JaiDataType, N} where {N};
            names::NTuple{N, String} where {N}=()) :: Int64
    return update!(devtype::DeviceType, kernel.accel, data...; names=names)
end

function argsdtypes(ainfo::AccelInfo,
            data::Vararg{JaiDataType, N} where {N};
        ) :: Tuple{Vector{DataType}, Vector{Tuple{T} where T<:Integer}}

    local N = length(data)

    dtypes = Vector{DataType}(undef, N)
    sizes = Vector{NTuple{M, T} where {M, T<:Integer}}(undef, N)

    for (index, arg) in enumerate(data)
        local arg = data[index]

        sizes[index] = size(arg)

        if typeof(arg) <: AbstractArray
            dtypes[index] = Ptr{typeof(arg)}

        elseif ainfo.acceltype in (JAI_CPP, JAI_CPP_OPENACC)
            dtypes[index] = typeof(arg)

        elseif ainfo.acceltype in (JAI_FORTRAN, JAI_FORTRAN_OPENACC)
            dtypes[index] = Ref{typeof(arg)}

        end
    end

#
#    for arg in data
#        push!(args, arg)
#        push!(sizes, size(args[end]))
#
#        if typeof(arg) <: AbstractArray
#            push!(dtypes, Ptr{typeof(args[end])})
#
#        elseif ainfo.acceltype in (JAI_CPP, JAI_CPP_OPENACC)
#            push!(dtypes, typeof(args[end]))
#
#        elseif ainfo.acceltype in (JAI_FORTRAN, JAI_FORTRAN_OPENACC)
#            push!(dtypes, Ref{typeof(args[end])})
#
#        end
#    end

    dtypes, sizes
end

# kernel launch
function launch!(kinfo::KernelInfo,
            invars::Vararg{JaiDataType, N} where {N};
            innames::NTuple{N, String} where {N}=(),
            outnames::NTuple{N, String} where {N}=(),
            outvars::NTuple{N,JaiDataType} where {N}=(),
            compile::Union{String, Nothing}=nothing)

    indtypes, insizes = argsdtypes(kinfo.accel, invars...)
    inargs = invars

    outdtypes, outsizes = argsdtypes(kinfo.accel, outvars...)
    outargs = outvars

    #args = vcat(inargs, outargs)
    args = (inargs..., outargs...)
    dtypes = vcat(indtypes, outdtypes)

    launchid = bytes2hex(sha1(string(JAI_LAUNCH, kinfo.kernelid, indtypes, insizes,
                            outdtypes, outsizes))[1:4])

    libpath = joinpath(kinfo.accel.workdir, "SL$(launchid).so")

    # load shared lib
    if haskey(kinfo.accel.sharedlibs, launchid)
        dlib = kinfo.accel.sharedlibs[launchid]

    else
        build!(kinfo, launchid, libpath, inargs, outargs,
                innames, outnames, compile)
        dlib = dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        kinfo.accel.sharedlibs[launchid] = dlib
    end


    kfunc = dlsym(dlib, :jai_launch)
    argtypes = Meta.parse(string(((dtypes...),)))
    ccallexpr = :(ccall($kfunc, Int64, $argtypes, $(args...)))


    @eval return $ccallexpr

end

function setup_build(acceltype::AccelType, buildtype::BuildType, launchid::String,
                compile::Union{String, Nothing})

    prefix = ACCEL_CODE[acceltype] * BUILD_CODE[buildtype]
 
    if acceltype == JAI_FORTRAN
        ext = ".F90"
        if compile == nothing
            compile = "gfortran -fPIC -shared -g -ffree-line-length-none"
        end

    elseif  acceltype == JAI_CPP
        ext = ".cpp"
        if compile == nothing
            compile = "g++ -fPIC -shared -g"
        end

    elseif  acceltype == JAI_FORTRAN_OPENACC
        ext = ".F90"
        if compile == nothing
            compile = "gfortran -fPIC -shared -fopenacc -g -ffree-line-length-none"
        end

    else
        error(string(acceltype) * " is not supported yet.")

    end

    (prefix*launchid*ext, compile)
end


# kernel build
function build!(kinfo::KernelInfo, launchid::String, outpath::String,
                inargs::NTuple{N, JaiDataType} where {N},
                outargs::NTuple{M, JaiDataType} where {M},
                innames::NTuple{N, String} where {N},
                outnames::NTuple{M, String} where {M},
                compile::Union{String, Nothing}=nothing) :: String

    if compile == nothing
        compile = kinfo.accel.compile
    end

    srcfile, compile = setup_build(kinfo.accel.acceltype, JAI_LAUNCH, launchid,
                                    compile)

    srcpath = joinpath(kinfo.accel.workdir, srcfile)
    pidfile = outpath * ".pid"

    # generate source code
    if !ispath(outpath)
        code = generate!(kinfo, launchid, inargs, outargs, innames, outnames)

        if !ispath(outpath)

            curdir = pwd()

            try
                procdir = mktempdir()
                cd(procdir)

                open(srcfile, "w") do io
                    write(io, code)
                end

                outfile = basename(outpath)

                if !ispath(outpath)
                    compilelog = read(run(`$(split(compile)) -o $outfile $(srcfile)`), String)

                    if !ispath(outpath)
                        open(pidfile, "w") do io
                            write(io, string(getpid()))
                        end

                        if !ispath(outpath)
                            cp(outfile, outpath)
                        end

                        rm(pidfile)
                    end
                end
            catch err

            finally
                cd(curdir)
            end
        end
    end

    timeout(pidfile, TIMEOUT, waittoexist=false)

    outpath
end


# non-kernel build
function build!(ainfo::AccelInfo, buildtype::BuildType, launchid::String,
                outpath::String,
                inargs::NTuple{N, JaiDataType} where {N},
                innames::NTuple{N, String} where {N}) :: String

    srcfile, compile = setup_build(ainfo.acceltype, buildtype,
                launchid, ainfo.compile)

    srcpath = joinpath(ainfo.workdir, srcfile)
    pidfile = outpath * ".pid"

    # generate source code
    if !ispath(outpath)
        code = generate!(ainfo, buildtype, launchid, inargs, innames)

        if !ispath(outpath)

            curdir = pwd()

            try
                procdir = mktempdir()
                cd(procdir)

                open(srcfile, "w") do io
                       write(io, code)
                end

                outfile = basename(outpath)

                if !ispath(outpath)
                    compilelog = read(run(`$(split(compile)) -o $outfile $(srcfile)`), String)

                    if !ispath(outpath)
                        open(pidfile, "w") do io
                            write(io, string(getpid()))
                        end

                        if !ispath(outpath)
                            cp(outfile, outpath)
                        end

                        rm(pidfile)
                    end
                end
            catch err

            finally
                cd(curdir)
            end
        end
    end

    timeout(pidfile, TIMEOUT, waittoexist=false)

    outpath
end

# kernel generate
function generate!(kinfo::KernelInfo, launchid::String,
                inargs::NTuple{N, JaiDataType} where {N},
                outargs::NTuple{M, JaiDataType} where {M},
                innames::NTuple{N, String} where {N},
                outnames::NTuple{M, String} where {M}) :: String

    body = kinfo.kerneldef.body

    if kinfo.accel.acceltype == JAI_FORTRAN
        code = gencode_fortran_kernel(kinfo, launchid, body, inargs, outargs, innames, outnames)

    elseif kinfo.accel.acceltype == JAI_CPP
        code = gencode_cpp_kernel(kinfo, launchid, body, inargs, outargs, innames, outnames)

    elseif kinfo.accel.acceltype == JAI_FORTRAN_OPENACC
        code = gencode_fortran_kernel(kinfo, launchid, body, inargs, outargs, innames, outnames)

    end

    code
end

# accel generate
function generate!(ainfo::AccelInfo, buildtype::BuildType,
                launchid::String,
                inargs::NTuple{N, JaiDataType} where {N},
                innames::NTuple{N, String} where {N}) :: String


    if ainfo.acceltype == JAI_FORTRAN_OPENACC
        code = gencode_fortran_openacc(ainfo, buildtype, launchid, inargs, innames)

    else
        error(string(ainfo.acceltype) * " is not supported for allocation.")

    end

    code
end

end
