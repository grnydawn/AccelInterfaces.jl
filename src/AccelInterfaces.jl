module AccelInterfaces

using Serialization

import Pkg.TOML

import Libdl.dlopen,
       Libdl.RTLD_LAZY,
       Libdl.RTLD_DEEPBIND,
       Libdl.RTLD_GLOBAL,
       Libdl.dlext,
       Libdl.dlsym

import SHA.sha1
import Dates.now,
       Dates.Millisecond

import OffsetArrays.OffsetArray,
       OffsetArrays.OffsetVector

# TODO: simplified user interface
# [myaccel1] = @jaccel
# [mykernel1] = @jkernel kernel_text
# retval = @jlaunch([mykernel1,] x, y; output=(z,))

export JAI_VERSION, @jenterdata, @jexitdata, @jlaunch, @jaccel, @jkernel, @jdecel, @jwait

        
const JAI_VERSION = TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"]
const TIMEOUT = 10

@enum BuildType::Int64 begin
    JAI_ACCEL       = 10
    JAI_ALLOCATE    = 20
    JAI_UPDATETO    = 30
    JAI_LAUNCH      = 40
    JAI_UPDATEFROM  = 50
    JAI_DEALLOCATE  = 60
    JAI_WAIT        = 70
end

@enum AccelType begin
        JAI_FORTRAN
        JAI_FORTRAN_OPENACC
        JAI_FORTRAN_OMPTARGET
        JAI_CPP
        JAI_CPP_OPENACC
        JAI_CPP_OMPTARGET
        JAI_HIP
        JAI_CUDA
        JAI_ANYACCEL
        JAI_HEADER
end

const _accelmap = Dict{String, AccelType}(
    "fortran" => JAI_FORTRAN,
    "fortran_openacc" => JAI_FORTRAN_OPENACC,
    "fortran_omptarget" => JAI_FORTRAN_OMPTARGET,
    "cpp" => JAI_CPP,
    "cpp_openacc" => JAI_CPP_OPENACC,
    "cpp_omptarget" => JAI_CPP_OMPTARGET,
    "hip" => JAI_HIP,
    "cuda" => JAI_CUDA,
    "any" => JAI_ANYACCEL
)

function _cmap(jc, jcf, c, cf) :: String
    return ((haskey(ENV, jc) ? ENV[jc] : (haskey(ENV, c) ? ENV[c] : "")) * " " *
            (haskey(ENV, jcf) ? ENV[jcf] : (haskey(ENV, cf) ? ENV[cf] : "")))
end

#const _compilemap = Dict{String, String}(
#    "fortran" => _cmap("JAI_FC", "JAI_FFLAGS", "FC", "FFLAGS"),
#    "fortran_openacc" => _cmap("JAI_FC", "JAI_FFLAGS", "FC", "FFLAGS"),
#    "fortran_omptarget" => _cmap("JAI_FC", "JAI_FFLAGS", "FC", "FFLAGS"),
#    "cpp" => _cmap("JAI_CXX", "JAI_CXXFLAGS", "CXX", "CXXFLAGS"),
#    "cpp_openacc" => _cmap("JAI_CXX", "JAI_CXXFLAGS", "CXX", "CXXFLAGS"),
#    "cpp_omptarget" => _cmap("JAI_CXX", "JAI_CXXFLAGS", "CXX", "CXXFLAGS"),
#    "hip" => _cmap("JAI_CXX", "JAI_CXXFLAGS", "CXX", "CXXFLAGS"),
#    "cuda" => _cmap("JAI_CXX", "JAI_CXXFLAGS", "CXX", "CXXFLAGS")
#)


const JaiConstType = Union{Number, NTuple{N, T}, AbstractArray{T, N}} where {N, T<:Number}
const JaiDataType = JaiConstType

struct AccelInfo

    accelid::String
    acceltype::AccelType
    ismaster::Bool
    device_num::Int64
    const_vars::NTuple{N,JaiConstType} where {N}
    const_names::NTuple{N, String} where {N}
    compile::Union{String, Nothing}
    sharedlibs::Dict{String, Ptr{Nothing}}
    workdir::Union{String, Nothing}
    debugdir::Union{String, Nothing}
    ccallcache::Dict{Tuple{BuildType, Int64, Int64, String},
                    Tuple{Ptr{Nothing}, Vector{DataType}}}

    function AccelInfo(;master::Bool=true,
            const_vars::NTuple{N,JaiConstType} where {N}=(),
            const_names::NTuple{N, String} where {N}=(),
            framework::NTuple{N, Tuple{String, Union{NTuple{M, Tuple{String, Union{String, Nothing}}}, String, Nothing}}} where {N, M}=nothing,
            device::NTuple{N, Integer} where {N}=(),
            compile::NTuple{N, String} where {N}=(),
            workdir::Union{String, Nothing}=nothing,
            debugdir::Union{String, Nothing}=nothing,
            _lineno_::Union{Int64, Nothing}=nothing,
            _filepath_::Union{String, Nothing}=nothing)

        # TODO: check if acceltype is supported in this system(h/w, compiler, ...)
        #     : detect available acceltypes according to h/w, compiler, flags, ...


        if workdir == nothing
            workdir = joinpath(pwd(), ".jaitmp")
        end

        if master && !isdir(workdir)
            mkdir(workdir)
        end

        if debugdir != nothing
            debugdir = abspath(debugdir)
            if master && !isdir(debugdir)
                mkdir(debugdir)
            end
        end

        io = IOBuffer()
        ser = serialize(io, (Sys.STDLIB, JAI_VERSION, const_vars,
                        const_names, _lineno_, _filepath_))
        accelid = bytes2hex(sha1(String(take!(io)))[1:4])

        dlib = nothing
        acceltype = nothing   
        sharedlibs = nothing   
        compile = nothing   

        # TODO: support multiple framework arguments
        for (frameworkname, frameconfig) in framework
            acceltype = _accelmap[frameworkname]

            if frameconfig isa Nothing
                if startswith(frameworkname, "fortran")
                    compile = ((haskey(ENV, "JAI_FC") ? ENV["JAI_FC"] :
                                    (haskey(ENV, "FC") ? ENV["FC"] : "")) * " " *
                               (haskey(ENV, "JAI_FFLAGS") ? ENV["JAI_FFLAGS"] :
                                    (haskey(ENV, "FFLAGS") ? ENV["FFLAGS"] : "")))

                elseif startswith(frameworkname, "cpp")
                    compile = ((haskey(ENV, "JAI_CXX") ? ENV["JAI_CXX"] :
                                    (haskey(ENV, "CXX") ? ENV["CXX"] : "")) * " " *
                               (haskey(ENV, "JAI_CXXFLAGS") ? ENV["JAI_CXXFLAGS"] :
                                    (haskey(ENV, "CXXFLAGS") ? ENV["CXXFLAGS"] : "")))
                end

            elseif frameconfig isa String
                compile = frameconfig

            else
                for (cfgname, cfg) in frameconfig
                    if cfgname == "compile"
                        compile = cfg
                    end
                end

            end

            if compile == nothing
                error("No compile information is available.")
            end

            io = IOBuffer()
            ser = serialize(io, (accelid, acceltype, compile))
            accelid = bytes2hex(sha1(String(take!(io)))[1:4])

            libpath = joinpath(workdir, "SL" * frameworkname * JAI_VERSION * "." * dlext)

            try
                build_accel!(workdir, debugdir, acceltype, compile, frameworkname, libpath)

                dlib = dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
             
                sharedlibs = Dict{String, Ptr{Nothing}}()
                sharedlibs[accelid] = dlib

            catch err

            end

            break
        end

        buf = fill(-1, 1)

        dfunc = dlsym(dlib, :jai_get_num_devices)
        ccall(dfunc, Int64, (Ptr{Vector{Int64}},), buf)
        if buf[1] < 1
            error("The number of devices is less than 1.")
        end

        if length(device) == 1
            buf[1] = device[1]
            dfunc = dlsym(dlib, :jai_set_device_num)
            ccall(dfunc, Int64, (Ptr{Vector{Int64}},), buf)
            device_num = device[1]

        else
            buf[1] = -1
            dfunc = dlsym(dlib, :jai_get_device_num)
            ccall(dfunc, Int64, (Ptr{Vector{Int64}},), buf)
            device_num = buf[1]

        end
               
        new(accelid, acceltype, master, device_num, const_vars,
            const_names, compile, sharedlibs,
            workdir, debugdir, Dict{Tuple{BuildType, Int64, Int64, String},
            Tuple{Ptr{Nothing}, Expr}}())
    end
end

const _accelcache = Dict{String, AccelInfo}()

function jai_accel_init(name::String; master::Bool=true,
            const_vars::NTuple{N,JaiConstType} where {N}=(),
            const_names::NTuple{N, String} where {N}=(),
            framework::NTuple{N, Tuple{String, Union{NTuple{M, Tuple{String, Union{String, Nothing}}}, String, Nothing}}} where {N, M}=nothing,
            device::NTuple{N, Integer} where {N}=(),
            compile::NTuple{N, String} where {N}=(),
            workdir::Union{String, Nothing}=nothing,
            debugdir::Union{String, Nothing}=nothing,
            _lineno_::Union{Int64, Nothing}=nothing,
            _filepath_::Union{String, Nothing}=nothing)

    accel = AccelInfo(master=master, const_vars=const_vars, const_names=const_names, 
                    framework=framework, device=device, compile=compile,
                    workdir=workdir, debugdir=debugdir, _lineno_=_lineno_,
                    _filepath_=_filepath_)

    global _accelcache[name] = accel
end

function jai_accel_fini(name::String;
            _lineno_::Union{Int64, Nothing}=nothing,
            _filepath_::Union{String, Nothing}=nothing)

    delete!(_accelcache, name)
end

function jai_accel_wait(name::String;
            _lineno_::Union{Int64, Nothing}=nothing,
            _filepath_::Union{String, Nothing}=nothing)

    accel = _accelcache[name]

    if accel.acceltype in (JAI_FORTRAN, JAI_CPP)
        return 0::Int64
    end

    # load shared lib
    dlib = accel.sharedlibs[accel.accelid]
    local dfunc = dlsym(dlib, :jai_wait)

    ccallexpr = :(ccall($dfunc, Int64, ()))
    retval = @eval $ccallexpr

end

# NOTE: keep the order of the following includes
include("./kernel.jl")
include("./fortran.jl")
include("./fortran_openacc.jl")
include("./fortran_omptarget.jl")
include("./cpp.jl")
include("./hip.jl")

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

#function get_accel(acceltype::AccelType; ismaster::Bool=true,
#                    const_vars::NTuple{N,JaiConstType} where {N}=(),
#                    compile::Union{String, Nothing}=nothing,
#                    const_names::NTuple{N, String} where {N}=()) :: AccelInfo
#
#    return AccelInfo(acceltype, ismaster=ismaster, const_vars=const_vars,
#                    compile=compile, const_names=const_names)
#end
#
#function get_kernel(accel::AccelInfo, path::String) :: KernelInfo
#    return KernelInfo(accel, path)
#end

function jai_directive(accel::String, buildtype::BuildType,
            buildtypecount::Int64,
            data::Vararg{JaiDataType, N} where {N};
            names::NTuple{N, String} where {N}=(),
            control::Vector{String},
            _lineno_::Union{Int64, Nothing}=nothing,
            _filepath_::Union{String, Nothing}=nothing) :: Int64

    accel = _accelcache[accel]

    data = (accel.device_num, data...)
    names = ("jai_arg_device_num", names...)

    if accel.acceltype in (JAI_FORTRAN, JAI_CPP)
        return 0::Int64
    end

    cachekey = (buildtype, buildtypecount, _lineno_, _filepath_)

    if _lineno_ isa Int64 && _filepath_ isa String
        if haskey(accel.ccallcache, cachekey)
            dfunc, dtypes = accel.ccallcache[cachekey]
            ccallexpr = :(ccall($dfunc, Int64, ($(dtypes...),), $(data...)))
            retval = @eval $ccallexpr
            return retval
        end
    end

    dtypes, sizes = argsdtypes(accel, data...)

    io = IOBuffer()
    ser = serialize(io, (buildtype, accel.accelid, dtypes, sizes))
    launchid = bytes2hex(sha1(String(take!(io)))[1:4])

    local libpath = joinpath(accel.workdir, "SL$(launchid)." * dlext)

    # load shared lib
    if haskey(accel.sharedlibs, launchid)
        dlib = accel.sharedlibs[launchid]

    else
        build_directive!(accel, buildtype, launchid, libpath, data, names, control)
        dlib = dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        accel.sharedlibs[launchid] = dlib
    end

    if buildtype == JAI_ALLOCATE
        local dfunc = dlsym(dlib, :jai_allocate)

    elseif buildtype == JAI_UPDATETO
        local dfunc = dlsym(dlib, :jai_updateto)

    elseif buildtype == JAI_UPDATEFROM
        local dfunc = dlsym(dlib, :jai_updatefrom)

    elseif buildtype == JAI_DEALLOCATE
        local dfunc = dlsym(dlib, :jai_deallocate)

    else
        error(string(buildtype) * " is not supported.")

    end

    # TODO: find out how to avoid @eval every time
    # TODO: find out how to avoid "ccall($dfunc, Int64, ($(dtypes...),)" part evert time

    ccallexpr = :(ccall($dfunc, Int64, ($(dtypes...),), $(data...)))

    if _lineno_ isa Int64 && _filepath_ isa String
        accel.ccallcache[cachekey] = (dfunc, dtypes)
    end

    retval = @eval $ccallexpr

end

function argsdtypes(ainfo::AccelInfo,
            data::Vararg{JaiDataType, N} where {N};
            ) :: Tuple{Vector{DataType}, Vector{NTuple{M, T} where {M, T<:Integer}}}

    local N = length(data)

    dtypes = Vector{DataType}(undef, N)
    sizes = Vector{NTuple{M, T} where {M, T<:Integer}}(undef, N)

    for (index, arg) in enumerate(data)
        local arg = data[index]

        sizes[index] = size(arg)

        if typeof(arg) <: AbstractArray
            dtype = Ptr{typeof(arg)}

        elseif ainfo.acceltype in (JAI_CPP, JAI_CPP_OPENACC)
            dtype = typeof(arg)

        elseif ainfo.acceltype in (JAI_FORTRAN, JAI_FORTRAN_OPENACC,
                    JAI_FORTRAN_OMPTARGET)
            dtype = Ref{typeof(arg)}
        end

        dtypes[index] = dtype
    end

    dtypes, sizes
end

# kernel launch
function launch_kernel(kname::String,
            invars::Vararg{JaiDataType, N} where {N};
            innames::NTuple{N, String} where {N}=(),
            outnames::NTuple{N, String} where {N}=(),
            output::NTuple{N,JaiDataType} where {N}=(),
            _lineno_::Union{Int64, Nothing}=nothing,
            _filepath_::Union{String, Nothing}=nothing) :: Int64

    kinfo = _kernelcache[kname]

    invars = (kinfo.accel.device_num, invars...)
    innames = ("jai_arg_device_num", innames...)

    args = (invars..., output...)
    cachekey = (JAI_LAUNCH, 0::Int64, _lineno_, _filepath_)

    if _lineno_ isa Int64 && _filepath_ isa String
        if haskey(kinfo.accel.ccallcache, cachekey)
            kfunc, ddtypes = kinfo.accel.ccallcache[cachekey]
            ccallexpr = :(ccall($kfunc, Int64, ($(ddtypes...),), $(args...)))
            retval = @eval $ccallexpr
            return retval
        end
    end

    indtypes, insizes = argsdtypes(kinfo.accel, invars...)
    outdtypes, outsizes = argsdtypes(kinfo.accel, output...)
    dtypes = vcat(indtypes, outdtypes)

    io = IOBuffer()
    ser = serialize(io, (JAI_LAUNCH, kinfo.kernelid, indtypes, insizes, outdtypes, outsizes))
    launchid = bytes2hex(sha1(String(take!(io)))[1:4])

    libpath = joinpath(kinfo.accel.workdir, "SL$(launchid)." * dlext)

    # load shared lib
    if haskey(kinfo.accel.sharedlibs, launchid)
        dlib = kinfo.accel.sharedlibs[launchid]

    else
        build_kernel!(kinfo, launchid, libpath, invars, output, innames, outnames)
        dlib = dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        kinfo.accel.sharedlibs[launchid] = dlib
    end

    kfunc = dlsym(dlib, :jai_launch)
    ccallexpr = :(ccall($kfunc, Int64, ($(dtypes...),), $(args...)))

    if _lineno_ isa Int64 && _filepath_ isa String
        kinfo.accel.ccallcache[cachekey] = (kfunc, dtypes)
    end

    retval = @eval $ccallexpr

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

    elseif  acceltype == JAI_FORTRAN_OMPTARGET
        ext = ".F90"
        if compile == nothing
            compile = "gfortran -fPIC -shared -fopenmp -g -ffree-line-length-none"
        end

    else
        error(string(acceltype) * " is not supported yet.")

    end

    (prefix*launchid*ext, compile)
end


function _gensrcfile(outpath::String, srcfile::String, code::String,
    debugdir::Union{String, Nothing}, compile::String, pidfile::String)

    if !ispath(outpath)

        curdir = pwd()

        try
            outpath = abspath(outpath)
            pidfile = abspath(pidfile)

            if debugdir != nothing
                debugfile = joinpath(abspath(debugdir), srcfile)
                if !ispath(debugfile)
                    open(debugfile, "w") do io
                        write(io, code)
                    end
                end
            end

            procdir = mktempdir()
            cd(procdir)

            open(srcfile, "w") do io
                write(io, code)
            end

            outfile = basename(outpath)

            if !ispath(outpath)
                compilelog = read(run(`$(split(compile)) -o $outfile $(srcfile)`), String)

                if !ispath(outpath) && !ispath(pidfile)
                    open(pidfile, "w") do io
                        write(io, string(getpid()))
                    end

                    if !ispath(outpath) && ispath(pidfile)
                        cp(outfile, outpath)
                    end

                    if ispath(pidfile)
                        rm(pidfile)
                    end
                end
            end
        catch err
            println("X4", string(err))

        finally
            cd(curdir)
        end
    end
end

# accel build
function build_accel!(workdir::String, debugdir::Union{String, Nothing}, acceltype::AccelType,
    compile::Union{String, Nothing}, accelid::String, outpath::String) :: String

    srcfile, compile = setup_build(acceltype, JAI_ACCEL, accelid, compile)

    srcpath = joinpath(workdir, srcfile)
    pidfile = outpath * ".pid"

    # generate source code
    if !ispath(outpath)

        code = generate_accel!(workdir, acceltype, compile, accelid)

        _gensrcfile(outpath, srcfile, code, debugdir, compile, pidfile)
    end

    timeout(pidfile, TIMEOUT, waittoexist=false)

    outpath

end

# kernel build
function build_kernel!(kinfo::KernelInfo, launchid::String, outpath::String,
                inargs::NTuple{N, JaiDataType} where {N},
                outargs::NTuple{M, JaiDataType} where {M},
                innames::NTuple{N, String} where {N},
                outnames::NTuple{M, String} where {M}) :: String

    compile = kinfo.accel.compile

    srcfile, compile = setup_build(kinfo.accel.acceltype, JAI_LAUNCH, launchid,
                                    compile)

    srcpath = joinpath(kinfo.accel.workdir, srcfile)
    pidfile = outpath * ".pid"

    # generate source code
    if !ispath(outpath)

        code = generate_kernel!(kinfo, launchid, inargs, outargs, innames, outnames)

        _gensrcfile(outpath, srcfile, code, kinfo.accel.debugdir, compile, pidfile)
    end

    timeout(pidfile, TIMEOUT, waittoexist=false)

    outpath
end


# directive build
function build_directive!(ainfo::AccelInfo, buildtype::BuildType, launchid::String,
                outpath::String,
                args::NTuple{N, JaiDataType} where {N},
                names::NTuple{N, String} where {N},
                control::Vector{String}) :: String

    srcfile, compile = setup_build(ainfo.acceltype, buildtype,
                launchid, ainfo.compile)

    srcpath = joinpath(ainfo.workdir, srcfile)
    pidfile = outpath * ".pid"

    # generate source code
    if !ispath(outpath)
        code = generate_directive!(ainfo, buildtype, launchid, args, names, control)

        _gensrcfile(outpath, srcfile, code, ainfo.debugdir, compile, pidfile)
    end

    timeout(pidfile, TIMEOUT, waittoexist=false)

    outpath
end

# accel generate
function generate_accel!(workdir::String, acceltype::AccelType,
        compile::Union{String, Nothing}, accelid::String) :: String

    if acceltype == JAI_FORTRAN
        code = gencode_fortran_accel(accelid)

    elseif acceltype == JAI_CPP
        code = gencode_cpp_accel()

    elseif acceltype == JAI_FORTRAN_OPENACC
        code = gencode_fortran_openacc_accel(accelid)

    elseif acceltype == JAI_FORTRAN_OMPTARGET
        code = gencode_fortran_omptarget_accel(accelid)

    else
        error(string(acceltype) * " is not supported yet.")
    end

    code

end

# kernel generate
function generate_kernel!(kinfo::KernelInfo, launchid::String,
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

    elseif kinfo.accel.acceltype == JAI_FORTRAN_OMPTARGET
        code = gencode_fortran_kernel(kinfo, launchid, body, inargs, outargs, innames, outnames)

    elseif kinfo.accel.acceltype == JAI_HIP
        code = gencode_cpp_kernel(kinfo, launchid, body, inargs, outargs, innames, outnames)

    else
        error(string(kinfo.accel.acceltype) * " is not supported yet.")
    end

    code
end

# accel generate
function generate_directive!(ainfo::AccelInfo, buildtype::BuildType,
                launchid::String,
                args::NTuple{N, JaiDataType} where {N},
                names::NTuple{N, String} where {N},
                control::Vector{String}) :: String


    if ainfo.acceltype == JAI_FORTRAN_OPENACC
        code = gencode_fortran_openacc(ainfo, buildtype, launchid, args, names, control)

    elseif ainfo.acceltype == JAI_FORTRAN_OMPTARGET
        code = gencode_fortran_omptarget(ainfo, buildtype, launchid, args, names, control)

    else
        error(string(ainfo.acceltype) * " is not supported for allocation.")

    end

    code
end


"""
    @jenter accel directs...

Initialize accelerator task

A more detailed explanation can go here, although I guess it is not needed in this case

# Arguments
* `accel`: a literal string to identify jai accelerator task
* `directs`: one or more jenterdata clauses

# Notes
* Notes can go here

# Examples
```julia
julia> @jenterdata myaccel framework(fortran_openacc)
```
"""
macro jenterdata(directs...)

    tmp = Expr(:block)

    allocs = Expr[]
    nonallocs = Expr[]
    alloccount = 1
    updatetocount = 1
    allocnames = String[]
    updatenames = String[]
    control = String[]

    lendir = length(directs)

    if lendir == 0
        stracc = "jai_accel_default"

    elseif directs[1] isa Symbol
        stracc = string(directs[1])
        directs = directs[2:end]

    else
        stracc = "jai_accel_default"

    end

    for direct in directs

        if direct isa Symbol
            push!(control, string(direct))

        elseif direct.args[1] == :allocate

            for idx in range(2, stop=length(direct.args))
                push!(allocnames, String(direct.args[idx]))
                direct.args[idx] = esc(direct.args[idx])
            end

            insert!(direct.args, 2, stracc)

            insert!(direct.args, 3, JAI_ALLOCATE)
            insert!(direct.args, 4, alloccount)
            alloccount += 1
            push!(allocs, direct)

        elseif direct.args[1] == :updateto

            for idx in range(2, stop=length(direct.args))
                push!(updatenames, String(direct.args[idx]))
                direct.args[idx] = esc(direct.args[idx])
            end

            insert!(direct.args, 2, stracc)

            #for dvar in direct.args[3:end]
            #    push!(updatenames, String(dvar))
            #end
            insert!(direct.args, 3, JAI_UPDATETO)
            insert!(direct.args, 4, updatetocount)
            updatetocount += 1
            push!(nonallocs, direct)

        elseif direct.args[1] in (:async,)
            push!(control, string(direct.args[1]))

        else
            error(string(direct.args[1]) * " is not supported.")

        end

    end

    for direct in (allocs..., nonallocs...)

        if direct.args[1] == :updateto
            kwupdatenames = Expr(:kw, :names, Expr(:tuple, updatenames...))
            push!(direct.args, kwupdatenames)

        elseif direct.args[1] == :allocate

            kwallocnames = Expr(:kw, :names, Expr(:tuple, allocnames...))
            push!(direct.args, kwallocnames)

        end

        kwcontrol = Expr(:kw, :control, control)
        push!(direct.args, kwcontrol)

        kwline = Expr(:kw, :_lineno_, __source__.line)
        push!(direct.args, kwline)

        kwfile = Expr(:kw, :_filepath_, string(__source__.file))
        push!(direct.args, kwfile)

        direct.args[1] = :jai_directive

        push!(tmp.args, direct)
    end

    #dump(tmp)
    return(tmp)
end

macro jexitdata(directs...)

    tmp = Expr(:block)
    deallocs = Expr[]
    nondeallocs = Expr[]
    updatefromcount = 1
    dealloccount = 1
    deallocnames = String[]
    updatenames = String[]
    control = String[]

    lendir = length(directs)

    if lendir == 0
        accstr = "jai_accel_default"

    elseif directs[1] isa Symbol
        accstr = string(directs[1])
        directs = directs[2:end]

    else
        accstr = "jai_accel_default"

    end

    for direct in directs

        if direct isa Symbol
            push!(control, string(direct))

        elseif direct.args[1] == :updatefrom

            for idx in range(2, stop=length(direct.args))
                push!(updatenames, String(direct.args[idx]))
                direct.args[idx] = esc(direct.args[idx])
            end

            insert!(direct.args, 2, accstr)

#            for uvar in direct.args[3:end]
#                push!(updatenames, String(uvar))
#            end
            insert!(direct.args, 3, JAI_UPDATEFROM)
            insert!(direct.args, 4, updatefromcount)
            updatefromcount += 1
            push!(nondeallocs, direct)

        elseif direct.args[1] == :deallocate

            for idx in range(2, stop=length(direct.args))
                push!(deallocnames, String(direct.args[idx]))
                direct.args[idx] = esc(direct.args[idx])
            end

            insert!(direct.args, 2, accstr)
#
#            for dvar in direct.args[3:end]
#                push!(deallocnames, String(dvar))
#            end
            insert!(direct.args, 3, JAI_DEALLOCATE)
            insert!(direct.args, 4, dealloccount)
            dealloccount += 1
            push!(deallocs, direct)

        elseif direct.args[1] in (:async,)
            push!(control, string(direct.args[1]))

        else
            error(string(direct.args[1]) * " is not supported.")

        end
    end

    for direct in (nondeallocs..., deallocs...)

        if direct.args[1] == :updatefrom
            kwupdatenames = Expr(:kw, :names, Expr(:tuple, updatenames...))
            #kwupdatenames = Expr(:kw, :names, :($((updatenames...),)))
            push!(direct.args, kwupdatenames)

        elseif direct.args[1] == :deallocate

            kwdeallocnames = Expr(:kw, :names, Expr(:tuple, deallocnames...))
            #kwdeallocnames = Expr(:kw, :names, :($((deallocnames...),)))
            push!(direct.args, kwdeallocnames)

        end

        kwcontrol = Expr(:kw, :control, control)
        push!(direct.args, kwcontrol)

        kwline = Expr(:kw, :_lineno_, __source__.line)
        push!(direct.args, kwline)

        kwfile = Expr(:kw, :_filepath_, string(__source__.file))
        push!(direct.args, kwfile)

        direct.args[1] = :jai_directive

        push!(tmp.args, direct)
    end

    #dump(tmp)
    return(tmp)
end

macro jkernel(clauses...)

    tmp = Expr(:call)
    push!(tmp.args, :jai_kernel_init)

    lenclauses = length(clauses)

    if lenclauses < 2
        error("@jkernel macro requires at least two clauses.")

    elseif lenclauses == 2
        push!(tmp.args, String(clauses[1]))
        push!(tmp.args, esc(clauses[2]))
        push!(tmp.args, "jai_accel_default")

    else
        push!(tmp.args, String(clauses[1]))
        push!(tmp.args, esc(clauses[2]))

        if clauses[3] isa Symbol
            push!(tmp.args, String(clauses[3]))
            start = 4
        else
            push!(tmp.args, "jai_accel_default")
            start = 3
        end

        for clause in clauses[start:end]
        end
    end

    kwline = Expr(:kw, :_lineno_, __source__.line)
    push!(tmp.args, kwline)

    kwfile = Expr(:kw, :_filepath_, string(__source__.file))
    push!(tmp.args, kwfile)

    #dump(tmp)
    return(tmp)

end


macro jlaunch(largs...)

    tmp = Expr(:call)
    push!(tmp.args, :launch_kernel)
    innames = String[]
    outnames = String[]

    flag = true

    for larg in largs
        if larg isa Symbol
            if flag
                push!(tmp.args, string(larg))
                flag = false
            else
                push!(innames, String(larg))
                push!(tmp.args, esc(larg))
            end

        elseif larg.head == :parameters
            for param in larg.args
                if param.head  == :kw && param.args[1] == :output
                    for ovar in param.args[2].args
                        push!(outnames, String(ovar))
                    end
                end
            end
            push!(tmp.args, esc(larg))
        end
    end

    kwinnames = Expr(:kw, :innames, Expr(:tuple, innames...))
    push!(tmp.args, kwinnames)

    kwoutnames = Expr(:kw, :outnames, Expr(:tuple, outnames...))
    push!(tmp.args, kwoutnames)

    kwline = Expr(:kw, :_lineno_, __source__.line)
    push!(tmp.args, kwline)

    kwfile = Expr(:kw, :_filepath_, string(__source__.file))
    push!(tmp.args, kwfile)

    #dump(tmp)
    return(tmp)

end

macro jaccel(clauses...)

    eval(quote import AccelInterfaces.jai_directive   end)

    tmp = Expr(:block)

    init = Expr(:call)
    push!(init.args, :jai_accel_init)

    if length(clauses) == 0
        push!(init.args, "jai_accel_default")

    elseif clauses[1] isa Symbol
        push!(init.args, string(clauses[1]))
        clauses = clauses[2:end]

    else
        push!(init.args, "jai_accel_default")

    end


    for clause in clauses

        if clause.args[1] == :constant
            const_vars = clause.args[2:end]
            const_names = [String(n) for n in const_vars]
            const_vars = (esc(c) for c in const_vars)

            push!(init.args, Expr(:kw, :const_vars, Expr(:tuple, const_vars...))) 
            push!(init.args, Expr(:kw, :const_names, Expr(:tuple, const_names...)))

        elseif clause.args[1] == :device
            device = (esc(d) for d in clause.args[2:end])

            push!(init.args, Expr(:kw, :device, Expr(:tuple, device...))) 

        elseif clause.args[1] == :framework

            items = Vector{Expr}()

            for item in clause.args[2:end]

                if item isa Symbol
                    push!(items, Expr(:tuple, String(item), :nothing))

                elseif item.head == :kw

                    if item.args[2] isa Symbol || item.args[2] isa String
                        push!(items, Expr(:tuple, string(item.args[1]), esc(item.args[2])))

                    elseif item.args[2].head == :tuple
                        subitems = []
                        for subargs in item.args[2].args
                            push!(subitems, Expr(:tuple, string(subargs.args[1]), esc(subargs.args[2])))
                        end
                        push!(items, Expr(:tuple, string(item.args[1]), Expr(:tuple, subitems...)))

                    else
                        error("Wrong framework syntax")

                    end

                else
                    error("Wrong framework syntax")
                end
            end

            #dump(items)

            #framework = (f for f in items)

            #push!(init.args, Expr(:kw, :framework, Expr(:tuple, framework...))) 
            push!(init.args, Expr(:kw, :framework, Expr(:tuple, items...))) 

        elseif clause.args[1] == :compile
            compile = (esc(c) for c in clause.args[2:end])

            push!(init.args, Expr(:kw, :compile, Expr(:tuple, compile...))) 

        elseif clause.args[1] == :set

            for kwarg in clause.args[2:end]
                if kwarg.head == :kw
                    push!(init.args, Expr(:kw, kwarg.args[1], esc(kwarg.args[2]))) 
                else
                    error("set clause allows keyword argument only.")
                end
            end

        else
            error(string(clause.args[1]) * " is not supported.")

        end
    end

    kwline = Expr(:kw, :_lineno_, __source__.line)
    push!(init.args, kwline)

    kwfile = Expr(:kw, :_filepath_, string(__source__.file))
    push!(init.args, kwfile)

    push!(tmp.args, init)

    #dump(tmp)
    return(tmp)
end


macro jdecel(clauses...)

    tmp = Expr(:block)

    fini = Expr(:call)
    push!(fini.args, :jai_accel_fini)

    if length(clauses) == 0
        push!(fini.args, "jai_accel_default")

    elseif clauses[1] isa Symbol
        push!(fini.args, string(clauses[1]))
        clauses = clauses[2:end]

    else
        push!(fini.args, "jai_accel_default")

    end

    for clause in clauses
    end

    kwline = Expr(:kw, :_lineno_, __source__.line)
    push!(fini.args, kwline)

    kwfile = Expr(:kw, :_filepath_, string(__source__.file))
    push!(fini.args, kwfile)

    push!(tmp.args, fini)

    #dump(tmp)
    return(tmp)
end

macro jwait(clauses...)

    tmp = Expr(:block)

    expr = Expr(:call)
    push!(expr.args, :jai_accel_wait)

    if length(clauses) == 0
        push!(expr.args, "jai_accel_default")

    elseif clauses[1] isa Symbol
        push!(expr.args, string(clauses[1]))
        clauses = clauses[2:end]

    else
        push!(expr.args, "jai_accel_default")

    end

    kwline = Expr(:kw, :_lineno_, __source__.line)
    push!(expr.args, kwline)

    kwfile = Expr(:kw, :_filepath_, string(__source__.file))
    push!(expr.args, kwfile)

    push!(tmp.args, expr)

    return(tmp)
end


end
