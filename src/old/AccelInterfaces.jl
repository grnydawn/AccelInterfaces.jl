module AccelInterfaces

using Serialization

import Pkg.TOML

import Libdl.dlopen,
       Libdl.RTLD_LAZY,
       Libdl.RTLD_DEEPBIND,
       Libdl.RTLD_GLOBAL,
       Libdl.dlext,
       Libdl.dlsym,
       Libdl.dlclose

import SHA.sha1
import Dates.now,
       Dates.Millisecond

import OffsetArrays.OffsetArray,
       OffsetArrays.OffsetVector

import Pidfile.mkpidlock


# TODO: simplified user interface
# [myaccel1] = @jaccel
# [mykernel1] = @jkernel kernel_text
# retval = @jlaunch([mykernel1,] x, y; output=(z,))

export JAI_VERSION, @jenterdata, @jexitdata, @jlaunch, @jaccel, @jkernel, @jdecel, @jwait

        
const JAI_VERSION = TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"]
const TIMEOUT = 10
const _IDLEN = 3

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
        JAI_CPP_HIP
        JAI_CPP_CUDA
        JAI_ANYACCEL
        JAI_HEADER
end

const ACCEL_SYMBOLS = (:fortran, :fortran_openacc, :fortran_omptarget,
        :cpp, :cpp_openacc, :cpp_omptarget, :cuda, :hip)

const _accelmap = Dict{String, AccelType}(
    "fortran" => JAI_FORTRAN,
    "fortran_openacc" => JAI_FORTRAN_OPENACC,
    "fortran_omptarget" => JAI_FORTRAN_OMPTARGET,
    "cpp" => JAI_CPP,
    "cpp_openacc" => JAI_CPP_OPENACC,
    "cpp_omptarget" => JAI_CPP_OMPTARGET,
    "hip" => JAI_CPP_HIP,
    "cuda" => JAI_CPP_CUDA,
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


const JaiConstType = Union{Number, String, NTuple{N, T}, AbstractArray{T, N}} where {N, T<:Number}
const JaiDataType = JaiConstType

function select_data_framework(frameworks::NTuple{N, Tuple{String, Union{NTuple{M, Tuple{String,
                    Union{String, Nothing}}}, String, Nothing}}} where {N, M}) :: String	

	framework = ""

	for (name, config) in frameworks

		if framework == ""
			framework = name

		elseif name in ("fortran", "cpp")

		elseif name == "fortran_omptarget"
			framework = name

		elseif name == "cpp_omptarget"
			if !(framework in ("fortran_omptarget",))
				framework = name
			end

		elseif name == "fortran_openacc"
			if !(framework in ("fortran_omptarget, cpp_omptarget"))
				framework = name
			end

		elseif name == "cpp_openacc"
			if !(framework in ("fortran_omptarget, cpp_omptarget",
							 "fortran_openacc"))
				framework = name
			end

		elseif name == "cuda"
			if !(framework in ("fortran_omptarget, cpp_omptarget",
							 "fortran_openacc", "cpp_openacc"))
				framework = name
			end


		elseif name == "hip"
			if !(framework in ("fortran_omptarget, cpp_omptarget",
							 "fortran_openacc", "cpp_openacc", "cuda"))
				framework = name
			end
		end
	end

	return framework
end

struct AccelInfo

    accelid::String
    ismaster::Bool
    device_num::Int64
    const_vars::NTuple{N,JaiConstType} where {N}
    const_names::NTuple{N, String} where {N}
    data_framework::String
    compile_frameworks::Dict{String, String}
    sharedlibs::Dict{String, Ptr{Nothing}}
    workdir::Union{String, Nothing}
    debugdir::Union{String, Nothing}
    ccallcache::Dict{Tuple{BuildType, Int64, Int64, String}, Tuple{Ptr{Nothing}, String}}

    function AccelInfo(;master::Bool=true,
            const_vars::NTuple{N,JaiConstType} where {N}=(),
            const_names::NTuple{N, String} where {N}=(),
            framework::NTuple{N, Tuple{String, Union{NTuple{M, Tuple{String,
                    Union{String, Nothing}}}, String, Nothing}}} where {N, M}=nothing,
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

		# determine which framework will do memory management
        data_framework = select_data_framework(framework)
        compile_frameworks = Dict{String, String}()

        dlib = nothing
        acceltype = nothing   
        sharedlibs = nothing   
        compile = ""   

        # TODO: support multiple framework arguments
        for (frameworkname, frameconfig) in framework
            acceltype = _accelmap[frameworkname]

			compile = ""

            if frameconfig isa Nothing
                if startswith(frameworkname, "fortran")
                    compile = get(ENV, "JAI_FC", get(ENV, "FC", "")) * " " *
                                get(ENV, "JAI_FFLAGS", get(ENV, "FFLAGS", ""))

                elseif startswith(frameworkname, "cpp")
                    compile = get(ENV, "JAI_CXX", get(ENV, "CXX", "")) * " " *
                                get(ENV, "JAI_CXXFLAGS", get(ENV, "CXXFLAGS", ""))

                elseif frameworkname == "cuda"
                    compile = get(ENV, "JAI_NVCC", "") * " " *
                                get(ENV, "JAI_NVCCFLAGS", "")

                elseif frameworkname == "hip"
                    compile = get(ENV, "JAI_HIPCC", "") * " " *
                                get(ENV, "JAI_HIPCCFLAGS", "")

                else
                    error(string(frameworkname * " is not supported."))
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

            if strip(compile) == ""
                error("No compile information is available.")
            end

            io = IOBuffer()
            ser = serialize(io, (accelid, acceltype, compile))
            accelid = bytes2hex(sha1(String(take!(io)))[1:4])

            libpath = joinpath(workdir, "LIB" * accelid * "." * dlext)

            try
                build_accel!(workdir, debugdir, acceltype, compile, accelid, libpath)

				if frameworkname == data_framework
					dlib = dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
				 
					sharedlibs = Dict{String, Ptr{Nothing}}()
					sharedlibs[accelid] = dlib

				end

                compile_frameworks[frameworkname] = compile

            catch e
                rethrow(e)

            end

        end

        buf = fill(-1, 1)

        func = dlsym(dlib, Symbol("jai_get_num_devices_" * accelid[1:_IDLEN]))
        ccall(func, Int64, (Ptr{Vector{Int64}},), buf)
        if buf[1] < 1
            error("The number of devices is less than 1.")
        end

        if length(device) == 1
            buf[1] = device[1]
            func = dlsym(dlib, Symbol("jai_set_device_num_" * accelid[1:_IDLEN]))
            ccall(func, Int64, (Ptr{Vector{Int64}},), buf)
            device_num = device[1]

        else
            buf[1] = -1
            func = dlsym(dlib, Symbol("jai_get_device_num_" * accelid[1:_IDLEN]))
            ccall(func, Int64, (Ptr{Vector{Int64}},), buf)
            device_num = buf[1]

        end

        func = dlsym(dlib, Symbol("jai_device_init_" * accelid[1:_IDLEN]))
        ccall(func, Int64, (Ptr{Vector{Int64}},), buf)
              
        new(accelid, master, device_num, const_vars,
            const_names, data_framework, compile_frameworks, sharedlibs,
            workdir, debugdir,
            Dict{Tuple{BuildType, Int64, Int64, String}, Ptr{Nothing}}())

    end
end

const _accelcache = Dict{String, AccelInfo}()
const _ccall_cache = Dict()
const _dummyargs = ["a[$i]" for i in range(1, stop=200)]
const _ccs = ("(f,a) -> ccall(f, Int64, (", ",), " , ")")

function jai_ccall(dtypestr::String, libfunc::Ptr{Nothing}, args) :: Int64

    funcstr = _ccs[1] * dtypestr * _ccs[2] * join(_dummyargs[1:length(args)], ",") * _ccs[3]
    fid = read(IOBuffer(sha1(funcstr)[1:8]), Int64)

    if ! haskey(_ccall_cache, fid)
        _ccall_cache[fid] = eval(Meta.parse(funcstr))
    end

    Base.invokelatest(_ccall_cache[fid], libfunc, args)
end

function jai_accel_init(name::String; master::Bool=true,
            const_vars::NTuple{N,JaiConstType} where {N}=(),
            const_names::NTuple{N, String} where {N}=(),
            framework::NTuple{N, Tuple{String, Union{NTuple{M, Tuple{String,
                        Union{String, Nothing}}}, String, Nothing}}} where {N, M}=nothing,
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

    buf = fill(-1, 1)
    accel = _accelcache[name]
    dlib = accel.sharedlibs[accel.accelid]

    func = dlsym(dlib, Symbol("jai_device_fini_" * accel.accelid[1:_IDLEN]))
    ccall(func, Int64, (Ptr{Vector{Int64}},), buf)
    dlclose(dlib)

    delete!(_accelcache, name)
end

function jai_accel_wait(name::String;
            _lineno_::Union{Int64, Nothing}=nothing,
            _filepath_::Union{String, Nothing}=nothing)

    acceltype = get(_accelcache, name, get(_kernelcache, name, nothing))

    if accel.acceltype in (JAI_FORTRAN, JAI_CPP)
        return 0::Int64
    end

    # load shared lib
    dlib = accel.sharedlibs[accel.accelid]
    local func = dlsym(dlib, Symbol("jai_wait_" * accel.accelid[1:_IDLEN]))

    return ccall(func, Int64, ())

end

# NOTE: keep the order of the following includes
include("./kernel.jl")
include("./fortran.jl")
include("./fortran_openacc.jl")
include("./fortran_omptarget.jl")
include("./cpp.jl")
include("./cuda.jl")
include("./hip.jl")

function timeout(libpath::String, duration::Real; waittoexist::Bool=true) :: Nothing

    local tstart = now()

    while true
        local check = waittoexist ? ispath(libpath) : ~ispath(libpath)

        if check
            break

        elseif ((now() - tstart)/ Millisecond(1000)) > duration
            error("Timeout: " * libpath)

        else
            sleep(0.1)
        end
    end
end

function jai_directive(
            accelname::String,
            buildtype::BuildType,
            buildtypecount::Int64,
            data::Vararg{JaiDataType, N} where {N};
            names::NTuple{N, String} where {N}=(),
            control::Vector{String},
            _lineno_::Union{Int64, Nothing}=nothing,
            _filepath_::Union{String, Nothing}=nothing) :: Int64

    accel = _accelcache[accelname]

    if accel.acceltype in (JAI_FORTRAN, JAI_CPP)
        return 0::Int64
    end

    args = (accel.device_num, data...)
    names = ("jai_arg_device_num", names...)
    dtypes, sizes = argsdtypes(accel, args...)

    # TODO: add file modified date
    cachekey = (buildtype, buildtypecount, _lineno_, _filepath_)

    if _lineno_ isa Int64 && _filepath_ isa String
        if haskey(accel.ccallcache, cachekey)
            func, dtypestr = accel.ccallcache[cachekey]
            return jai_ccall(dtypestr, func, args)
        end
    end

    dtypestr = join([string(t) for t in dtypes], ", ")

    io = IOBuffer()
    ser = serialize(io, (buildtype, accel.accelid, [typeof(d) for d in args]))
    launchid = bytes2hex(sha1(String(take!(io)))[1:4])
    shortid = launchid[1:_IDLEN]

    local libpath = joinpath(accel.workdir, "LIB$(launchid)." * dlext)

    # load shared lib
    if haskey(accel.sharedlibs, launchid)
        dlib = accel.sharedlibs[launchid]

    else
        build_directive!(accel, buildtype, launchid, libpath, args, names, control)
        dlib = dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        accel.sharedlibs[launchid] = dlib
    end

    if buildtype == JAI_ALLOCATE
        local func = dlsym(dlib, Symbol("jai_allocate_" * shortid))

    elseif buildtype == JAI_UPDATETO
        local func = dlsym(dlib, Symbol("jai_updateto_" * shortid))

    elseif buildtype == JAI_UPDATEFROM
        local func = dlsym(dlib, Symbol("jai_updatefrom_" * shortid))

    elseif buildtype == JAI_DEALLOCATE
        local func = dlsym(dlib, Symbol("jai_deallocate_" * shortid))

    else
        error(string(buildtype) * " is not supported.")

    end

    if _lineno_ isa Int64 && _filepath_ isa String
        accel.ccallcache[cachekey] = (func, dtypestr)
    end

    return jai_ccall(dtypestr, func, args)
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

        elseif ainfo.acceltype in (JAI_CPP, JAI_CPP_OPENACC, JAI_CPP_CUDA,
                    JAI_CPP_HIP)
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
function launch_kernel(
            aname::String,
            kname::String;
            innames::NTuple{N, String} where {N}=(),
            outnames::NTuple{N, String} where {N}=(),
            input::NTuple{N,JaiDataType} where {N}=(),
            output::NTuple{N,JaiDataType} where {N}=(),
            cpp::Dict{String}=Dict{String, JaiDataType}(),
            cuda::Dict{String}=Dict{String, JaiDataType}(),
            hip::Dict{String}=Dict{String,JaiDataType}(),
            fortran::Dict{String}=Dict{String, JaiDataType}(),
            fortran_openacc::Dict{String}=Dict{String,JaiDataType}(),
            fortran_omptarget::Dict{String}=Dict{String,JaiDataType}(),
            _lineno_::Union{Int64, Nothing}=nothing,
            _filepath_::Union{String, Nothing}=nothing) :: Int64

    launchopts = Dict(
        "cpp" => cpp,
        "cuda" => cuda,
        "hip" => hip,
        "fortran" => fortran,
        "fortran_openacc" => fortran_openacc,
        "fortran_omptarget" => fortran_omptarget
    )

    kinfo = _kernelcache[aname * kname]

    invars = (kinfo.accel.device_num, input...)
    innames = ("jai_arg_device_num", innames...)

    args, names = merge_args(invars, output, innames, outnames)
    dtypes, sizes = argsdtypes(kinfo.accel, args...)

    # TODO: add launchopts into cachekey
    cachekey = (JAI_LAUNCH, 0::Int64, _lineno_, _filepath_)


    if _lineno_ isa Int64 && _filepath_ isa String
        if haskey(kinfo.accel.ccallcache, cachekey)
            func, dtypestr = kinfo.accel.ccallcache[cachekey]
            return jai_ccall(dtypestr, func, args)
        end
    end

    dtypestr = join([string(t) for t in dtypes], ", ")

    io = IOBuffer()
    ser = serialize(io, (JAI_LAUNCH, kinfo.kernelid, [typeof(a) for a in args]))
    launchid = bytes2hex(sha1(String(take!(io)))[1:4])

    libpath = joinpath(kinfo.accel.workdir, "LIB$(launchid)." * dlext)

    # load shared lib
    if haskey(kinfo.accel.sharedlibs, launchid)
        dlib = kinfo.accel.sharedlibs[launchid]

    else
        build_kernel!(kinfo, launchid, launchopts, libpath, invars, output, innames, outnames)
        dlib = dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        kinfo.accel.sharedlibs[launchid] = dlib
    end

    func = dlsym(dlib, Symbol("jai_launch_" * launchid[1:_IDLEN]))

    if _lineno_ isa Int64 && _filepath_ isa String
        kinfo.accel.ccallcache[cachekey] = (func, dtypestr)
    end

    ret = jai_ccall(dtypestr, func, args)
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

    elseif  acceltype == JAI_CPP_CUDA
        ext = ".cu"
        if compile == nothing

            compile = "nvcc --linker-options=\"-fPIC\" --shared -g"
        end

    elseif  acceltype == JAI_CPP_HIP
        ext = ".cpp"
        if compile == nothing

            compile = "hipcc -shared -fPIC -lamdhip64 -g"
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


function _genlibfile(outpath::String, srcfile::String, code::String,
    debugdir::Union{String, Nothing}, compile::String, pidfile::String)

    curdir = pwd()

    try
        outpath = abspath(outpath)

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

        compilelog = read(run(`$(split(compile)) -o $outfile $(srcfile)`), String)

        if ispath(outfile)
            cp(outfile, outpath, force=true)
        end

    catch e
        rethrow(e)

    finally
        cd(curdir)
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

        lock = nothing

        try
            lock = mkpidlock(pidfile, stale_age=3)

            if !ispath(outpath)
                code = generate_accel!(workdir, acceltype, compile, accelid)
                _genlibfile(outpath, srcfile, code, debugdir, compile, pidfile)
            end

        catch e
            rethrow(e)

        finally
            if lock != nothing
                close(lock)
            end
        end
    end

    timeout(outpath, TIMEOUT)

    outpath

end

# kernel build
function build_kernel!(kinfo::KernelInfo, launchid::String,
                launchopts::Dict{String},
                outpath::String,
                inargs::NTuple{N, JaiDataType} where {N},
                outargs::NTuple{M, JaiDataType} where {M},
                innames::NTuple{N, String} where {N},
                outnames::NTuple{M, String} where {M}) :: String

    # TODO: select acceltype for kernel
    # TODO: get compile and srcfile, ... accordingly

    srcfile, compile = setup_build(kinfo.acceltype, JAI_LAUNCH, launchid,
                                    kinfo.compile)

    srcpath = joinpath(kinfo.accel.workdir, srcfile)
    pidfile = outpath * ".pid"

    # generate shared lib
    if !ispath(outpath)

        lock = nothing

        try
            lock = mkpidlock(pidfile, stale_age=3)

            if !ispath(outpath)
                code = generate_kernel!(kinfo, launchid[1:_IDLEN], launchopts,
                                        inargs, outargs, innames, outnames)
                _genlibfile(outpath, srcfile, code, kinfo.accel.debugdir, compile, pidfile)
            end

        catch e
            rethrow(e)

        finally
            if lock != nothing
                close(lock)
            end
        end
    end

    timeout(outpath, TIMEOUT)

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

    # generate shared lib
    if !ispath(outpath)

        lock = nothing

        try
            lock = mkpidlock(pidfile, stale_age=3)

            if !ispath(outpath)
                code = generate_directive!(ainfo, buildtype, launchid[1:_IDLEN],
                                        args, names, control)
                _genlibfile(outpath, srcfile, code, ainfo.debugdir, compile, pidfile)
            end

        catch e
            rethrow(e)

        finally
            if lock != nothing
                close(lock)
            end
        end
    end

    timeout(outpath, TIMEOUT)

    outpath
end

# accel generate
function generate_accel!(workdir::String, acceltype::AccelType,
        compile::Union{String, Nothing}, accelid::String) :: String

    shortid = accelid[1:_IDLEN]

    if acceltype == JAI_FORTRAN
        code = gencode_fortran_accel(shortid)

    elseif acceltype == JAI_CPP
        code = gencode_cpp_accel(shortid)

    elseif acceltype == JAI_CPP_CUDA
        code = gencode_cpp_cuda_accel(shortid)

    elseif acceltype == JAI_CPP_HIP
        code = gencode_cpp_hip_accel(shortid)

    elseif acceltype == JAI_FORTRAN_OPENACC
        code = gencode_fortran_openacc_accel(shortid)

    elseif acceltype == JAI_FORTRAN_OMPTARGET
        code = gencode_fortran_omptarget_accel(shortid)

    else
        error(string(acceltype) * " is not supported yet.")
    end

    return code

end

# kernel generate
                #launchopts::Union{Dict{String, Dict{String, <:Any}}, Dict{String, Nothing}},
function generate_kernel!(kinfo::KernelInfo, launchid::String,
                launchopts::Dict{String},
                inargs::NTuple{N, JaiDataType} where {N},
                outargs::NTuple{M, JaiDataType} where {M},
                innames::NTuple{N, String} where {N},
                outnames::NTuple{M, String} where {M}) :: String

    body = kinfo.kerneldef.body

    if kinfo.acceltype == JAI_FORTRAN
        fortopts = launchopts["fortran"] != nothing ? launchopts["fortran"] : Dict{String, Any}()
        code = gencode_fortran_kernel(kinfo, launchid, fortopts, body,
                                inargs, outargs, innames, outnames)

    elseif kinfo.acceltype == JAI_CPP
        cppopts = launchopts["cpp"] != nothing ? launchopts["cpp"] : Dict{String, Any}()
        code = gencode_cpp_kernel(kinfo, launchid, cppopts, body,
                                inargs, outargs, innames, outnames)

    elseif kinfo.acceltype == JAI_CPP_CUDA
        cudaopts = launchopts["cuda"] != nothing ? launchopts["cuda"] : Dict{String, Any}()
        code = gencode_cpp_cuda_kernel(kinfo, launchid, cudaopts, body,
                                inargs, outargs, innames, outnames)

    elseif kinfo.acceltype == JAI_CPP_HIP
        hipopts = launchopts["hip"] != nothing ? launchopts["hip"] : Dict{String}()
        code = gencode_cpp_hip_kernel(kinfo, launchid, hipopts, body,
                                inargs, outargs, innames, outnames)

    elseif kinfo.acceltype == JAI_FORTRAN_OPENACC
        foaccopts = launchopts["fortran_openacc"] != nothing ? launchopts["fortran_openacc"] : Dict{String, Any}()
        code = gencode_fortran_kernel(kinfo, launchid, foaccopts, body,
                                inargs, outargs, innames, outnames)

    elseif kinfo.acceltype == JAI_FORTRAN_OMPTARGET
        fomptopts = launchopts["fortran_omptarget"] != nothing ? launchopts["fortran_omptarget"] : Dict{String}()
        code = gencode_fortran_kernel(kinfo, launchid, fomptopts, body,
                                inargs, outargs, innames, outnames)

    else
        error(string(kinfo.acceltype) * " is not supported yet.")
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
        code = gencode_fortran_openacc_directive(ainfo, buildtype, launchid,
                                                args, names, control)

    elseif ainfo.acceltype == JAI_FORTRAN_OMPTARGET
        code = gencode_fortran_omptarget_directive(ainfo, buildtype, launchid,
                                                args, names, control)

    elseif ainfo.acceltype == JAI_CPP_CUDA
        code = gencode_cpp_cuda_directive(ainfo, buildtype, launchid, args,
                                                names, control)

    elseif ainfo.acceltype == JAI_CPP_HIP
        code = gencode_cpp_hip_directive(ainfo, buildtype, launchid, args,
                                                names, control)

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
macro jenterdata(accname, directs...)

    tmp = Expr(:block)

    allocs = Expr[]
    nonallocs = Expr[]
    alloccount = 1
    updatetocount = 1
    allocnames = String[]
    updatenames = String[]
    control = String[]

    stracc = string(accname)

    for direct in directs

        if direct isa Symbol
            push!(control, string(direct))

        elseif direct.args[1] == :allocate

            for idx in range(2, stop=length(direct.args))
                push!(allocnames, string(direct.args[idx]))
                direct.args[idx] = esc(direct.args[idx])
            end

            insert!(direct.args, 2, stracc)
            insert!(direct.args, 3, JAI_ALLOCATE)
            insert!(direct.args, 4, alloccount)
            alloccount += 1
            push!(allocs, direct)

        elseif direct.args[1] == :updateto

            for idx in range(2, stop=length(direct.args))
                push!(updatenames, string(direct.args[idx]))
                direct.args[idx] = esc(direct.args[idx])
            end

            insert!(direct.args, 2, stracc)
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

macro jexitdata(accname, directs...)

    tmp = Expr(:block)
    deallocs = Expr[]
    nondeallocs = Expr[]
    updatefromcount = 1
    dealloccount = 1
    deallocnames = String[]
    updatenames = String[]
    control = String[]

    accstr = string(accname)

    for direct in directs

        if direct isa Symbol
            push!(control, string(direct))

        elseif direct.args[1] == :updatefrom

            for idx in range(2, stop=length(direct.args))
                push!(updatenames, String(direct.args[idx]))
                direct.args[idx] = esc(direct.args[idx])
            end

            insert!(direct.args, 2, accstr)
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
            push!(direct.args, kwupdatenames)

        elseif direct.args[1] == :deallocate

            kwdeallocnames = Expr(:kw, :names, Expr(:tuple, deallocnames...))
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
    #println(tmp)
    return(tmp)
end

macro jkernel(accname, knlname, knldef, clauses...)

    tmp = Expr(:call)
    push!(tmp.args, :jai_kernel_init)

    push!(tmp.args, string(accname))
    push!(tmp.args, string(knlname))
    push!(tmp.args, esc(knldef))

    for clause in clauses
        if clause.args[1] == :framework

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

            push!(tmp.args, Expr(:kw, :framework, Expr(:tuple, items...))) 
        end
    end

    kwline = Expr(:kw, :_lineno_, __source__.line)
    push!(tmp.args, kwline)

    kwfile = Expr(:kw, :_filepath_, string(__source__.file))
    push!(tmp.args, kwfile)

    #dump(tmp)
    #println(tmp)
    return(tmp)

end

macro jlaunch(accname, knlname, clauses...)

    tmp = Expr(:call)
    push!(tmp.args, :launch_kernel)
    input = :(())
    output = :(())
    innames = String[]
    outnames = String[]

    push!(tmp.args, string(accname))
    push!(tmp.args, string(knlname))

    for clause in clauses
        if clause.head == :call
            if clause.args[1] == :input
                for invar in clause.args[2:end]
                    push!(innames, String(invar))
                    push!(input.args, esc(invar))
                end
            elseif clause.args[1] == :output
                for outvar in clause.args[2:end]
                    push!(outnames, String(outvar))
                    push!(output.args, esc(outvar))
                end
            elseif clause.args[1] in ACCEL_SYMBOLS
                #kvs = :(Dict())
                kvs = :(Dict{String, Any}())
                for kv in clause.args[2:end]
                    push!(kvs.args, esc(kv))
                end
                push!(tmp.args, Expr(:kw, clause.args[1], kvs))
            end
        end
    end

    kwinput = Expr(:kw, :input, input)
    push!(tmp.args, kwinput)

    kwoutput = Expr(:kw, :output, output)
    push!(tmp.args, kwoutput)

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

macro jaccel(accname, clauses...)

    tmp = Expr(:block)

    init = Expr(:call)
    push!(init.args, :jai_accel_init)
    push!(init.args, string(accname))

    for clause in clauses

        if clause.args[1] == :constant
            const_vars = clause.args[2:end]
            const_names = [string(n) for n in const_vars]
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
                    push!(items, Expr(:tuple, string(item), :nothing))

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


macro jdecel(accname, clauses...)

    tmp = Expr(:block)

    fini = Expr(:call)
    push!(fini.args, :jai_accel_fini)
    push!(fini.args, string(accname))

    #for clause in clauses
    #end

    kwline = Expr(:kw, :_lineno_, __source__.line)
    push!(fini.args, kwline)

    kwfile = Expr(:kw, :_filepath_, string(__source__.file))
    push!(fini.args, kwfile)

    push!(tmp.args, fini)

    #dump(tmp)
    return(tmp)
end

macro jwait(accname, clauses...)

    tmp = Expr(:block)

    expr = Expr(:call)
    push!(expr.args, :jai_accel_wait)
    push!(expr.args, string(accname))

    kwline = Expr(:kw, :_lineno_, __source__.line)
    push!(expr.args, kwline)

    kwfile = Expr(:kw, :_filepath_, string(__source__.file))
    push!(expr.args, kwfile)

    push!(tmp.args, expr)

    return(tmp)
end


end