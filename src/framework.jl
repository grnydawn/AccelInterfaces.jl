# framework.jl: implement common functions for framework interfaces

import InteractiveUtils.subtypes
import Libdl: dlopen, RTLD_LAZY, RTLD_DEEPBIND, RTLD_GLOBAL, dlext, dlsym, dlclose

const JAI_FORTRAN                   = JAI_TYPE_FORTRAN()
const JAI_FORTRAN_OPENACC           = JAI_TYPE_FORTRAN_OPENACC()
const JAI_FORTRAN_OMPTARGET         = JAI_TYPE_FORTRAN_OMPTARGET()
const JAI_CPP                       = JAI_TYPE_CPP()
const JAI_CPP_OPENACC               = JAI_TYPE_CPP_OPENACC()
const JAI_CPP_OMPTARGET             = JAI_TYPE_CPP_OMPTARGET()
const JAI_CUDA                      = JAI_TYPE_CUDA()
const JAI_HIP                       = JAI_TYPE_HIP()

const JAI_TYPE_FORTRAN_FRAMEWORKS   = Union{
        JAI_TYPE_FORTRAN_OMPTARGET,
        JAI_TYPE_FORTRAN_OPENACC,
        JAI_TYPE_FORTRAN
    }

const JAI_FORTRAN_FRAMEWORKS        = (
        JAI_FORTRAN_OMPTARGET,
        JAI_FORTRAN_OPENACC,
        JAI_FORTRAN
    )
  
const JAI_TYPE_CPP_FRAMEWORKS       = Union{
        JAI_TYPE_CUDA,
        JAI_TYPE_HIP,
        JAI_TYPE_CPP_OMPTARGET,
        JAI_TYPE_CPP_OPENACC,
        JAI_TYPE_CPP
    }

const JAI_CPP_FRAMEWORKS            = (
        JAI_CUDA,
        JAI_HIP,
        JAI_CPP_OMPTARGET,
        JAI_CPP_OPENACC,
        JAI_CPP
    )

# Jai supported frameworks with priority
const JAI_SUPPORTED_FRAMEWORKS      = (
        JAI_CUDA,
        JAI_HIP,
        JAI_FORTRAN_OMPTARGET,
        JAI_FORTRAN_OPENACC,
        JAI_CPP_OMPTARGET,
        JAI_CPP_OPENACC,
        JAI_FORTRAN,
        JAI_CPP
    )

const JAI_MAP_SYMBOL_FRAMEWORK = OrderedDict(
        :cuda               => JAI_CUDA,
        :hip                => JAI_HIP,
        :fortran_omptarget  => JAI_FORTRAN_OMPTARGET,
        :fortran_openacc    => JAI_FORTRAN_OPENACC,
        :cpp_omptarget      => JAI_CPP_OMPTARGET,
        :cpp_openacc        => JAI_CPP_OPENACC,
        :fortran            => JAI_FORTRAN,
        :cpp                => JAI_CPP
    )

const JAI_MAP_FRAMEWORK_STRING = OrderedDict(
        JAI_CUDA                => "cuda",
        JAI_HIP                 => "hip",
        JAI_FORTRAN_OMPTARGET   => "fortran_omptarget",
        JAI_FORTRAN_OPENACC     => "fortran_openacc",
        JAI_CPP_OMPTARGET       => "cpp_omptarget",
        JAI_CPP_OPENACC         => "cpp_openacc",
        JAI_FORTRAN             => "fortran",
        JAI_CPP                 => "cpp"
    )

const JAI_MAP_API_FUNCNAME = Dict{JAI_TYPE_API, String}(
        JAI_ACCEL       => "accel",
        JAI_ALLOCATE    => "alloc",
        JAI_DEALLOCATE  => "dealloc",
        JAI_UPDATETO    => "updateto",
        JAI_UPDATEFROM  => "updatefrom",
        JAI_LAUNCH      => "kernel",
        JAI_WAIT        => "wait"
    )

FORTRAN_TEMPLATE_MODULE = """
MODULE mod_{prefix}{suffix}
USE, INTRINSIC :: ISO_C_BINDING

{specpart}

CONTAINS

{subppart}

END MODULE
"""

FORTRAN_TEMPLATE_FUNCTION = """
INTEGER (C_INT64_T) FUNCTION {prefix}{suffix}({dummyargs}) BIND(C, name="{prefix}{suffix}")
USE, INTRINSIC :: ISO_C_BINDING

{specpart}

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

{execpart}

{prefix}{suffix} = JAI_ERRORCODE

END FUNCTION
"""

const _ccall_cache = Dict{Int64, Function}()
const _dummyargs = Tuple("a[$i]" for i in range(1, stop=200))
const _ccs = ("(f,a) -> ccall(f, Int64, (", ",), " , ")")

function jai_ccall(dtypestr::String, libfunc::Ptr{Nothing}, args::JAI_TYPE_ARGS) :: Int64

    funcstr = _ccs[1] * dtypestr * _ccs[2] * join(_dummyargs[1:length(args)], ",") * _ccs[3]
    fid = read(IOBuffer(sha1(funcstr)[1:8]), Int64)

    if ! haskey(_ccall_cache, fid)
        _ccall_cache[fid] = eval(Meta.parse(funcstr))
    end

    actual_args = map(x -> x[1], args)
    Base.invokelatest(_ccall_cache[fid], libfunc, actual_args)
end


function argsdtypes(
        frame   ::JAI_TYPE_FRAMEWORK,
        args    ::JAI_TYPE_ARGS
    ) :: Vector{DataType}

    local N = length(args)

    dtypes = Vector{DataType}(undef, N)

    for (i, (arg, name, inout, addr, shape, offsets)) in enumerate(args)

        if typeof(arg) <: AbstractArray
            dtype = Ptr{typeof(arg)}

        elseif frame in JAI_CPP_FRAMEWORKS
            dtype = typeof(arg)

        elseif frame in JAI_FORTRAN_FRAMEWORKS
            dtype = Ref{typeof(arg)}

        end

        dtypes[i] = dtype

    end

    dtypes
end


function compile_code(
        frame   ::JAI_TYPE_FRAMEWORK,
        code    ::String,
        compile ::String,
        srcname ::String,
        outname ::String,
        workdir ::String
    ) ::String

    curdir = pwd()

    try
        #workdir = get_config("workdir")

        cd(workdir)

        open(srcname, "w") do io
            write(io, code)
        end

        output = read(run(`$(split(compile)) -o $outname $srcname`), String)

        isfile(outname) || throw(JAI_ERROR_COMPILE_NOSHAREDLIB(compile, output))

        return joinpath(workdir, outname)

    catch e
        rethrow(e)

    finally
        cd(curdir)
    end
end


function load_sharedlib(libpath::String)
    return dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
end


function invoke_sharedfunc(
        frame   ::JAI_TYPE_FRAMEWORK,
        slib    ::Ptr{Nothing},
        fname   ::String,
        args    ::JAI_TYPE_ARGS
    )

    dtypes = argsdtypes(frame, args)
    dtypestr = join([string(t) for t in dtypes], ",")

    libfunc = dlsym(slib, Symbol(fname))

    check_retval(jai_ccall(dtypestr, libfunc, args))

end

function generate_code(
        frame       ::JAI_TYPE_FORTRAN_FRAMEWORKS,
        apitype     ::JAI_TYPE_API,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, String} where N
    ) :: String

    suffix   = JAI_MAP_API_FUNCNAME[apitype]
    specpart = code_module_specpart(frame, apitype, prefix, args, data)
    subppart = code_module_subppart(frame, apitype, prefix, args, data)

    return jaifmt(FORTRAN_TEMPLATE_MODULE, prefix=prefix,
                  suffix=suffix, specpart=specpart, subppart=subppart)
end


include("fortran.jl")
include("compiler.jl")
include("machine.jl")

function generate_sharedlib(
        frame       ::JAI_TYPE_FORTRAN,
        apitype     ::JAI_TYPE_API,
        prefix      ::String,
        compile     ::String,
        workdir     ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::Vararg{String, N} where N
    ) :: Ptr{Nothing}

    code = generate_code(frame, apitype, prefix, args, data)

    srcname = prefix * JAI_MAP_API_FUNCNAME[apitype] * ".F90"
    outname = prefix * JAI_MAP_API_FUNCNAME[apitype] * "." * dlext

    slibpath = compile_code(frame, code, compile, srcname, outname, workdir)

    slib = load_sharedlib(slibpath)

    # init device
    if apitype == JAI_ACCEL
        invoke_sharedfunc(frame, slib, prefix * "device_init", args)
    end

    return slib

end


function get_framework(
        frametype   ::JAI_TYPE_FRAMEWORK,
        fconfig     ::JAI_TYPE_CONFIG_VALUE,
        compiler    ::JAI_TYPE_CONFIG,
        workdir     ::String
    ) :: Union{JAI_TYPE_CONTEXT_FRAMEWORK, Nothing}
 
    if fconfig isa String
        compile = fconfig
    else
        try
            compile = fconfig["compile"]
        catch
            compile = nothing
        end
    end

    frameworks = JAI["frameworks"]

    if frametype in keys(frameworks)
        frames = frameworks[frametype]

        if length(frames) > 0
            if compile isa String
                cid = generate_jid(compile)
                if cid in keys(frames)
                    return frames[cid]
                end
            elseif compile == nothing
                return first(frames)[2]
            else
                error("Wrong compile type: " * string(typeof(compile)))
            end
        end
    end
   
    if compile isa String
        compiles = [compile]
    else
        compiles = get_compiles(frametype, compiler)
    end

    args = JAI_TYPE_ARGS()
    push!(args, pack_arg(fill(Int64(-1), 1)))

    for compile in compiles

        cid     = generate_jid(compile)
        prefix  = generate_prefix(JAI_MAP_FRAMEWORK_STRING[frametype], cid)
        slib    = generate_sharedlib(frametype, JAI_ACCEL, prefix, compile, workdir, args)

        if slib isa Ptr{Nothing}
            if frametype in keys(frameworks)
                frames = frameworks[frametype]
            else
                frames = OrderedDict{UInt32, JAI_TYPE_CONTEXT_FRAMEWORK}()
                frameworks[frametype] = frames
            end

            if cid in keys(frames)
                error("framework is already exist: " * string(cid))
            else
                frames[cid] = JAI_TYPE_CONTEXT_FRAMEWORK(frametype, slib, compile, prefix)
                return frames[cid]
            end
        end
    end

    return nothing
end

function select_framework(
        userframe   ::Union{JAI_TYPE_CONFIG, Nothing},
        compiler    ::Union{JAI_TYPE_CONFIG, Nothing},
        workdir     ::String
    ) :: Union{JAI_TYPE_CONTEXT_FRAMEWORK, Nothing}

    if userframe == nothing
        userframe = JAI_TYPE_CONFIG()
    end

    if compiler == nothing
        compiler = JAI_TYPE_CONFIG()
    end

    if length(userframe) == 0
        for frame in JAI_SUPPORTED_FRAMEWORKS
            userframe[frame] = nothing
        end
    end

    for (frame, config) in userframe
        framework = get_framework(frame, config, compiler, workdir)
        if framework isa JAI_TYPE_CONTEXT_FRAMEWORK
            return framework
        end
    end

    throw(JAI_ERROR_NOAVAILABLE_FRAMEWORK())
end



function select_framework(
        ctx_accel   ::JAI_TYPE_CONTEXT_ACCEL,
        userframe   ::Union{JAI_TYPE_CONFIG, Nothing},
        compiler    ::Union{JAI_TYPE_CONFIG, Nothing},
        workdir     ::String
    ) :: Union{JAI_TYPE_CONTEXT_FRAMEWORK, Nothing}

    if userframe == nothing
        userframe = JAI_TYPE_CONFIG()
    end

    if length(userframe) == 0
        userframe[ctx_accel.framework.type] = nothing
    end

    return select_framework(userframe, compiler, workdir)
end
