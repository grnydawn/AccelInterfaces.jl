# framework.jl: implement common functions for framework interfaces

import InteractiveUtils.subtypes
import Libdl: dlopen, RTLD_LAZY, RTLD_DEEPBIND, RTLD_GLOBAL, dlext, dlsym, dlclose
import IOCapture

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

const JAI_ACCEL_FUNCTIONS = (
        ("get_num_devices", JAI_ARG_OUT),
        ("get_device_num",  JAI_ARG_OUT),
        ("set_device_num",  JAI_ARG_IN ),
        ("device_init",     JAI_ARG_IN ),
        ("device_fini",     JAI_ARG_IN ),
        ("wait",            JAI_ARG_IN )
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

CPP_TEMPLATE_HEADER = """
#include <stdint.h>
#include <stdio.h>

{jmacros}

{cpp_header}

extern "C" {{

{c_header}

{functions}

}}
"""

C_TEMPLATE_FUNCTION = """
int64_t {name}({dargs}) {{

int64_t jai_res;
jai_res = 0;

{body}

return jai_res;
}}
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
#    jai_ccall2(dtypestr, libfunc, actual_args)
end


#function jai_ccall2(dtypestr::String, libfunc::Ptr{Nothing}, args::Vector{Ptr{Any}}) :: Int64
#
#    funcstr = _ccs[1] * dtypestr * _ccs[2] * join(_dummyargs[1:length(args)], ",") * _ccs[3]
#    fid = read(IOBuffer(sha1(funcstr)[1:8]), Int64)
#
#    if ! haskey(_ccall_cache, fid)
#        _ccall_cache[fid] = eval(Meta.parse(funcstr))
#    end
#
#    #actual_args = map(x -> x[1], args)
#    #Base.invokelatest(_ccall_cache[fid], libfunc, actual_args)
#    Base.invokelatest(_ccall_cache[fid], libfunc, args)
#end

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
        frametype   ::JAI_TYPE_FRAMEWORK,
        code    ::String,
        compile ::String,
        srcname ::String,
        outname ::String,
        workdir ::String
    ) ::String

    curdir = pwd()

    try
        cd(workdir)

        open(srcname, "w") do io
            write(io, code)
        end

        #cmds = get_prerun()
        #cmd = Cmd(`bash -c "module load rocm; module load craype-accel-amd-gfx90a; $(compile) -o $(outname) $(srcname); env"`, dir=workdir)
        #c = IOCapture.capture() do
        #    run(cmd3)
        #end
        output = read(run(`$(split(compile)) -o $outname $srcname`), String)

        isfile(outname) || throw(JAI_ERROR_COMPILE_NOSHAREDLIB(compile, output))

        return joinpath(workdir, outname)

    catch e
        rethrow(e)

    finally
        cd(curdir)
    end
end


function load_sharedlib(libpath::String) :: Ptr{Nothing}
    return dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
end


#function invoke_sharedfunc(
#        frame   ::JAI_TYPE_FRAMEWORK,
#        apitype     ::JAI_TYPE_LAUNCH,
#        slib    ::Ptr{Nothing},
#        fname   ::String,
#        args    ::JAI_TYPE_ARGS
#    ) :: Nothing
#
#    if DEBUG
#        println("Enter invoke_sharedfunc2: \n    " * string(frame) * "\n    " * fname)
#    end
#
#    dtypes = argsdtypes(frame, args[1:end-1])
#    dtypestr = join([string(t) for t in dtypes], ",")
#
#    println("DTYPESTR: ", dtypestr)
#    println("args last: ", args)
#
#
#    libfunc = dlsym(slib, Symbol(fname))
#
#    check_retval(jai_ccall2(dtypestr, libfunc, args[end][1]))
#
#    if DEBUG
#    #    println("Exit invoke_sharedfunc2")
#    end
#end

function invoke_sharedfunc(
        frame   ::JAI_TYPE_FRAMEWORK,
        slib    ::Ptr{Nothing},
        fname   ::String,
        args    ::JAI_TYPE_ARGS
    ) :: Nothing

    if DEBUG
        println("Enter invoke_sharedfunc: \n    " * string(frame) * "\n    " * fname)
    end

    dtypes = argsdtypes(frame, args)
    dtypestr = join([string(t) for t in dtypes], ",")

    libfunc = dlsym(slib, Symbol(fname))

    check_retval(jai_ccall(dtypestr, libfunc, args))

    if DEBUG
        println("Exit invoke_sharedfunc")
    end
end

function code_cpp_macros(
        frametype   ::JAI_TYPE_CPP_FRAMEWORKS,
        apitype     ::JAI_TYPE_API,
        interop_frametypes  ::Vector{JAI_TYPE_FRAMEWORK},
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, String} where N
    ) :: String

    macros = Vector{String}()

    push!(macros, "#define JLENGTH(varname, dim) $(prefix)length_##varname##dim")
    push!(macros, "#define JSIZE(varname) $(prefix)size_##varname")

    device = frametype in (JAI_CUDA, JAI_HIP) ? "__device__ " : ""

    for (var, dtype, vname, vinout, addr, vshape, voffset) in args
        if var isa AbstractArray
            accum = 1
            for (idx, len) in enumerate(reverse(vshape))
                push!(macros, device*"const uint64_t "*prefix*"length_"*vname*string(idx-1)*
                        " = "*string(len)*";")
                accum *= len
            end
            push!(macros, device*"const uint64_t "*prefix*"size_"*vname*" = "*string(accum)*";" )
        end
    end

    return join(macros, "\n")

end

function generate_code(
        frametype   ::JAI_TYPE_FORTRAN_FRAMEWORKS,
        apitype     ::JAI_TYPE_API,
        interop_frametypes  ::Vector{JAI_TYPE_FRAMEWORK},
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, String} where N;
        launch_config   ::Union{JAI_TYPE_CONFIG, Nothing} = nothing
    ) :: String

    suffix   = JAI_MAP_API_FUNCNAME[apitype]
    specpart = code_module_specpart(frametype, apitype, interop_frametypes, prefix, args, data)
    subppart = code_module_subppart(frametype, apitype, interop_frametypes, prefix, args, data)

    return jaifmt(FORTRAN_TEMPLATE_MODULE, prefix=prefix,
                  suffix=suffix, specpart=specpart, subppart=subppart)
end

function generate_code(
        frametype   ::JAI_TYPE_CPP_FRAMEWORKS,
        apitype     ::JAI_TYPE_API,
        interop_frametypes  ::Vector{JAI_TYPE_FRAMEWORK},
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, String} where N;
        launch_config   ::Union{JAI_TYPE_CONFIG, Nothing} = nothing
    ) :: String


    jmacros = code_cpp_macros(frametype, apitype, interop_frametypes, prefix, args, data)
    cpp_hdr = code_cpp_header(frametype, apitype, interop_frametypes, prefix, args, data)
    c_hdr   = code_c_header(frametype, apitype, interop_frametypes, prefix, args, data)

    if frametype in (JAI_CUDA, JAI_HIP) && apitype == JAI_LAUNCH
        funcs   = code_c_functions(frametype, apitype, interop_frametypes, prefix,
                                        args, data, launch_config)
    else
        funcs   = code_c_functions(frametype, apitype, interop_frametypes, prefix,
                                        args, data)
    end

    return jaifmt(CPP_TEMPLATE_HEADER, jmacros=jmacros, cpp_header=cpp_hdr,
                  c_header=c_hdr, functions=funcs)
end


include("fortran.jl")
include("fortran_omptarget.jl")
include("cpp.jl")
include("cpp_omptarget.jl")
include("hip.jl")
include("compiler.jl")
include("machine.jl")

function generate_sharedlib(
        frametype   ::JAI_TYPE_FRAMEWORK,
        apitype     ::JAI_TYPE_API,
        prefix      ::String,
        compile     ::String,
        workdir     ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::Vararg{String, N} where N;
        launch_config   ::Union{JAI_TYPE_CONFIG, Nothing} = nothing,
        interop_frametypes  ::Union{Vector{JAI_TYPE_FRAMEWORK},
                                Nothing} = nothing
    ) :: Ptr{Nothing}

    # NOTE: interop_frames: 1) jai_launch case -> accel framework
    #                       2) jai_data case -> kernel frameworks

    if DEBUG
        println("Enter generate_sharedlib: \n    " * string(frametype) *
                "\n    " * string(apitype))
    end

    if interop_frametypes == nothing
        interop_frametypes = Vector{JAI_TYPE_FRAMEWORK}()
    end

    code = generate_code(frametype, apitype, interop_frametypes, prefix, args,
                    data, launch_config=launch_config)

    if frametype isa JAI_TYPE_FORTRAN_FRAMEWORKS
        srcname = prefix * JAI_MAP_API_FUNCNAME[apitype] * ".F90"

    elseif frametype isa JAI_TYPE_CPP_FRAMEWORKS
        srcname = prefix * JAI_MAP_API_FUNCNAME[apitype] * ".cpp"
    else
        error("Unknown language: " * string(frametype))
    end

    outname = prefix * JAI_MAP_API_FUNCNAME[apitype] * "." * dlext

    slibpath = compile_code(frametype, code, compile, srcname, outname, workdir)

    slib = load_sharedlib(slibpath)

    # init device
    if apitype == JAI_ACCEL
        invoke_sharedfunc(frametype, slib, prefix * "device_init", args)
    end

    if DEBUG
        println("Exit generate_sharedlib")
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

        # per each frametype, there could be multiple compiler command lines
        frames = frameworks[frametype]

        if length(frames) > 0
            if compile isa String
                cid = generate_jid(compile)
                if cid in keys(frames)
                    return frames[cid]
                end
            elseif compile == nothing
                # frames Pair(Int64, JAI_TYPE_CONTEXT_FRAMEWORK)
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

    for (frametype, config) in userframe
        framework = get_framework(frametype, config, compiler, workdir)
        if framework isa JAI_TYPE_CONTEXT_FRAMEWORK
            return framework
        end
    end

    throw(JAI_ERROR_NOAVAILABLE_FRAMEWORK())
end


# accel context
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
