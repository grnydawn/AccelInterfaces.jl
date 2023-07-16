# framework.jl: implement common functions for framework interfaces

#import InteractiveUtils.subtypes
import UUIDs.uuid1
import Pidfile: mkpidlock, LockMonitor
import Libdl: dlopen, RTLD_LAZY, RTLD_DEEPBIND, RTLD_GLOBAL, dlext, dlsym, dlclose
#import IOCapture
import OffsetArrays: OffsetVector, OffsetArray

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

# mapping what data framework can interoperate with other frameworks
# accel/kernel framework => [data frameworks]
# step 1: union accel/kernel frameworks, and union data frameworks
# step 2: loop over data frameworks that supports union of accel/kernels
const JAI_MAP_INTEROP_FRAMEWORK = OrderedDict(
        JAI_FORTRAN             => [
                            JAI_FORTRAN,
                            JAI_CPP,
                            JAI_FORTRAN_OMPTARGET,
                            JAI_FORTRAN_OPENACC,
                            JAI_CPP_OMPTARGET,
                            JAI_CPP_OPENACC,
                            JAI_CUDA,
                            JAI_HIP
                        ],
        JAI_CPP                 => [
                            JAI_CPP,
                            JAI_FORTRAN,
                            JAI_FORTRAN_OMPTARGET,
                            JAI_FORTRAN_OPENACC,
                            JAI_CPP_OMPTARGET,
                            JAI_CPP_OPENACC,
                            JAI_CUDA,
                            JAI_HIP
                        ],
        JAI_CUDA                => [
                            JAI_CUDA
                        ],
        JAI_HIP                 => [
                            JAI_HIP
                        ],
        JAI_FORTRAN_OMPTARGET   => [  
                            JAI_FORTRAN_OMPTARGET,
                            JAI_HIP
                        ],
        JAI_FORTRAN_OPENACC     => [  
                            JAI_FORTRAN_OPENACC
                        ],
        JAI_CPP_OMPTARGET   => [  
                            JAI_CPP_OMPTARGET,
                            JAI_HIP
                        ],
        JAI_CPP_OPENACC     => [  
                            JAI_CPP_OPENACC
                        ]
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

!print *, "Exits {prefix}{suffix}"

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

//printf("Entering %s\\n", "{name}");

{body}

//printf("Exiting %s\\n", "{name}");
return jai_res;
}}
"""

function argsdtypes(
        frame   ::JAI_TYPE_FRAMEWORK,
        args    ::JAI_TYPE_ARGS
    ) :: Vector{DataType}

    local N = length(args)

    dtypes = Vector{DataType}(undef, N)

    for (i, (arg, name, inout, addr, shape, offsets, extname)) in enumerate(args)

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
        srcname     ::String,
        frametype   ::JAI_TYPE_FRAMEWORK,
        apitype     ::JAI_TYPE_API,
        prefix      ::String,
        cvars       ::JAI_TYPE_ARGS,
        args        ::JAI_TYPE_ARGS,
        clauses     ::JAI_TYPE_CONFIG,
        data        ::NTuple{N, String} where N,
        launch_config   ::Union{JAI_TYPE_CONFIG, Nothing},
        compile     ::String,
        outname     ::String
    )

    # TODO: add compile argument to generate_code

    if !isfile(srcname)
        code = generate_code(frametype, apitype, prefix, cvars, args,
                            clauses, data, launch_config=launch_config)

        open(srcname, "w") do io
            write(io, code)
        end
    end

    serr = IOBuffer()
    run(pipeline(`$(split(compile)) -o $outname $srcname`, stderr=serr))

    isfile(outname) || throw(JAI_ERROR_COMPILE_NOSHAREDLIB(compile, take!(serr)))
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

const _dummyargs = Tuple("a[$i]" for i in range(1, stop=200))
const _ccs = ("(f,a) -> ccall(f, Int64, (", ",), " , ")")

function invoke_sharedfunc(
        frame   ::JAI_TYPE_FRAMEWORK,
        slib    ::Ptr{Nothing},
        fname   ::String,
        args    ::JAI_TYPE_ARGS
    ) :: Nothing

    if DEBUG
        println("Enter invoke_sharedfunc: \n    " * string(frame) * "\n    " * fname)
    end

    libfunc = dlsym(slib, Symbol(fname))
    dtypes = argsdtypes(frame, args)
    fid = hash(dtypes)

    if ! haskey(JAI["ccall_jids"], fid)
        dtypestr = join([string(t) for t in dtypes], ",")

        funcstr = _ccs[1] * dtypestr * _ccs[2] * join(_dummyargs[1:length(args)], ",") * _ccs[3]

        JAI["ccall_jids"][fid] = eval(Meta.parse(funcstr))
    end

    actual_args = map(x -> x[1], args)
    retval = Base.invokelatest(JAI["ccall_jids"][fid], libfunc, actual_args)

    check_retval(retval)

    if DEBUG
        println("Exit invoke_sharedfunc")
    end
end

function code_cpp_macros(
        frametype   ::JAI_TYPE_CPP_FRAMEWORKS,
        apitype     ::JAI_TYPE_API,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        clauses     ::JAI_TYPE_CONFIG,
        data        ::NTuple{N, String} where N
    ) :: String

    macros = Vector{String}()

    push!(macros, "#define JLENGTH(varname, dim) $(prefix)length_##varname##dim")
    push!(macros, "#define JSIZE(varname) $(prefix)size_##varname")

    device = frametype in (JAI_CUDA, JAI_HIP) ? "__device__ " : ""

    for (var, dtype, vname, vinout, addr, vshape, voffset, extname) in args
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
        prefix      ::String,
        cvars       ::JAI_TYPE_ARGS,
        args        ::JAI_TYPE_ARGS,
        clauses     ::JAI_TYPE_CONFIG,
        data        ::NTuple{N, String} where N;
        launch_config   ::Union{JAI_TYPE_CONFIG, Nothing} = nothing,
        difftest    ::Union{Dict{String, Any}, Nothing} = nothing,
    ) :: String

    suffix   = JAI_MAP_API_FUNCNAME[apitype]
    specpart = code_module_specpart(frametype, apitype, prefix, cvars, args, clauses, data)
    subppart = code_module_subppart(frametype, apitype, prefix, args, clauses, data)

    return jaifmt(FORTRAN_TEMPLATE_MODULE, prefix=prefix,
                  suffix=suffix, specpart=specpart, subppart=subppart)
end

function generate_code(
        frametype   ::JAI_TYPE_CPP_FRAMEWORKS,
        apitype     ::JAI_TYPE_API,
        prefix      ::String,
        cvars       ::JAI_TYPE_ARGS,
        args        ::JAI_TYPE_ARGS,
        clauses     ::JAI_TYPE_CONFIG,
        data        ::NTuple{N, String} where N;
        launch_config   ::Union{JAI_TYPE_CONFIG, Nothing} = nothing,
        difftest    ::Union{Dict{String, Any}, Nothing} = nothing,
    ) :: String


    jmacros = code_cpp_macros(frametype, apitype, prefix, args, clauses, data)
    cpp_hdr = code_cpp_header(frametype, apitype, prefix, cvars, args, clauses, data)
    c_hdr   = code_c_header(frametype, apitype, prefix, args, clauses, data)

    if frametype in (JAI_CUDA, JAI_HIP) && apitype == JAI_LAUNCH
        funcs   = code_c_functions(frametype, apitype, prefix, args, clauses, data, launch_config)
    else
        funcs   = code_c_functions(frametype, apitype, prefix, args, clauses, data)
    end

    return jaifmt(CPP_TEMPLATE_HEADER, jmacros=jmacros, cpp_header=cpp_hdr,
                  c_header=c_hdr, functions=funcs)
end


include("fortran.jl")
include("fortran_omptarget.jl")
include("fortran_openacc.jl")
include("cpp.jl")
include("cpp_omptarget.jl")
include("cuda.jl")
include("hip.jl")
include("compiler.jl")
include("machine.jl")

function generate_sharedlib(
        frametype   ::JAI_TYPE_FRAMEWORK,
        apitype     ::JAI_TYPE_API,
        prefix      ::String,
        compile     ::String,
        workdir     ::String,
        cachedir    ::String,
        debugdir    ::String,
        cvars       ::JAI_TYPE_ARGS,
        args        ::JAI_TYPE_ARGS,
        clauses     ::JAI_TYPE_CONFIG,
        data        ::Vararg{String, N} where N;
        launch_config   ::Union{JAI_TYPE_CONFIG, Nothing} = nothing,
        difftest    ::Union{Dict{String, Any}, Nothing} = nothing,
    ) :: Ptr{Nothing}

    if DEBUG
        println("Enter generate_sharedlib: \n    " * string(frametype) *
                "\n    " * string(apitype))
    end

    if frametype isa JAI_TYPE_FORTRAN_FRAMEWORKS
        suffix = ".F90"

    elseif frametype isa JAI_TYPE_CPP_FRAMEWORKS
        if frametype isa JAI_TYPE_CUDA
            suffix = ".cu"
        else
            suffix = ".cpp"
        end
    else
        error("Unknown language: " * string(frametype))
    end

    srcname = prefix * JAI_MAP_API_FUNCNAME[apitype] * suffix
    outname = prefix * JAI_MAP_API_FUNCNAME[apitype] * "." * dlext

    slibpath = joinpath(cachedir, outname)

    if !isfile(slibpath)
 
        curdir = pwd()
        genlock = nothing

        myuid = string(uuid1())
        myworkdir = joinpath(workdir, myuid)

        try

            # geneate shared library
            pidgenfile = joinpath(workdir, outname * ".genpid")
            genlock = mkpidlock(pidgenfile, stale_age=3)

            if !isfile(slibpath)

                if !isdir(myworkdir)
                    mkdir(myworkdir)
                end

                cd(myworkdir)

                compile_code(srcname, frametype, apitype, prefix, cvars, args,
                            clauses, data, launch_config, compile, outname)

                copylock = nothing

                # copy shared library
                try

                    pidcopyfile = joinpath(workdir, outname * ".copypid")
                    copylock = mkpidlock(pidcopyfile, stale_age=3)

                    if !isfile(slibpath)
                        cp(outname, slibpath)
                    end

                    if isdir(debugdir)
                        for name in readdir(myworkdir)
                            mv(name, joinpath(debugdir, name), force=true)
                        end
                    end

                catch err
                    if isdir(debugdir)
                        try
                            if isfile(outname)
                                mv(outname, joinpath(debugdir, outname), force=true)
                            end
                            fd = open(joinpath(debugdir, outname * ".compile"), "w")
                            io = IOBuffer();
                            showerror(io, e)
                            write(fd, String(take!(io)) * "\n\n")
                            write(fd, compile * "\n\n")
                            #write(fd, string(launch_config))
                            close(fd)
                        catch e
                        end
                    end
                    rethrow(err)

                finally

                    if copylock isa LockMonitor
                        close(copylock)
                    end
                end
            end
        catch e
            if isdir(debugdir)
                try
                    if isfile(srcname)
                        mv(srcname, joinpath(debugdir, srcname), force=true)
                    end
                    fd = open(joinpath(debugdir, outname * ".compile"), "w")
                    io = IOBuffer();
                    showerror(io, e)
                    write(fd, String(take!(io)) * "\n\n")
                    write(fd, compile * "\n\n")
                    #write(fd, string(launch_config))
                    close(fd)
                catch e
                end
            end

            rethrow(e)

        finally

            if genlock isa LockMonitor
                close(genlock)
            end

            cd(curdir)

            if isdir(myworkdir)
                rm(myworkdir, force=true, recursive=true)
            end
        end
    end

    slib = nothing
    delta = Second(10)
    start  = now()

    while (!(slib isa Ptr{Nothing}) && start + delta > now())  

        if filesize(slibpath) < 1000
            sleep(0.1)
            continue
        end

        try
            slib = load_sharedlib(slibpath)
        catch e
        finally
            sleep(0.1)
        end
    end

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
        devices     ::Dict{Integer, Bool},
        compiler    ::Union{JAI_TYPE_CONFIG, Nothing},
        workdir     ::String,
        cachedir    ::String,
        debugdir    ::String
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

    if compiler == nothing
        compiler = JAI_TYPE_CONFIG()
    end

    frameworks = JAI["frameworks"]

    if frametype in keys(frameworks) #16

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
    vbuf = fill(Int64(0), 1)
    push!(args, pack_arg(vbuf, nothing, nothing))
    cvars = JAI_TYPE_ARGS()
    clauses = JAI_TYPE_CONFIG()

    for compile in compiles

        cid     = generate_jid(compile)
        prefix  = generate_prefix(JAI_MAP_FRAMEWORK_STRING[frametype], cid)
        slib    = generate_sharedlib(frametype, JAI_ACCEL, prefix, compile,
                        workdir, cachedir, debugdir, cvars, args, clauses)

        if slib isa Ptr{Nothing}

            # check if devices is configured in accel
            if length(devices) > 0

                # for now, support only one device per accel context
                devnum = [k for k in keys(devices)][1]

                if !devices[devnum]
                    # set device num
                    vbuf[1] = devnum

                    invoke_sharedfunc(frametype, slib, prefix * "set_device_num", args)

                    devices[devnum] = true
                end
            end

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
        workdir     ::String,
        cachedir    ::String,
        debugdir    ::String
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
        framework = get_framework(frametype, config, compiler, workdir,
                        cachedir, debugdir)
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
        workdir     ::String,
        debugdir    ::String
    ) :: Union{JAI_TYPE_CONTEXT_FRAMEWORK, Nothing}

    if userframe == nothing
        userframe = JAI_TYPE_CONFIG()
    end

    if length(userframe) == 0
        userframe[ctx_accel.framework.type] = nothing
    end

    return select_framework(userframe, compiler, workdir, debugdir)
end

# data framework
function select_data_framework(
        ctx_accel   ::JAI_TYPE_CONTEXT_ACCEL
    ) :: JAI_TYPE_CONTEXT_FRAMEWORK

    code_frames = Vector{JAI_TYPE_FRAMEWORK}()
    data_frames = Vector{JAI_TYPE_FRAMEWORK}()
    ctx_frames = Dict{JAI_TYPE_FRAMEWORK, JAI_TYPE_CONTEXT_FRAMEWORK}()

    for ctx_kernel in ctx_accel.ctx_kernels
        for ctx_frame in ctx_kernel.frameworks

            kframe = ctx_frame.type
            ctx_frames[ctx_frame.type] = ctx_frame

            if !(kframe in code_frames)
                push!(code_frames, kframe)
            end

            len = length(data_frames)

            if len == 0
                data_frames = copy(JAI_MAP_INTEROP_FRAMEWORK[kframe])
            else
                dframes = JAI_MAP_INTEROP_FRAMEWORK[kframe]
                idx = 1
                while idx <= length(data_frames)
                    if !(data_frames[idx] in dframes)
                        popat!(data_frames, idx)
                    else
                        idx += 1
                    end
                end
            end
            if length(data_frames) == 0
                break
            end
        end
        if length(data_frames) == 0
            break
        end
    end

    data_frame = nothing

    for dframe in data_frames
        found = true

        for cframe in code_frames
            dframes = JAI_MAP_INTEROP_FRAMEWORK[cframe]
            if !(dframe in dframes)
                found = false
                break
            end
        end

        if found
            data_frame = dframe
            break
        end
    end

    ret_ctx = nothing

    if data_frame != nothing
        ret_ctx = ctx_frames[data_frame]
        push!(ctx_accel.data_framework, ret_ctx)
    end

    return ret_ctx
end
