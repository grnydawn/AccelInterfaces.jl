# main.jl: processes user requests from Jai API


# Jai context
abstract type JAI_TYPE_CONTEXT end

# Jai host context
struct JAI_TYPE_CONTEXT_HOST <: JAI_TYPE_CONTEXT

    acfg ::Dict{String, JAI_TYPE_CONFIG}

    function JAI_TYPE_CONTEXT_HOST(acfg::Dict{String, JAI_TYPE_CONFIG})

        for (name, default) in JAI_ACCEL_CONFIGS
            if !haskey(acfg, name)
                acfg[name] = default
            end
        end

        new(acfg)
    end
end

# accelerator context
struct JAI_TYPE_CONTEXT_ACCEL <: JAI_TYPE_CONTEXT
    aname           ::String
    aid             ::UInt32
    ctx_host        ::JAI_TYPE_CONTEXT_HOST

    function JAI_TYPE_CONTEXT_ACCEL(aname, aid, ctx_host)

        # create directories
        workdir = ctx_host.acfg["workdir"]

        if !isdir(workdir)
            pidfile = joinpath(ctx_host.acfg["pidfilename"])
            locked_filetask(pidfile, workdir, mkdir, workdir)
        end

        # select data framework and generate a shared library for accel

        # init device and gather device information

        new(aname, aid, ctx_host)
    end
end

# kernel context
struct JAI_TYPE_CONTEXT_KERNEL <: JAI_TYPE_CONTEXT
    kname           ::String
    accelctx        ::JAI_TYPE_CONTEXT_ACCEL

    function JAI_TYPE_CONTEXT_KERNEL(kname, actx)

        # generate kernel context id
        
        # load kernel definition

        new(kname, actx)
    end
end


# accelerator context cache
const ctx_accels = Vector{JAI_TYPE_CONTEXT_ACCEL}()

# kernel context cache
const ctx_kernels = Vector{JAI_TYPE_CONTEXT_KERNEL}()


function get_context(contexts, name)

    ret = nothing

    for ctx in contexts
        if ctx.aname == name
            ret = ctx
            break
        end
    end

    if name == "" && length(contexts) == 1
        ret = contexts[1]
    end

    return ret
end

get_accel(name)     = get_context(ctx_accels, name)
get_kernel(name)    = get_context(ctx_kernels, name)


struct JAI_TYPE_KERNELDEF
end

"""
    function jai_accel

Process @jaccel macro

# Implementation
  * creates AccelContext and save it in a stack
  * select a framework for data transfer
  * save constant variables
  * configure accel parameters

"""

function jai_accel(
        aname       ::String,
        lineno      ::Integer,
        filepath    ::String;
        const_vars  ::NTuple{N, JAI_TYPE_DATA} where N   = (),
        const_names ::NTuple{N, String} where N         = (),
        framework   ::Dict{String, JAI_TYPE_CONFIG}= JAI_CONFIG_BLANK,
        device      ::NTuple{N, Integer} where N        = (),
        compile     ::NTuple{N, String} where N         = (),
        set         ::Dict{String, JAI_TYPE_CONFIG}= JAI_CONFIG_BLANK
    )

    ctx_host = JAI_TYPE_CONTEXT_HOST(set)

    aid = generate_jid(aname, Sys.STDLIB, JAI_VERSION, lineno, filepath)

    if aname == ""
        aname = string(aid, base = 16)
    end

    ctx_accel = JAI_TYPE_CONTEXT_ACCEL(aname, aid, ctx_host)

    push!(ctx_accels, ctx_accel)
end

"""
    function jai_data

Process @jenterdata and @jexitdata macros

# Implementation
  * drive the generation of data transfer source files based on the framework selected at jai_accel
  * compile and load a shared library
  * invoke data transfer and memory allocation/deallocation functions
  * make aliases of device variables for inter-operability to kernels

"""

function jai_data(
        aname       ::String,
        apitype     ::JAI_TYPE_API,
        apicount    ::Integer,
        names       ::Vector{String},
        control     ::Vector{String},
        lineno      ::Integer,
        filepath    ::String,
        data        ::Vararg{JAI_TYPE_DATA, N} where N
    )

    # pack data and variable names

    # jai ccall if cached

    # generate source file hash

    # generate source file

    # compile source file to shared library

    # jai ccall and save it in cache

end


"""
    function jai_kernel

Process @jkernel

# Implementation
  * parse kernel file
  * select a framework for kernel
  * pre-process as much as possible to reduce kernel runtime

"""

function jai_kernel(
        kdef        ::Union{JAI_TYPE_KERNELDEF, String},
        kname       ::String,
        aname       ::String,
        lineno      ::Integer,
        filepath    ::String;
        framework   ::Dict{String, JAI_TYPE_CONFIG}= JAI_CONFIG_BLANK
    )

    # find ctx_accel
    ctx_accel = get_accel(aname)

    ctx_kernel = JAI_TYPE_CONTEXT_KERNEL(kname, ctx_accel)

    push!(ctx_kernels, ctx_kernel)

end


"""
    function jai_launch

Process @jlaunch

# Implementation
  * drive the generation of kernel source files
  * compile and load a shared library
  * launch kernel on device
"""

function jai_launch(
        kname       ::String,
        aname       ::String,
        input       ::NTuple{N, JAI_TYPE_DATA} where N,
        output      ::NTuple{N, JAI_TYPE_DATA} where N,
        innames     ::Vector{String},
        outnames    ::Vector{String},
        lineno      ::Integer,
        filepath    ::String,
        config      ::Dict{String, Union{Dict{String, JAI_TYPE_CONFIG}, Nothing}}
    )

    # pack data and variable names

    # jai ccall if cached

    # generate source file hash

    # generate source file

    # compile source file to shared library

    # jai ccall and save it in cache


end


"""
    function jai_wait

Process @jwait

# Implementation
  * invoke wait-equivalent framework api

"""

function jai_wait(
        aname       ::String,
        lineno      ::Integer,
        filepath    ::String
    )

    # jai ccall

end


"""
    function jai_decel

Process @jdecel macro

# Implementation
  * remove AccelContext

"""

function jai_decel(
        aname       ::String,
        lineno      ::Integer,
        filepath    ::String
    )

    # jai ccall

    if aname == ""
        if length(ctx_accels) > 0
            pop!(ctx_accels)
        else
            println("WARNING: no accel context exists")
        end
    else
        ctxidx = nothing

        for idx in range(1, length(ctx_accels))
            if ctx_accels[idx].aname == aname
                ctxidx = idx
                break
            end
        end

        if ctxidx isa Number
            deleteat!(ctx_accels, ctxidx)
        else
            println("WARNING: no accel context name: " * aname)
        end
    end
end
