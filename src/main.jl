# main.jl: processes user requests from Jai API


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
        device      ::NTuple{N, Integer} where N        = (),
        framework   ::JAI_TYPE_CONFIG = JAI_CONFIG_BLANK,
        machine     ::JAI_TYPE_CONFIG = JAI_CONFIG_BLANK,
        compiler    ::JAI_TYPE_CONFIG = JAI_CONFIG_BLANK,
        set         ::JAI_TYPE_CONFIG = JAI_CONFIG_BLANK
    )

    ctx_host = JAI_TYPE_CONTEXT_HOST(set)

    aid = generate_jid(aname, Sys.STDLIB, JAI_VERSION, lineno, filepath)

    if aname == ""
        aname = string(aid, base = 16)
    end

    # create directories
    workdir = ctx_host.config["workdir"]

    if !isdir(workdir)
        pidfile = joinpath(ctx_host.config["pidfilename"])
        locked_filetask(pidfile, workdir, mkdir, workdir)
    end

    cvars = OrderedDict{String, JAI_TYPE_DATA}()
    for (name, var) in zip(const_names, const_vars)
        cvars[name] = var
    end

    prefix = join(["jai", aname, string(aid, base=16)], "_") * "_"

    # select data framework and generate a shared library for accel
    frame, fslib, fcfg = select_framework(framework, prefix, workdir)

    slibcache = Dict{UInt32, Ptr{Nothing}}()

    ctx_kernels = Vector{JAI_TYPE_CONTEXT_KERNEL}()

    ctx_accel = JAI_TYPE_CONTEXT_ACCEL(aname, aid, prefix, ctx_host, cvars,
                    device, frame, fslib, fcfg, slibcache, ctx_kernels)

    push!(JAI["ctx_accels"], ctx_accel)
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
    args = JAI_TYPE_ARGS()
    for (n, d) in zip(names, data)
        arg = pack_arg(d, name=n, inout=JAI_MAP_APITYPE_INOUT[apitype])
        push!(args, arg)
    end

    ctx_accel   = get_accel(aname)
    uid         = generate_jid(ctx_accel.aid, apitype, lineno, filepath)
    prefix      = join(["jai", aname, string(uid, base=16)], "_") * "_"
    workdir     = ctx_accel.ctx_host.config["workdir"]

    try
        # build if not cached
        if !(uid in keys(ctx_accel.slibcache))
            slib = genslib_data(ctx_accel.frame, apitype, prefix, workdir, args)
            ctx_accel.slibcache[uid] = slib
        end

        # jai ccall and save it in cache
        invoke_slibfunc(ctx_accel.frame, ctx_accel.slibcache[uid],
                        prefix * JAI_MAP_API_FUNCNAME[apitype], args)

    catch err
        rethrow()
    end

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
        framework   ::JAI_TYPE_CONFIG= JAI_CONFIG_BLANK
    )

    # find ctx_accel
    ctx_accel = get_accel(aname)

    if !(kdef isa JAI_TYPE_KERNELDEF)
        kdef = parse_kerneldef(kdef)
    end

    # generate kernel context id
    kid = generate_jid(ctx_accel.aid, kname, kdef.kdid, lineno, filepath)
        
    frame, fcfg = select_framework(ctx_accel, framework)

    ctx_kernel = JAI_TYPE_CONTEXT_KERNEL(kid, kname, frame, fcfg, kdef)

    push!(ctx_accel.ctx_kernels, ctx_kernel)

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
        config      ::JAI_TYPE_CONFIG
    )

    apitype = JAI_LAUNCH

    args = pack_args(innames, input, outnames, output)

    ctx_accel   = get_accel(aname)
    ctx_kernel  = get_kernel(ctx_accel, kname)
    uid         = generate_jid(ctx_kernel.kid, apitype, lineno, filepath)
    prefix      = join(["jai", kname, string(uid, base=16)], "_") * "_"
    workdir     = ctx_accel.ctx_host.config["workdir"]
    knlbody     = get_knlbody(ctx_kernel)

    try
        # build if not cached
        if !(uid in keys(ctx_accel.slibcache))
            slib = genslib_kernel(ctx_accel.frame, prefix, workdir, args, knlbody)
            ctx_accel.slibcache[uid] = slib
        end

        # jai ccall and save it in cache
        invoke_slibfunc(ctx_accel.frame, ctx_accel.slibcache[uid],
                        prefix * JAI_MAP_API_FUNCNAME[apitype], args)

    catch err
        rethrow()
    end


end


"""
    function jai_wait

Process @jwait

# Implementation
  * invoke wait-equivalent framework api

"""

function jai_wait(
        name       ::String,
        lineno      ::Integer,
        filepath    ::String
    )

    slib = nothing
    ctx_accel = nothing
    ctx_kernel = nothing

    if name == ""
        ctx_accel   = get_accel(name)
        ctx_kernel   = get_kernel(ctx_accel, name)
        if ctx_kernel != nothing
            slib = JAI_AVAILABLE_FRAMEWORKS[ctx_kernel.frame]
        else
            slib = JAI_AVAILABLE_FRAMEWORKS[ctx_accel.frame]
        end
    else
        ctx_accel   = get_accel(name)
        if ctx_accel == nothing
            ctx_kernel = nothing
            for ctx_accel in JAI["ctx_accels"]
                ctx_kernel = get_kernel(ctx_accel, name)
                if ctx_kernel != nothing
                    slib = JAI_AVAILABLE_FRAMEWORKS[ctx_kernel.frame]
                    break
                end
            end

            if slib == nothing
                error("Can not find shared library for jwait with " * name)
            end
        else
            ctx_kernel   = get_kernel(ctx_accel, name)
            if ctx_kernel != nothing
                slib = JAI_AVAILABLE_FRAMEWORKS[ctx_kernel.frame]
            else
                slib = JAI_AVAILABLE_FRAMEWORKS[ctx_accel.frame]
            end
        end
    end

    args = JAI_TYPE_ARGS()
    push!(args, pack_arg(fill(Int64(-1), 1)))

    # jai ccall and save it in cache
    invoke_slibfunc(ctx_accel.frame, slib,
                    ctx_accel.prefix * JAI_MAP_API_FUNCNAME[JAI_WAIT], args)

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

    ctx_accel = nothing

    if aname == ""
        if length(JAI["ctx_accels"]) > 0
            ctx_accel = pop!(JAI["ctx_accels"])
        else
            println("WARNING: no accel context exists")
        end
    else
        ctxidx = nothing

        for idx in range(1, length(JAI["ctx_accels"]))
            if JAI["ctx_accels"][idx].aname == aname
                ctxidx = idx
                break
            end
        end

        if ctxidx isa Number
            ctx_accel = popat!(JAI["ctx_accels"], ctxidx)
        else
            println("WARNING: no accel context name: " * aname)
        end
    end

    if ctx_accel == nothing
        println("WARNING: no accel context name: " * aname)
    else

        for ctx_kernel in ctx_accel.ctx_kernels
            # TODO: terminate ctx_kernel
        end

        # TODO: terminate ctx_accel
    end
end
