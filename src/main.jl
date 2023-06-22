# main.jl: processes user requests from Jai API

"""
    function jai_config

Process @jconfig macro

# Implementation
  * handles application-wide configurations

"""

function jai_config(
        lineno      ::Integer,
        filepath    ::String;
        const_vars  ::Union{NTuple{N, JAI_TYPE_DATA} where N, Nothing}= nothing,
        const_names ::Union{NTuple{N, String} where N, Nothing}       = nothing,
        device      ::Union{NTuple{N, Integer} where N, Nothing}      = nothing,
        machine     ::Union{NTuple{N, String} where N, Nothing}       = nothing,
        framework   ::Union{JAI_TYPE_CONFIG, Nothing} = nothing,
        compiler    ::Union{JAI_TYPE_CONFIG, Nothing} = nothing,
        set         ::Union{JAI_TYPE_CONFIG, Nothing} = nothing
    )
 
    if set != nothing
        set_config(set)
    end
 
    if machine != nothing
        for m in machine
            set_machine(m)
        end
    end

end


"""
    function jai_accel

Process @jaccel macro

# Implementation
  * creates AccelContext and save it in a stack
  * select a framework type
  * save constant variables
  * configure accel parameters

"""

function jai_accel(
        aname       ::String,
        lineno      ::Integer,
        filepath    ::String;
        const_vars  ::Union{NTuple{N, JAI_TYPE_DATA} where N, Nothing}= nothing,
        const_names ::Union{NTuple{N, String} where N, Nothing}       = nothing,
        device      ::Union{NTuple{N, Integer} where N, Nothing}      = nothing,
        framework   ::Union{JAI_TYPE_CONFIG, Nothing}   = nothing,
        machine     ::Union{JAI_TYPE_CONFIG, Nothing}   = nothing,
        compiler    ::Union{JAI_TYPE_CONFIG, Nothing}   = nothing,
        set         ::Union{JAI_TYPE_CONFIG, Nothing}   = nothing
    )

    config = JAI_TYPE_CONFIG_USER()
    
    if set != nothing
        set_config(config, set)
    end

    if device == nothing
        device = ()
    end

    # TODO: apply accel specific config in set
    #
    aid = generate_jid(aname, Sys.STDLIB, JAI_VERSION, lineno, filepath)

    if aname == ""
        aname = string(aid, base = 16)
    end

    # pack const and variable names
    cvars = JAI_TYPE_ARGS()
    if const_names != nothing && const_vars != nothing
        for (n, v) in zip(const_names, const_vars)
            cvar = pack_arg(v, name=n, inout=JAI_ARG_IN)
            push!(cvars, cvar)
        end
    end

    workdir     = config.workdir
    if workdir == nothing
        workdir = get_config("workdir")
    end

    ctx_framework   = select_framework(framework, compiler, workdir)
    data_framework  = Vector{Tuple{JAI_TYPE_FRAMEWORK, String}}()
    ctx_kernels = Vector{JAI_TYPE_CONTEXT_KERNEL}()
    data_slibs  = Dict{UInt32, PtrAny}()

    ctx_accel = JAI_TYPE_CONTEXT_ACCEL(aname, aid, config, cvars, device,
                       ctx_framework, data_framework, data_slibs, ctx_kernels)

    push!(JAI["ctx_accels"], ctx_accel)
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
        framework   ::Union{JAI_TYPE_CONFIG, Nothing}= nothing,
        compiler    ::Union{JAI_TYPE_CONFIG, Nothing}= nothing
    )

    # find ctx_accel
    ctx_accel = get_accel(aname)

    if !(kdef isa JAI_TYPE_KERNELDEF)
        kdef = parse_kerneldef(kdef)
    end

    # generate kernel context id
    kid = generate_jid(ctx_accel.aid, kname, kdef.kdid, lineno, filepath)
 
    if kname == ""
        kname = string(kid, base = 16)
    end
       
    workdir     = get_config(ctx_accel, "workdir")
    ctx_frame   = select_framework(ctx_accel, framework, compiler, workdir)
    launch_slibs= Dict{UInt32, PtrAny}()

    ctx_kernel  = JAI_TYPE_CONTEXT_KERNEL(kid, kname, ctx_frame, launch_slibs, kdef)

    push!(ctx_accel.ctx_kernels, ctx_kernel)

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

    ctx_accel   = get_accel(aname)

    # pack data and variable names
    args = JAI_TYPE_ARGS()
    for (i, (n, d)) in enumerate(zip(names, data))
        arg = pack_arg(d, name=n, inout=JAI_MAP_APITYPE_INOUT[apitype])
        push!(args, arg)
    end

    uid         = generate_jid(ctx_accel.aid, apitype, apicount, lineno, filepath)
    frametype   = ctx_accel.framework.type
    prefix      = generate_prefix(aname, uid)

    try
        if uid in keys(ctx_accel.data_slibs)
            slib = ctx_accel.data_slibs[uid]
        else
            compile = ctx_accel.framework.compile
            workdir = get_config(ctx_accel, "workdir")

#            interop_frametypes  = Vector{JAI_TYPE_FRAMEWORK}()
#            for kctx in ctx_accel.ctx_kernels
#                push!(interop_frametypes, kctx.framework.type)
#            end

            data_frametype, data_compile = select_data_framework(ctx_accel)

            slib    = generate_sharedlib(frametype, apitype, data_frametype,
                        prefix, data_compile, workdir, ctx_accel.const_vars, args)

            ctx_accel.data_slibs[uid] = slib
        end

        funcname = prefix*JAI_MAP_API_FUNCNAME[apitype]
        invoke_sharedfunc(frametype, slib, funcname, args)

    catch err
        rethrow()
    end
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

    apitype     = JAI_LAUNCH
    args        = pack_args(innames, input, outnames, output)
    ctx_accel   = get_accel(aname)
    ctx_kernel  = get_kernel(ctx_accel, kname)
    uid         = generate_jid(ctx_kernel.kid, apitype, lineno, filepath)
    frametype   = ctx_kernel.framework.type
    prefix      = generate_prefix(kname, uid)

    try
        if uid in keys(ctx_kernel.launch_slibs)
            slib    = ctx_kernel.launch_slibs[uid]
        else
            compile = ctx_kernel.framework.compile
            workdir = get_config(ctx_accel, "workdir")
            knlbody = get_knlbody(ctx_kernel)

            data_frametype, data_compile = select_data_framework(ctx_accel)

            slib    = generate_sharedlib(frametype, apitype, data_frametype,
                        prefix, compile, workdir, ctx_accel.const_vars, args, knlbody,
                        launch_config=config)

            ctx_kernel.launch_slibs[uid] = slib
        end

        funcname = prefix*JAI_MAP_API_FUNCNAME[apitype]
        #invoke_sharedfunc(frame, apitype, slib, funcname, args)
        invoke_sharedfunc(frametype, slib, funcname, args)

    catch err
        rethrow()
    end

end


function _jwait(framework::JAI_TYPE_CONTEXT_FRAMEWORK)

    args = JAI_TYPE_ARGS()
    push!(args, pack_arg(fill(Int64(-1), 1)))

    frame   = framework.type
    slib    = framework.slib

    funcname = framework.prefix * JAI_MAP_API_FUNCNAME[JAI_WAIT]
    invoke_sharedfunc(frame, slib, funcname, args)

end

_jwait(ctx::JAI_TYPE_CONTEXT_ACCEL)  = _jwait(ctx.framework)
_jwait(ctx::JAI_TYPE_CONTEXT_KERNEL) = _jwait(ctx.framework)

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

    ctx = nothing
    slib = nothing

    if name == ""
        ctx_accel   = get_accel(name)
        ctx_kernel  = get_kernel(ctx_accel, name)
        ctx_kernel == nothing ? _jwait(ctx_accel) : _jwait(ctx_kernel)
    else
        ctx_accel   = get_accel(name)
        if ctx_accel == nothing
            ctx_kernel = nothing
            for ctx_accel in JAI["ctx_accels"]
                ctx_kernel = get_kernel(ctx_accel, name)
                if ctx_kernel != nothing
                    _jwait(ctx_kernel)
                    break
                end
            end

            if ctx_kernel == nothing
                error("Can not find shared library for jwait with " * name)
            end
        else
            ctx_kernel   = get_kernel(ctx_accel, name)
            ctx_kernel == nothing ? _jwait(ctx_accel) : _jwait(ctx_kernel)
        end
    end

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

    delete_accel!(aname)
end
