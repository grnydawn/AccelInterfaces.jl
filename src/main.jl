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
        framework   ::Union{JAI_TYPE_CONFIG, Nothing} = nothing,
        machine     ::Union{JAI_TYPE_CONFIG, Nothing} = nothing,
        compiler    ::Union{JAI_TYPE_CONFIG, Nothing} = nothing,
        set         ::Union{JAI_TYPE_CONFIG, Nothing} = nothing
    )

    # allow multiple calls anywhere
    println("CCCCC", lineno, filepath)

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
        const_vars  ::Union{NTuple{N, JAI_TYPE_DATA} where N, Nothing}= nothing,
        const_names ::Union{NTuple{N, String} where N, Nothing}       = nothing,
        device      ::Union{NTuple{N, Integer} where N, Nothing}      = nothing,
        framework   ::Union{JAI_TYPE_CONFIG, Nothing}   = nothing,
        machine     ::Union{JAI_TYPE_CONFIG, Nothing}   = nothing,
        compiler    ::Union{JAI_TYPE_CONFIG, Nothing}   = nothing,
        set         ::Union{JAI_TYPE_CONFIG, Nothing}   = nothing
    )

    if set != nothing
        set_config(set)
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

    cvars = OrderedDict{String, JAI_TYPE_DATA}()
    if const_names != nothing && const_vars != nothing
        for (name, var) in zip(const_names, const_vars)
            cvars[name] = var
        end
    end

    ctx_frame = select_framework(framework, compiler)
    ctx_kernels = Vector{JAI_TYPE_CONTEXT_KERNEL}()
    data_slibs = Dict{UInt32, Ptr{Nothing}}()

    ctx_accel = JAI_TYPE_CONTEXT_ACCEL(aname, aid, cvars, device, ctx_frame,
                                       data_slibs, ctx_kernels)

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
    uid         = generate_jid(ctx_accel.aid, apitype, apicount, lineno, filepath)
    frame       = ctx_accel.framework.type

    try
        if uid in keys(ctx_accel.data_slibs)
            slib = ctx_accel.data_slibs[uid]
        else
            prefix  = generate_prefix(aname, uid)
            compile = ctx_accel.framework.compile

            slib    = generate_sharedlib(frame, apitype, prefix, compile, args)

            ctx_accel.data_slibs[uid] = slib
        end

        funcname = prefix*JAI_MAP_API_FUNCNAME[apitype]
        invoke_sharedfunc(frame, slib, funcname, args)

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
       
    ctx_frame   = select_framework(ctx_accel, framework, compiler)
    launch_slibs= Dict{UInt32, Ptr{Nothing}}()

    ctx_kernel  = JAI_TYPE_CONTEXT_KERNEL(kid, kname, ctx_frame, launch_slibs, kdef)

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

    apitype     = JAI_LAUNCH
    args        = pack_args(innames, input, outnames, output)
    ctx_accel   = get_accel(aname)
    ctx_kernel  = get_kernel(ctx_accel, kname)
    uid         = generate_jid(ctx_kernel.kid, apitype, lineno, filepath)
    frame       = ctx_kernel.framework.type

    try
        if uid in keys(ctx_kernel.launch_slibs)
            slib    = ctx_kernel.launch_slibs[uid]
        else
            prefix  = generate_prefix(kname, uid)
            compile = ctx_kernel.framework.compile
            knlbody = get_knlbody(ctx_kernel)

            slib    = generate_sharedlib(frame, apitype, prefix, compile, args, knlbody)

            ctx_kernel.launch_slibs[uid] = slib
        end

        funcname = prefix*JAI_MAP_API_FUNCNAME[apitype]
        invoke_sharedfunc(frame, slib, funcname, args)

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
