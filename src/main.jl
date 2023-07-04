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

    externs     = Dict{String, String}()
    apitype     = JAI_ACCEL
    # pack const and variable names
    cvars = JAI_TYPE_ARGS()
    if const_names != nothing && const_vars != nothing
        for (n, v) in zip(const_names, const_vars)
            cvar = pack_arg(v, externs, apitype, name=n)
            push!(cvars, cvar)
        end
    end

    workdir     = config.workdir
    if workdir == nothing
        workdir = get_config("workdir")
    end

    #ctx_framework   = select_framework(framework, compiler, workdir)
    data_framework  = Vector{JAI_TYPE_CONTEXT_FRAMEWORK}()
    ctx_kernels = Vector{JAI_TYPE_CONTEXT_KERNEL}()
    data_slibs  = Dict{UInt32, PtrAny}()
    difftest    = Vector{Dict{String, Any}}()

    ctx_accel = JAI_TYPE_CONTEXT_ACCEL(aname, aid, config, cvars, device,
                        data_framework, data_slibs, ctx_kernels, externs,
                        difftest)

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

    kdef_frames = Vector{JAI_TYPE_FRAMEWORK}()
    for sec in kdef.sections
        push!(kdef_frames, sec.header.frame)
    end
    # generate kernel context id
    kid = generate_jid(ctx_accel.aid, kname, kdef.kdid, lineno, filepath)
 
    if kname == ""
        kname = string(kid, base = 16)
    end
       
    workdir     = get_config(ctx_accel, "workdir")
    ctx_frames = Vector{JAI_TYPE_CONTEXT_FRAMEWORK}()

    if framework isa JAI_TYPE_CONFIG
        for (fkey, fval) in framework
            if fkey in kdef_frames
                frame = get_framework(fkey, fval, compiler, workdir)
                if frame isa JAI_TYPE_CONTEXT_FRAMEWORK
                    push!(ctx_frames, frame)
                end
            end
        end
    else
        knlframes = get_kernel_frameworks(kdef)
        error("Default compilers are not defined yet.")
        # TODO : get compiler for knl frame from default compilers
    end

    #ctx_frame   = select_framework(ctx_accel, framework, compiler, workdir)
    launch_slibs= Dict{UInt32, PtrAny}()

    ctx_kernel  = JAI_TYPE_CONTEXT_KERNEL(kid, kname, ctx_frames, launch_slibs, kdef)

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
        config      ::Union{JAI_TYPE_CONFIG, Nothing},
        control     ::Vector{String},
        lineno      ::Integer,
        filepath    ::String,
        data        ::Vararg{JAI_TYPE_DATA, N} where N
    )

    if config != nothing
        if "enable_if" in keys(config) && !config["enable_if"]
            return
        end
    end 

    ctx_accel   = get_accel(aname)

    # pack data and variable names
    extnames = Vector{String}()
    args = JAI_TYPE_ARGS()
    for (i, (n, d)) in enumerate(zip(names, data))
        arg = pack_arg(d, ctx_accel.externs, apitype, name=n)
        push!(args, arg)
        push!(extnames, arg[8]*string(arg[2])*string(arg[6])*string(arg[7]))
    end

    if length(ctx_accel.data_framework) > 0
        ctx_frame = ctx_accel.data_framework[1]
    else
        ctx_frame = select_data_framework(ctx_accel)
    end

    data_frametype = ctx_frame.type
    data_compile = ctx_frame.compile

    extnameid   = join(extnames, "")
    uid         = generate_jid(ctx_accel.aid, apitype, apicount, lineno,
                                data_frametype, data_compile, filepath, extnameid)
    prefix      = generate_prefix(aname, uid)

    try
        if uid in keys(ctx_accel.data_slibs)
            slib = ctx_accel.data_slibs[uid]

        else
            #compile = ctx_accel.framework.compile
            workdir = get_config(ctx_accel, "workdir")

            #data_frametype, data_compile = select_data_framework(ctx_accel)

            slib    = generate_sharedlib(data_frametype, apitype, 
                        prefix, data_compile, workdir, ctx_accel.const_vars, args)

            ctx_accel.data_slibs[uid] = slib
        end

        funcname = prefix*JAI_MAP_API_FUNCNAME[apitype]
        #invoke_sharedfunc(frametype, slib, funcname, args)
        invoke_sharedfunc(data_frametype, slib, funcname, args)

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
    ctx_accel   = get_accel(aname)
    ctx_kernel  = get_kernel(ctx_accel, kname)

    args        = pack_args(innames, input, outnames, output,
                            ctx_accel.externs, apitype)

    # select a framework based on config and frameworks
    frametype = nothing
    compile = nothing

    disables = Vector{JAI_TYPE_FRAMEWORK}()

    for (key, value) in config

        if value != nothing && "enable_if" in keys(value) && !value["enable_if"]
            push!(disables, key)
            continue
        end

        for ctx_frame in ctx_kernel.frameworks
            if key == ctx_frame.type
                frametype = ctx_frame.type
                compile = ctx_frame.compile
                break
            end
        end

        if frametype != nothing
            break
        end
    end
 
    if frametype == nothing
        for ctx_frame in ctx_kernel.frameworks
            if ctx_frame.type in disables
                continue
            end 

            frametype = ctx_frame.type
            compile = ctx_frame.compile
            break
        end
    end

    extnames    = Vector{String}()
    for arg in args
        push!(extnames, arg[end])
    end

    extnameid   = join(extnames, "")
    uid         = generate_jid(ctx_kernel.kid, apitype, lineno, filepath, extnameid, frametype, compile)
    prefix      = generate_prefix(kname, uid)

    try
        if uid in keys(ctx_kernel.launch_slibs)
            slib    = ctx_kernel.launch_slibs[uid]
        else
            workdir = get_config(ctx_accel, "workdir")
            knlcode = get_kernel_code(ctx_kernel, frametype)
            difftest = (length(ctx_accel.difftest) > 0
                       ) ? ctx_accel.difftest[end] : nothing

            #data_frametype, data_compile = select_data_framework(ctx_accel)

            slib    = generate_sharedlib(frametype, apitype,
                        prefix, compile, workdir, ctx_accel.const_vars, args, knlcode,
                        launch_config=config, difftest=difftest)

            ctx_kernel.launch_slibs[uid] = slib
        end

        funcname = prefix*JAI_MAP_API_FUNCNAME[apitype]
        invoke_sharedfunc(frametype, slib, funcname, args)

        # support @jdiff
        # call jexitdata

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
#function jai_wait(
#        name       ::String,
#        lineno      ::Integer,
#        filepath    ::String
#    )
#
#    ctx = nothing
#    slib = nothing
#
#    if name == ""
#        ctx_accel   = get_accel(name)
#        ctx_kernel  = get_kernel(ctx_accel, name)
#        ctx_kernel == nothing ? _jwait(ctx_accel) : _jwait(ctx_kernel)
#    else
#        ctx_accel   = get_accel(name)
#        if ctx_accel == nothing
#            ctx_kernel = nothing
#            for ctx_accel in JAI["ctx_accels"]
#                ctx_kernel = get_kernel(ctx_accel, name)
#                if ctx_kernel != nothing
#                    _jwait(ctx_kernel)
#                    break
#                end
#            end
#
#            if ctx_kernel == nothing
#                error("Can not find shared library for jwait with " * name)
#            end
#        else
#            ctx_kernel   = get_kernel(ctx_accel, name)
#            ctx_kernel == nothing ? _jwait(ctx_accel) : _jwait(ctx_kernel)
#        end
#    end
#
#end

function jai_wait(
        name       ::String,
        lineno      ::Integer,
        filepath    ::String
    )

    ctx_accel   = get_accel(name)

    if length(ctx_accel.data_framework) > 0

        args = JAI_TYPE_ARGS()
        push!(args, pack_arg(fill(Int64(-1), 1), nothing, nothing))

        framework =  ctx_accel.data_framework[1]
        frame   = framework.type
        slib    = framework.slib

        funcname = framework.prefix * JAI_MAP_API_FUNCNAME[JAI_WAIT]
        invoke_sharedfunc(frame, slib, funcname, args)

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


"""
    function jai_diff

Process @jdiff macro

# Implementation
  * compare kernel implementations

"""

function jai_diff(
        aname       ::String,
        cases       ::Tuple{String, String},
        lineno      ::Integer,
        filepath    ::String
    )

    println("Starting DIFF test for $aname $cases")

    ctx_accel   = get_accel(aname)

end


function _jai_diffA(
        aname       ::String,
        cases       ::Tuple{String, String},
        lineno      ::Integer,
        filepath    ::String
    )

    case = cases[1]

    println("Begin test for $case")

    c1 = Dict{String, Any}()
    c1["name"] = case
    c1["test"] = "sum"

    ctx_accel   = get_accel(aname)
    push!(ctx_accel.difftest, c1)

end


function _jai_diffB(
        aname       ::String,
        cases       ::Tuple{String, String},
        lineno      ::Integer,
        filepath    ::String
    )

    case = cases[2]

    println("Begin test for $case")

    c2 = Dict{String, Any}()
    c2["name"] = case
    c2["test"] = "sum"

    ctx_accel   = get_accel(aname)
    push!(ctx_accel.difftest, c2)

end


function _jai_diffend(
        aname       ::String,
        cases       ::Tuple{String, String},
        lineno      ::Integer,
        filepath    ::String
    )

    println("Generate diff report for $(cases)")

    ctx_accel   = get_accel(aname)

    deleteat!(ctx_accel.difftest, 1:length(ctx_accel.difftest))

end


