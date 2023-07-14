# main.jl: processes user requests from Jai API

"""
    function jai_config

Process @jconfig macro

# Implementation
  * handles application-wide configurations

"""

function jai_config(
        callsite    ::UInt32;
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
        callsite    ::UInt32;
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

    # TODO: apply accel specific config in set
    #
    aid = generate_jid(aname, Sys.STDLIB, JAI_VERSION, callsite)

    if aname == ""
        aname = string(aid, base = 16)
    end

    externs     = Dict{UInt64, String}()
    apitype     = JAI_ACCEL
    # pack const and variable names
    cvars = JAI_TYPE_ARGS()
    if const_names != nothing && const_vars != nothing
        for (n, v) in zip(const_names, const_vars)
            cvar = pack_arg(v, externs, apitype, name=n)
            push!(cvars, cvar)
        end
    end

    if config.workdir == nothing
        workdir = get_config("workdir")
        set_config(config, "workdir", workdir)
    else
        workdir = config.workdir
    end

    if config.cachedir == nothing
        cachedir = get_config("cachedir")
        if cachedir == nothing
            cachedir = joinpath(workdir, "cachedir")
        end
        set_config(config, "cachedir", cachedir)
    else
        cachedir = config.cachedir
    end
        
    if !isdir(cachedir)
        try
            locked_filetask("cachedir" * JAI["pidfile"], cachedir, mkdir, cachedir)
        catch e
        end
    end

    #ctx_framework   = select_framework(framework, compiler, workdir)
    data_framework  = Vector{JAI_TYPE_CONTEXT_FRAMEWORK}()
    ctx_kernels = Vector{JAI_TYPE_CONTEXT_KERNEL}()
    data_slibs  = Dict{UInt32, PtrAny}()
    difftest    = Vector{Dict{String, Any}}()
    devices     = Dict{Integer, Bool}()

    if device != nothing
        for d in device
            devices[d] = false
        end
    end

    ctx_accel = JAI_TYPE_CONTEXT_ACCEL(aname, aid, config, cvars, devices,
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
        callsite    ::UInt32;
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
    kid = generate_jid(ctx_accel.aid, kname, kdef.kdid, callsite)
 
    if kname == ""
        kname = string(kid, base = 16)
    end
       
    workdir     = get_config(ctx_accel, "workdir")
    cachedir    = get_config(ctx_accel, "cachedir")

    ctx_frames = Vector{JAI_TYPE_CONTEXT_FRAMEWORK}()

    if framework isa JAI_TYPE_CONFIG
        for (fkey, fval) in framework
            if fkey in kdef_frames
				try
					frame = get_framework(fkey, fval, ctx_accel.devices,
                                    compiler, workdir, cachedir)
					if frame isa JAI_TYPE_CONTEXT_FRAMEWORK
						push!(ctx_frames, frame)
					end

				catch err
					if err isa Base.IOError

					elseif err isa JAI_ERROR_COMPILE_NOSHAREDLIB

					else
						rethrow(err)
					end
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
        clauses     ::JAI_TYPE_CONFIG,
        callsite    ::UInt32,
        data        ::Vararg{JAI_TYPE_DATA, N} where N
    )

    if clauses != nothing
        if "enable_if" in keys(clauses) && !clauses["enable_if"]
            return
        end
    end 

    ctx_accel   = get_accel(aname)

    # pack data and variable names
	varitems = []
    args = JAI_TYPE_ARGS()
    for (i, (n, d)) in enumerate(zip(names, data))
        arg = pack_arg(d, ctx_accel.externs, apitype, name=n)
        push!(args, arg)
		push!(varitems, arg[2]); push!(varitems, arg[6])
		push!(varitems, arg[7]); push!(varitems, arg[8])
    end

    if length(ctx_accel.data_framework) > 0
        ctx_frame = ctx_accel.data_framework[1]
    else
        ctx_frame = select_data_framework(ctx_accel)
    end

    data_frametype = ctx_frame.type
    data_compile = ctx_frame.compile

    uid         = generate_jid(ctx_accel.aid, apitype, apicount, callsite,
                                data_frametype, data_compile, varitems)
    prefix      = generate_prefix(aname, uid)

    try
        if uid in keys(ctx_accel.data_slibs)
            slib = ctx_accel.data_slibs[uid]

        else
            #compile = ctx_accel.framework.compile
            workdir = get_config(ctx_accel, "workdir")
            cachedir = get_config(ctx_accel, "cachedir")

            #data_frametype, data_compile = select_data_framework(ctx_accel)

            slib    = generate_sharedlib(data_frametype, apitype,
                        prefix, data_compile, workdir, cachedir,
                        ctx_accel.const_vars, args, clauses)

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
        frame_config::JAI_TYPE_CONFIG,
        clauses     ::JAI_TYPE_CONFIG,
        callsite    ::UInt32
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

    for (key, value) in frame_config

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

	varitems = []
    for arg in args
		push!(varitems, arg[2]); push!(varitems, arg[6])
		push!(varitems, arg[7]); push!(varitems, arg[8])
    end

    uid         = generate_jid(ctx_kernel.kid, apitype, callsite,
                                varitems, frametype, compile)
    prefix      = generate_prefix(kname, uid)

    try
        if uid in keys(ctx_kernel.launch_slibs)
            slib    = ctx_kernel.launch_slibs[uid]
        else
            workdir = get_config(ctx_accel, "workdir")
            cachedir = get_config(ctx_accel, "cachedir")
            knlcode = get_kernel_code(ctx_kernel, frametype)
            difftest = (length(ctx_accel.difftest) > 0
                       ) ? ctx_accel.difftest[end] : nothing

            #data_frametype, data_compile = select_data_framework(ctx_accel)

            slib    = generate_sharedlib(frametype, apitype,
                        prefix, compile, workdir, cachedir, ctx_accel.const_vars,
                        args, clauses, knlcode, launch_config=frame_config,
                        difftest=difftest)

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
function jai_wait(
        name       ::String,
        callsite    ::UInt32
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
        callsite    ::UInt32
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
        callsite    ::UInt32
    )

    println("Starting DIFF test for $aname $cases")

    ctx_accel   = get_accel(aname)

end


function _jai_diffA(
        aname       ::String,
        cases       ::Tuple{String, String},
        callsite    ::UInt32
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
        callsite    ::UInt32
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
        callsite    ::UInt32
    )

    println("Generate diff report for $(cases)")

    ctx_accel   = get_accel(aname)

    deleteat!(ctx_accel.difftest, 1:length(ctx_accel.difftest))

end


