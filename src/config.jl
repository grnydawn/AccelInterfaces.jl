# config.jl: include Jai configuration data and utilities

import Pkg.TOML

const DEBUG         = false # true
const JAI_VERSION   = TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"]
const _USER_CONFIGS = Tuple(string(n) for n in fieldnames(JAI_TYPE_CONFIG_USER))

# Jai global configurations
# frameworks: set from get_framework in @jkernel

const JAI = Dict(
        "config"        => JAI_TYPE_CONFIG_USER(
                                DEBUG,
                                joinpath(pwd(), ".jworkdir")
                           ),
        "pidfile"       => ".jtask.pid",
        "ctx_accels"    => Vector{JAI_TYPE_CONTEXT_ACCEL}(),
        "frameworks"    => OrderedDict{JAI_TYPE_FRAMEWORK,
                                       OrderedDict{UInt32, JAI_TYPE_CONTEXT_FRAMEWORK}}(),
        "machine"       => nothing
    )

function _find_accel(aname::String) ::Union{<:Integer, Nothing}
    ctxidx = nothing

    for idx in 1:length(JAI["ctx_accels"])
        if JAI["ctx_accels"][idx].aname == aname
            ctxidx = idx
            break
        end
    end

    return ctxidx
end

function get_config(CFG::JAI_TYPE_CONFIG_USER, name::String)

    if name in _USER_CONFIGS
        ret = getproperty(CFG, Symbol(name))

        if name == "workdir"
            if ret isa String && !isdir(ret)
                locked_filetask(JAI["pidfile"], ret, mkdir, ret)
            end
        end
    else
        ret = CFG[name]
    end

    return ret
end

function set_config(CFG::JAI_TYPE_CONFIG_USER, name::String, value)
    if name in _USER_CONFIGS
        type = fieldtype(JAI_TYPE_CONFIG_USER, Symbol(name))
        setproperty!(CFG, Symbol(name), value::type)
    else
        println(name * " can not be changed.")
    end
end


function set_config(CFG::JAI_TYPE_CONFIG_USER, configs::JAI_TYPE_CONFIG)
    for (key, value) in configs
        if key in _USER_CONFIGS
            set_config(CFG, key, value)
        end
    end
end

get_config(name::String)                = get_config(JAI["config"], name)
set_config(configs::JAI_TYPE_CONFIG)    = set_config(JAI["config"], configs)
set_config(name::String, value::Any)    = get_config(JAI["config"], name, value)

function get_config(ctx::JAI_TYPE_CONTEXT_ACCEL, name::String)
    value = get_config(ctx.config, name)
    if value == nothing
        value = get_config(name)
    end
    return value
end

function set_config(ctx::JAI_TYPE_CONTEXT_ACCEL, name::String, value::Any)
    get_config(ctx.config, name, value)
end

function set_config(ctx::JAI_TYPE_CONTEXT_ACCEL, configs::JAI_TYPE_CONFIG)
    set_config(ctx.config, configs)
end

function delete_accel!(aname)

    ctx_accel = nothing

    if aname == ""
        if length(JAI["ctx_accels"]) > 0
            ctx_accel = pop!(JAI["ctx_accels"])
        else
            println("WARNING: no accel context exists")
        end
    else
        ctxidx = _find_accel(aname)

        if ctxidx isa Integer
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

        # call fini
        #prefix = "jai_" * JAI_MAP_FRAMEWORK_STRING[ctx.frame] * "_accel_"
        # TODO: terminate ctx_accel
    end

end
