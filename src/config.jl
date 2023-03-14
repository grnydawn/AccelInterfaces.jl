# config.jl: include Jai configuration data and utilities

import Pkg.TOML
const JAI_VERSION = TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"]

# Jai global configurations
const JAI = Dict(
        "debug"         => true,
        "pidfile"       => ".jtask.pid",
        "workdir"       => joinpath(pwd(), ".jworkdir"),
        "debugdir"      => nothing,
        "ctx_accels"    => Vector{JAI_TYPE_CONTEXT_ACCEL}(),
        "frameworks"    => OrderedDict{JAI_TYPE_FRAMEWORK,
                                       OrderedDict{UInt32, JAI_TYPE_CONTEXT_FRAMEWORK}}()
    )

const _USER_CONFIGS = (
        "debug", "workdir", "debugdir"
    )

function get_config(name::String)

    if name == "workdir"
        if !isdir(JAI["workdir"])
            workdir = JAI["workdir"]
            locked_filetask(JAI["pidfile"], workdir, mkdir, workdir)
        end
        ret = JAI["workdir"]
    else
        ret = JAI[name]
    end

    return ret

end

function set_config(name::String, value)
    if name in _USER_CONFIGS
        JAI[name] = value
    else
        println(name * " can not be changed.")
    end
end


function set_config(configs::JAI_TYPE_CONFIG)
    for (key, value) in configs
        if key in _USER_CONFIGS
            set_config(key, value)
        end
    end
end

function get_sharedlib(jid::UInt32)
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

        # call fini
        #prefix = "jai_" * JAI_MAP_FRAMEWORK_STRING[ctx.frame] * "_accel_"
        # TODO: terminate ctx_accel
    end

end
