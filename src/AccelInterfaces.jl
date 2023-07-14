# Julia Accelerator Interfaces(Jai)


"""
Julia Accelerator Interfaces(Jai) module
"""
module AccelInterfaces

include("type.jl")
include("error.jl")
include("util.jl")
include("config.jl")
include("framework.jl")
include("kernel.jl")
include("main.jl")
include("api.jl")

export @jconfig, @jaccel, @jenterdata, @jkernel, @jlaunch, @jexitdata, @jwait, @jdecel, @jdiff

function _finalize()
end

function __init__()
    #global sharedlib_cache = @get_scratch!("sharedlib_files")
    atexit(_finalize)
end

end

