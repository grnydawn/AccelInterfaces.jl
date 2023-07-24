# Julia Accelerator Interfaces(Jai)


"""
Julia Accelerator Interfaces (Jai) is a Julia package to reuse large-scale simulations written in Fortran/C/C++.
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

export @jaccel, @jenterdata, @jkernel, @jlaunch, @jexitdata, @jwait, @jdecel
#export @jconfig, @jaccel, @jenterdata, @jkernel, @jlaunch, @jexitdata, @jwait, @jdecel, @jdiff

function _finalize()
end

function __init__()
    #global sharedlib_cache = @get_scratch!("sharedlib_files")
    atexit(_finalize)
end

end

