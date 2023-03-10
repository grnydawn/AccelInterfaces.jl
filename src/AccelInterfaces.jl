# Julia Accelerator Interfaces(Jai)


"""
Julia Accelerator Interfaces(Jai) module
"""
module AccelInterfaces

include("base.jl")
include("error.jl")
include("util.jl")
include("framework.jl")
include("kernel.jl")
include("fortran.jl")
include("main.jl")
include("api.jl")

export @jconfig, @jaccel, @jenterdata, @jkernel, @jlaunch, @jexitdata, @jwait, @jdecel

end

