# Julia Accelerator Interfaces(Jai)


"""
Julia Accelerator Interfaces(Jai) main module
"""
module AccelInterfaces

include("base.jl")
include("util.jl")
include("error.jl")
include("framework.jl")
include("kernel.jl")
include("fortran.jl")
include("main.jl")
include("api.jl")

export @jaccel, @jenterdata, @jkernel, @jlaunch, @jexitdata, @jwait, @jdecel

end

