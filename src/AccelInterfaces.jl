# Julia Accelerator Interfaces(Jai)


"""
Julia Accelerator Interfaces(Jai) main module
"""
module AccelInterfaces

#includes utilities
include("util.jl")

#includes common data types and variables
include("base.jl")

#includes common functions for framework interfaces
include("framework.jl")

#includes Jai KNL file handling functions
include("kernel.jl")

#includes functions for Fortran framework
include("fortran.jl")

#processes user API requests
include("main.jl")

#includes user interface macros
include("api.jl")

export @jaccel, @jenterdata, @jkernel, @jlaunch, @jexitdata, @jwait, @jdecel

end

# documentation example
#
#"""
#    bar(x[, y])
#
#Compute the Bar index between `x` and `y`.
#
#If `y` is unspecified, compute the Bar index between all pairs of columns of `x`.
#
## Arguments
#- `n::Integer`: the number of elements to compute.
#- `dim::Integer=1`: the dimensions along which to perform the computation.
#
#See also [`bar!`](@ref), [`baz`](@ref), [`baaz`](@ref).
#
## Examples
#```julia-repl
#julia> bar([1, 2], [1, 2])
#1
#```
#
#```jldoctest
#julia> a = [1 2; 3 4]
#2×2 Array{Int64,2}:
# 1  2
# 3  4
#[...]
#```
#
## Implementation
#For developers
#
#"""
#function bar(x, y) ...
