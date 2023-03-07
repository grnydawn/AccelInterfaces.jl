# base.jl: include common data types and variables

import Pkg.TOML

const JAI_VERSION = TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"]

# Jai API types
abstract type JAI_TYPE_API end

struct JAI_TYPE_ACCEL       <: JAI_TYPE_API end
struct JAI_TYPE_KERNEL      <: JAI_TYPE_API end
struct JAI_TYPE_ALLOCATE    <: JAI_TYPE_API end
struct JAI_TYPE_DEALLOCATE  <: JAI_TYPE_API end
struct JAI_TYPE_UPDATETO    <: JAI_TYPE_API end
struct JAI_TYPE_UPDATEFROM  <: JAI_TYPE_API end
struct JAI_TYPE_LAUNCH      <: JAI_TYPE_API end
struct JAI_TYPE_WAIT        <: JAI_TYPE_API end

const JAI_ACCEL             = JAI_TYPE_ACCEL()
const JAI_KERNEL            = JAI_TYPE_KERNEL()
const JAI_ALLOCATE          = JAI_TYPE_ALLOCATE()
const JAI_DEALLOCATE        = JAI_TYPE_DEALLOCATE()
const JAI_UPDATETO          = JAI_TYPE_UPDATETO()
const JAI_UPDATEFROM        = JAI_TYPE_UPDATEFROM()
const JAI_LAUNCH            = JAI_TYPE_LAUNCH()
const JAI_WAIT              = JAI_TYPE_WAIT()

# Jai data types
const JAI_TYPE_DATA  =   Union{T, String, NTuple{N, T}, AbstractArray{T, N}
                             } where {N, T<:Number}

const JAI_TYPE_CONFIG  = Union{String, Nothing}
const JAI_CONFIG_BLANK  = Dict{String, JAI_TYPE_CONFIG}()
const JAI_ACCEL_CONFIGS = (
            ("pidfilename", ".jtask.pid"),
            ("debugdir",    nothing),
            ("workdir",     joinpath(pwd(), ".jworkdir"))
        )
