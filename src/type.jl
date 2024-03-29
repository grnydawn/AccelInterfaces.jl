# base.jl: include common data types and variables

import DataStructures.OrderedDict
import OffsetArrays.OffsetArray

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
struct JAI_TYPE_NULLAPI     <: JAI_TYPE_API end

const JAI_TYPE_API_DATA = Union{JAI_TYPE_ALLOCATE, JAI_TYPE_DEALLOCATE,
                                JAI_TYPE_UPDATETO, JAI_TYPE_UPDATEFROM}

const JAI_ACCEL             = JAI_TYPE_ACCEL()
const JAI_KERNEL            = JAI_TYPE_KERNEL()
const JAI_ALLOCATE          = JAI_TYPE_ALLOCATE()
const JAI_DEALLOCATE        = JAI_TYPE_DEALLOCATE()
const JAI_UPDATETO          = JAI_TYPE_UPDATETO()
const JAI_UPDATEFROM        = JAI_TYPE_UPDATEFROM()
const JAI_LAUNCH            = JAI_TYPE_LAUNCH()
const JAI_WAIT              = JAI_TYPE_WAIT()
const JAI_NULLAPI           = JAI_TYPE_NULLAPI()

# Jai data types
const PtrAny    = Ptr{Any}
const JAI_TYPE_DATA = Union{Nothing, String, Bool, T, NTuple{N, T}, AbstractArray{T, N},
                            AbstractArray{PtrAny, N}} where {N, T<:Number}

# Jai Framework types
abstract type JAI_TYPE_FRAMEWORK end

struct JAI_TYPE_FORTRAN             <: JAI_TYPE_FRAMEWORK end
struct JAI_TYPE_FORTRAN_OPENACC     <: JAI_TYPE_FRAMEWORK end
struct JAI_TYPE_FORTRAN_OMPTARGET   <: JAI_TYPE_FRAMEWORK end
struct JAI_TYPE_CPP                 <: JAI_TYPE_FRAMEWORK end
struct JAI_TYPE_CPP_OPENACC         <: JAI_TYPE_FRAMEWORK end
struct JAI_TYPE_CPP_OMPTARGET       <: JAI_TYPE_FRAMEWORK end
struct JAI_TYPE_CUDA                <: JAI_TYPE_FRAMEWORK end
struct JAI_TYPE_HIP                 <: JAI_TYPE_FRAMEWORK end

const JAI_TYPE_CONFIG_KEY   = Union{JAI_TYPE_FRAMEWORK, String}
const JAI_TYPE_CONFIG_VALUE = Union{
                                    <:JAI_TYPE_DATA,
                                    Dict{Symbol, Any},
                                    Dict{String, <:JAI_TYPE_DATA},
                                    NTuple{N, Union{JAI_TYPE_DATA, NTuple{N, JAI_TYPE_DATA}}} where N
                                }
const JAI_TYPE_CONFIG       = OrderedDict{
                                    JAI_TYPE_CONFIG_KEY,
                                    Union{
                                        JAI_TYPE_CONFIG_VALUE,
                                        OrderedDict{String, JAI_TYPE_CONFIG_VALUE}
                                    }
                                }

# Jai user config type
mutable struct JAI_TYPE_CONFIG_USER
    debug       ::Union{Bool, Nothing}
    workdir     ::Union{String, Nothing}
    cachedir    ::Union{String, Nothing}
    debugdir    ::Union{String, Nothing}

    function JAI_TYPE_CONFIG_USER()
        new(nothing, nothing, nothing, nothing)
    end

    function JAI_TYPE_CONFIG_USER(debug, workdir, cachedir, debugdir)
        new(debug, workdir, cachedir, debugdir)
    end

end

@enum JAI_TYPE_INOUT JAI_ARG_IN=1 JAI_ARG_OUT=2 JAI_ARG_INOUT=3 JAI_ARG_UNKNOWN=4

const JAI_TYPE_ARG = Tuple{ JAI_TYPE_DATA,
                            DataType,
                            String,
                            JAI_TYPE_INOUT,
                            Int64,
                            NTuple{N, T} where {N, T<:Integer},
                            NTuple{N, T} where {N, T<:Integer},
                            String
                        }
const JAI_TYPE_ARGS = Vector{JAI_TYPE_ARG}

const JAI_MAP_APITYPE_INOUT = Dict{JAI_TYPE_API, JAI_TYPE_INOUT}(
        JAI_ACCEL       => JAI_ARG_IN,
        JAI_KERNEL      => JAI_ARG_IN,
        JAI_ALLOCATE    => JAI_ARG_IN,
        JAI_DEALLOCATE  => JAI_ARG_IN,
        JAI_UPDATETO    => JAI_ARG_IN,
        JAI_UPDATEFROM  => JAI_ARG_INOUT,
        JAI_LAUNCH      => JAI_ARG_IN,
        JAI_WAIT        => JAI_ARG_IN
    )


# Jai context
abstract type JAI_TYPE_CONTEXT end

# Jai framework context
struct JAI_TYPE_CONTEXT_FRAMEWORK <: JAI_TYPE_CONTEXT
    type    ::JAI_TYPE_FRAMEWORK
    slib    ::Ptr{Nothing}
    compile ::String
    prefix  ::String
end

struct JAI_TYPE_KERNELHDR
    frame   ::Union{JAI_TYPE_FRAMEWORK, Symbol}
    argnames::Vector{String}
    params  ::Union{Expr, Nothing}
end

struct JAI_TYPE_KERNELBODY
    body    ::String
end

struct JAI_TYPE_KERNELINITSEC
    ksid    ::UInt32
    argnames::Vector{String}
    env     ::Module
end

struct JAI_TYPE_KERNELSEC
    ksid    ::UInt32
    header  ::JAI_TYPE_KERNELHDR
    body    ::JAI_TYPE_KERNELBODY
end

struct JAI_TYPE_KERNELDEF
    kdid    ::UInt32
    doc     ::String
    init    ::Vector{JAI_TYPE_KERNELINITSEC}
    sections::Vector{JAI_TYPE_KERNELSEC}
end

# kernel context
struct JAI_TYPE_CONTEXT_KERNEL <: JAI_TYPE_CONTEXT
    kid             ::UInt32
    kname           ::String
    frameworks      ::Vector{JAI_TYPE_CONTEXT_FRAMEWORK}
    launch_slibs    ::Dict{UInt32, Ptr{Nothing}}
    kdef            ::JAI_TYPE_KERNELDEF
end

# accelerator context
struct JAI_TYPE_CONTEXT_ACCEL <: JAI_TYPE_CONTEXT
    aname           ::String
    aid             ::UInt32
    config          ::JAI_TYPE_CONFIG_USER
    const_vars      ::JAI_TYPE_ARGS
    devices         ::Dict{Integer, Bool}
    data_framework  ::Vector{JAI_TYPE_CONTEXT_FRAMEWORK}
    data_slibs      ::Dict{UInt32, Ptr{Nothing}}
    ctx_kernels     ::Vector{JAI_TYPE_CONTEXT_KERNEL}
    externs         ::Dict{UInt64, String}
    difftest        ::Vector{Dict{String, Any}}
    framework       ::Union{JAI_TYPE_CONFIG, Nothing}
end

struct JAI_TYPE_OS
    name    ::String
end

struct JAI_TYPE_MACHINE
    desc    ::String
    modules ::Vector{String}
end
