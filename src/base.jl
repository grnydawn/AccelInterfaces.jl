# base.jl: include common data types and variables

import Pkg.TOML
import DataStructures.OrderedDict

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
const JAI_TYPE_CONFIG_VALUE = Union{Vector{String}, String, Nothing}
const JAI_TYPE_CONFIG       = OrderedDict{JAI_TYPE_CONFIG_KEY, Union{
    JAI_TYPE_CONFIG_VALUE, OrderedDict{String, JAI_TYPE_CONFIG_VALUE}}}
const JAI_CONFIG_BLANK      = JAI_TYPE_CONFIG()

const JAI_ACCEL_CONFIGS = (
            ("pidfilename", ".jtask.pid"),
            ("debugdir",    nothing),
            ("workdir",     joinpath(pwd(), ".jworkdir"))
           ) :: NTuple{N, Tuple{JAI_TYPE_CONFIG_KEY, JAI_TYPE_CONFIG_VALUE}} where N

@enum JAI_TYPE_INOUT JAI_ARG_IN=1 JAI_ARG_OUT=2 JAI_ARG_INOUT=3 JAI_ARG_UNKNOWN=4

const JAI_TYPE_ARG = Tuple{JAI_TYPE_DATA, String, JAI_TYPE_INOUT, Tuple{<:Integer}}
const JAI_TYPE_ARGS = Vector{JAI_TYPE_ARG}

# Jai context
abstract type JAI_TYPE_CONTEXT end

# Jai host context
struct JAI_TYPE_CONTEXT_HOST <: JAI_TYPE_CONTEXT

    acfg ::OrderedDict{String, JAI_TYPE_CONFIG_VALUE}

    function JAI_TYPE_CONTEXT_HOST(acfg::JAI_TYPE_CONFIG)

        for (name, default) in JAI_ACCEL_CONFIGS
            if !haskey(acfg, name)
                acfg[name] = default
            end
        end

        new(acfg)
    end
end

# accelerator context
struct JAI_TYPE_CONTEXT_ACCEL <: JAI_TYPE_CONTEXT
    aname           ::String
    aid             ::UInt32
    prefix          ::String
    ctx_host        ::JAI_TYPE_CONTEXT_HOST
    const_vars      ::OrderedDict{String, JAI_TYPE_DATA}
    devices         ::NTuple{N, Integer} where N
    frame           ::JAI_TYPE_FRAMEWORK
    fslib           ::Ptr{Nothing}
    fconfig         ::JAI_TYPE_CONFIG_VALUE

    function JAI_TYPE_CONTEXT_ACCEL(
            aname       ::String,
            aid         ::UInt32,
            ctx_host    ::JAI_TYPE_CONTEXT_HOST,
            framework   ::JAI_TYPE_CONFIG,
            cvars       ::NTuple{N, JAI_TYPE_DATA} where N,
            cnames      ::NTuple{N, String} where N,
            device      ::NTuple{N, Integer} where N
        )

        # create directories
        workdir = ctx_host.acfg["workdir"]

        if !isdir(workdir)
            pidfile = joinpath(ctx_host.acfg["pidfilename"])
            locked_filetask(pidfile, workdir, mkdir, workdir)
        end

        const_vars = OrderedDict{String, JAI_TYPE_DATA}()
        for (name, var) in zip(cnames, cvars)
            const_vars[name] = var
        end

        prefix = join(["jai", aname, string(aid, base=16)], "_") * "_"

        # select data framework and generate a shared library for accel
        frame, fslib, fcfg = select_framework(framework, prefix, workdir)

        new(aname, aid, prefix, ctx_host, const_vars, device, frame, fslib, fcfg)
    end
end

# kernel context
struct JAI_TYPE_CONTEXT_KERNEL <: JAI_TYPE_CONTEXT
    kname           ::String
    accelctx        ::JAI_TYPE_CONTEXT_ACCEL

    function JAI_TYPE_CONTEXT_KERNEL(kname, actx)

        # generate kernel context id
        
        # load kernel definition

        new(kname, actx)
    end
end

