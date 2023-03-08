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

const JAI_TYPE_CONFIG  = Union{Vector{String}, String, Nothing}
const JAI_CONFIG_BLANK  = Dict{String, JAI_TYPE_CONFIG}()
const JAI_ACCEL_CONFIGS = (
            ("pidfilename", ".jtask.pid"),
            ("debugdir",    nothing),
            ("workdir",     joinpath(pwd(), ".jworkdir"))
        )

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

# Jai context
abstract type JAI_TYPE_CONTEXT end

# Jai host context
struct JAI_TYPE_CONTEXT_HOST <: JAI_TYPE_CONTEXT

    acfg ::Dict{String, JAI_TYPE_CONFIG}

    function JAI_TYPE_CONTEXT_HOST(acfg::Dict{String, JAI_TYPE_CONFIG})

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
    ctx_host        ::JAI_TYPE_CONTEXT_HOST
    const_vars      ::Dict{String, JAI_TYPE_DATA}
    devices         ::NTuple{N, Integer} where N

    function JAI_TYPE_CONTEXT_ACCEL(aname, aid, ctx_host, framework,
                                    cvars, cnames, device)

        # create directories
        workdir = ctx_host.acfg["workdir"]

        if !isdir(workdir)
            pidfile = joinpath(ctx_host.acfg["pidfilename"])
            locked_filetask(pidfile, workdir, mkdir, workdir)
        end

        const_vars = Dict{String, JAI_TYPE_DATA}()
        for (name, var) in zip(cnames, cvars)
            const_vars[name] = var
        end

        # select data framework and generate a shared library for accel
        fname, fslib = select_framework(framework, aname)

        # init device and gather device information

        new(aname, aid, ctx_host, const_vars, device)
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

