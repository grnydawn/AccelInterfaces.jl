# base.jl: include common data types and variables

import InteractiveUtils.subtypes

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

const JAI_FORTRAN                   = JAI_TYPE_FORTRAN()
const JAI_FORTRAN_OPENACC           = JAI_TYPE_FORTRAN_OPENACC()
const JAI_FORTRAN_OMPTARGET         = JAI_TYPE_FORTRAN_OMPTARGET()
const JAI_CPP                       = JAI_TYPE_CPP()
const JAI_CPP_OPENACC               = JAI_TYPE_CPP_OPENACC()
const JAI_CPP_OMPTARGET             = JAI_TYPE_CPP_OMPTARGET()
const JAI_CUDA                      = JAI_TYPE_CUDA()
const JAI_HIP                       = JAI_TYPE_HIP()

const JAI_SYMBOL_FRAMEWORKS = map(
        (x) -> Symbol(extract_name_from_frametype(x)),
        subtypes(JAI_TYPE_FRAMEWORK)
    )

# Jai data types
const JAI_TYPE_DATA  =   Union{T, String, NTuple{N, T}, AbstractArray{T, N}
                             } where {N, T<:Number}

# Jai context
abstract type JAI_TYPE_CONTEXT end

# Jai host context
struct JAI_TYPE_CONTEXT_HOST <: JAI_TYPE_CONTEXT
end

# accelerator context
struct JAI_TYPE_CONTEXT_ACCEL <: JAI_TYPE_CONTEXT
    aname           ::String
    hostctx         ::JAI_TYPE_CONTEXT_HOST

    function JAI_TYPE_CONTEXT_ACCEL(aname)
        new(aname, _ctx_host)
    end
end

# kernel context
struct JAI_TYPE_CONTEXT_KERNEL <: JAI_TYPE_CONTEXT
    kname           ::String
    accelctx        ::JAI_TYPE_CONTEXT_ACCEL

    function JAI_TYPE_CONTEXT_KERNEL(kname, actx)
        new(kname, actx)
    end
end

# host context object
# common to all threads
# no change during a session
const _ctx_host = JAI_TYPE_CONTEXT_HOST()

# accelerator context cache
const _ctx_accels = Vector{JAI_TYPE_CONTEXT_ACCEL}()

# kernel context cache
const _ctx_kernels = Vector{JAI_TYPE_CONTEXT_KERNEL}()


struct JAI_TYPE_KERNELDEF
end

const JAI_TYPE_CONFIG  = Union{String, Nothing}

const JAI_CONFIG_BLANK  = Dict{String, JAI_TYPE_CONFIG}()
