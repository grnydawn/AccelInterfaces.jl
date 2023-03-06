# main.jl: processes user requests from Jai API

"""
    function jai_accel

Process @jaccel macro

# Implementation
  * creates AccelContext and save it in a stack
  * select a framework for data transfer
  * save constant variables
  * configure accel parameters

"""

function jai_accel(
        aname       ::String,
        lineno      ::Integer,
        filepath    ::String;
        const_vars  ::NTuple{N, JAI_TYPE_DATA} where N   = (),
        const_names ::NTuple{N, String} where N         = (),
        framework   ::Dict{String, JAI_TYPE_CONFIG}= JAI_CONFIG_BLANK,
        device      ::NTuple{N, Integer} where N        = (),
        compile     ::NTuple{N, String} where N         = (),
        set         ::Dict{String, JAI_TYPE_CONFIG}= JAI_CONFIG_BLANK
    )

    ctx = JAI_TYPE_CONTEXT_ACCEL(aname)

    push!(_ctx_accels, ctx)

    gencode_accel(JAI_FORTRAN)
end

"""
    function jai_data

Process @jenterdata and @jexitdata macros

# Implementation
  * drive the generation of data transfer source files based on the framework selected at jai_accel
  * compile and load a shared library
  * invoke data transfer and memory allocation/deallocation functions
  * make aliases of device variables for inter-operability to kernels

"""

function jai_data(
        aname       ::String,
        apitype     ::JAI_TYPE_API,
        apicount    ::Integer,
        names       ::Vector{String},
        control     ::Vector{String},
        lineno      ::Integer,
        filepath    ::String,
        data        ::Vararg{JAI_TYPE_DATA, N} where N
    )
end


"""
    function jai_kernel

Process @jkernel

# Implementation
  * parse kernel file
  * select a framework for kernel
  * pre-process as much as possible to reduce kernel runtime

"""

function jai_kernel(
        kdef        ::Union{JAI_TYPE_KERNELDEF, String},
        kname       ::String,
        aname       ::String,
        lineno      ::Integer,
        filepath    ::String;
        framework   ::Dict{String, JAI_TYPE_CONFIG}= JAI_CONFIG_BLANK
    )

end


"""
    function jai_launch

Process @jlaunch

# Implementation
  * drive the generation of kernel source files
  * compile and load a shared library
  * launch kernel on device
"""

function jai_launch(
        kname       ::String,
        aname       ::String,
        input       ::NTuple{N, JAI_TYPE_DATA} where N,
        output      ::NTuple{N, JAI_TYPE_DATA} where N,
        innames     ::Vector{String},
        outnames    ::Vector{String},
        lineno      ::Integer,
        filepath    ::String,
        config      ::Dict{String, Union{Dict{String, JAI_TYPE_CONFIG}, Nothing}}
    )

end


"""
    function jai_wait

Process @jwait

# Implementation
  * invoke wait-equivalent framework api

"""

function jai_wait(
        aname       ::String,
        lineno      ::Integer,
        filepath    ::String
    )
end


"""
    function jai_decel

Process @jdecel macro

# Implementation
  * remove AccelContext

"""

function jai_decel(
        aname       ::String,
        lineno      ::Integer,
        filepath    ::String
    )

    if aname == ""
        if length(_ctx_accels) > 0
            pop!(_ctx_accels)
        else
            println("WARNING: no accel context exists")
        end
    else
        ctxidx = nothing

        for idx in range(1, length(_ctx_accels))
            if _ctx_accels[idx].aname == aname
                ctxidx = idx
                break
            end
        end

        if ctxidx isa Number
            deleteat!(_ctx_accels, ctxidx)
        else
            println("WARNING: no accel context name: " * aname)
        end
    end
end

