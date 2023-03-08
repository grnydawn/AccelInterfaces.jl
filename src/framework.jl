# framework.jl: implement common functions for framework interfaces

import InteractiveUtils.subtypes

const JAI_FORTRAN                   = JAI_TYPE_FORTRAN()
const JAI_FORTRAN_OPENACC           = JAI_TYPE_FORTRAN_OPENACC()
const JAI_FORTRAN_OMPTARGET         = JAI_TYPE_FORTRAN_OMPTARGET()
const JAI_CPP                       = JAI_TYPE_CPP()
const JAI_CPP_OPENACC               = JAI_TYPE_CPP_OPENACC()
const JAI_CPP_OMPTARGET             = JAI_TYPE_CPP_OMPTARGET()
const JAI_CUDA                      = JAI_TYPE_CUDA()
const JAI_HIP                       = JAI_TYPE_HIP()

# Jai supported frameworks with priority
const JAI_SUPPORTED_FRAMEWORKS      = (
        JAI_CUDA,
        JAI_HIP,
        JAI_FORTRAN_OMPTARGET,
        JAI_FORTRAN_OPENACC,
        JAI_CPP_OMPTARGET,
        JAI_CPP_OPENACC,
        JAI_FORTRAN,
        JAI_CPP
    )

const JAI_AVAILABLE_FRAMEWORKS      = Vector{JAI_TYPE_FRAMEWORK}()

const JAI_SYMBOL_FRAMEWORKS = map(
        (x) -> Symbol(extract_name_from_frametype(x)),
        subtypes(JAI_TYPE_FRAMEWORK)
    )

const _ccall_cache = Dict()
const _dummyargs = ["a[$i]" for i in range(1, stop=200)]
const _ccs = ("(f,a) -> ccall(f, Int64, (", ",), " , ")")

function jai_ccall(dtypestr::String, libfunc::Ptr{Nothing}, args) :: Int64

    funcstr = _ccs[1] * dtypestr * _ccs[2] * join(_dummyargs[1:length(args)], ",") * _ccs[3]
    fid = read(IOBuffer(sha1(funcstr)[1:8]), Int64)

    if ! haskey(_ccall_cache, fid)
        _ccall_cache[fid] = eval(Meta.parse(funcstr))
    end

    Base.invokelatest(_ccall_cache[fid], libfunc, args)
end


"""
    function genslib_accel(ftype, actx)

Generate framework shared library to drive device.

Using the generated shared library from this function, Jai will access and configure the device.

# Arguments
- `ftype`::JAI_TYPE_FRAMEWORK: a framework type identifier
- `actx`::JAI_TYPE_CONTEXT_ACCEL: a accel context

See also [`@jaccel`](jaccel), [`genslib_kernel`](genslib_kernel), [`genslib_data`](genslib_data)

# Examples
```julia-repl
julia> genslib_accel(JAI_TYPE_FORTRAN, actx)
```

# Implementation
T.B.D.

"""
function genslib_accel(
        frame       ::JAI_TYPE_FRAMEWORK,
        actx        ::JAI_TYPE_CONTEXT_ACCEL
    ) :: String

    throw(JAI_ERROR_NOTIMPLEMENTED_FRAMEWORK(frame))
end


"""
    function genslib_data(ftype)

Generate framework shared library to move data between host and device.

Using the generated shared library from this function, Jai will move data between host and device

# Arguments
- `ftype`::JAI_TYPE_FRAMEWORK: a framework type identifier

See also [`@jenterdata`](jenterdata), [`@jexitdata`](jexitdata), [`genslib_kernel`](genslib_kernel), [`genslib_accel`](genslib_accel)

# Examples
```julia-repl
julia> genslib_data(JAI_TYPE_FORTRAN)
```

# Implementation
T.B.D.

"""
function genslib_data(
        frame       ::JAI_TYPE_FRAMEWORK
    ) :: String

    error("ERROR: Framework-$(extract_name_from_frametype(typeof(frame))) " *
          "should implement 'genslib_data' function.")

end

"""
    function genslib_kernel(ftype)

Generate framework shared library to launch a kernel on device

Using the generated shared library from this function, Jai will launch a kernel on device

# Arguments
- `ftype`::JAI_TYPE_FRAMEWORK: a framework type identifier

See also [`@jkernel`](jkernel), [`@jlaunch`](jlaunch), [`genslib_data`](genslib_data), [`genslib_accel`](genslib_accel)

# Examples
```julia-repl
julia> genslib_kernel(JAI_TYPE_FORTRAN)
```

# Implementation
T.B.D.

"""
function genslib_kernel(
        frame       ::JAI_TYPE_FRAMEWORK
    ) :: String

    error("ERROR: Framework-$(extract_name_from_frametype(typeof(frame))) " *
          "should implement 'genslib_kernel' function.")

end

function check_available_frameworks(
        aname::String
    ) ::Nothing

    for frame in JAI_SUPPORTED_FRAMEWORKS

        try
            slib = genslib_accel(frame, aname)

        catch err

            if typeof(err) in (JAI_ERROR_NOTIMPLEMENTED_FRAMEWORK,)
                JAI["debug"] && @jdebug err
            elseif typeof(err) in (MethodError,)
                JAI["debug"] && @jdebug "ERROR: no framework accel of " * string(typeof(frame))
            else
                rethrow()
            end
        end
    end

    if length(JAI_AVAILABLE_FRAMEWORKS) == 0
        throw(JAI_ERROR_NOAVAILABLE_FRAMEWORK())
    end

    return nothing
end

function select_framework(
        userframe,
        aname::String
    )

    framename, frameslib = nothing, nothing

    if !haskey(userframe, "_order_framework")
        userframe["_order_framework"] = Vector{String}()
    end

    if length(userframe) == 1 && length(userframe["_order_framework"]) == 0
        if length(JAI_AVAILABLE_FRAMEWORKS) == 0
            check_available_frameworks(aname)
        end

        for (name, slib) in JAI_AVAILABLE_FRAMEWORKS
            userframe[name] = slib
            push!(userframe["_order_framework"], name)
        end
    end

    for frame in userframe["_order_framework"]
        println("WWWW", frame)
    end

    (framename, frameslib)
end


function invoke_slibfunc(frame, slib, fname, inargs, outargs, innames, outnames)

    dtypes = argsdtypes(frame, args...)
    dtypestr = join([string(t) for t in dtypes], ",")

    libfunc = dlsym(slib, Symbol("jai_launch_" * launchid[1:_IDLEN]))

    # minimum check of device
    check_retval(jai_ccall(dtypestr, libfunc, args))

end

