# framework.jl: implement common functions for framework interfaces

"""
    function gencode_accel(ftype)

Generate framework source code to drive device.

Using the generated code from this function, Jai will access and configure the device.

# Arguments
- `ftype`::JAI_TYPE_FRAMEWORK: a framework type identifier

See also [`@jaccel`](jaccel), [`gencode_kernel`](gencode_kernel), [`gencode_data`](gencode_data)

# Examples
```julia-repl
julia> gencode_accel(JAI_TYPE_FORTRAN)
```

# Implementation
T.B.D.

"""
function gencode_accel(
        frame       ::JAI_TYPE_FRAMEWORK
    ) :: String

    error("ERROR: Framework-$(extract_name_from_frametype(typeof(frame))) " *
          "should implement 'gencode_accel' function.")
end

"""
    function gencode_data(ftype)

Generate framework source code to move data between host and device.

Using the generated code from this function, Jai will move data between host and device

# Arguments
- `ftype`::JAI_TYPE_FRAMEWORK: a framework type identifier

See also [`@jenterdata`](jenterdata), [`@jexitdata`](jexitdata), [`gencode_kernel`](gencode_kernel), [`gencode_accel`](gencode_accel)

# Examples
```julia-repl
julia> gencode_data(JAI_TYPE_FORTRAN)
```

# Implementation
T.B.D.

"""
function gencode_data(
        frame       ::JAI_TYPE_FRAMEWORK
    ) :: String

    error("ERROR: Framework-$(extract_name_from_frametype(typeof(frame))) " *
          "should implement 'gencode_data' function.")

end

"""
    function gencode_kernel(ftype)

Generate framework source code to launch a kernel on device

Using the generated code from this function, Jai will launch a kernel on device

# Arguments
- `ftype`::JAI_TYPE_FRAMEWORK: a framework type identifier

See also [`@jkernel`](jkernel), [`@jlaunch`](jlaunch), [`gencode_data`](gencode_data), [`gencode_accel`](gencode_accel)

# Examples
```julia-repl
julia> gencode_kernel(JAI_TYPE_FORTRAN)
```

# Implementation
T.B.D.

"""
function gencode_kernel(
        frame       ::JAI_TYPE_FRAMEWORK
    ) :: String

    error("ERROR: Framework-$(extract_name_from_frametype(typeof(frame))) " *
          "should implement 'gencode_kernel' function.")

end


