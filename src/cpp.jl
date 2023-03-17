# cpp.jl: implement functions for C++ framework


# Julia type to C type conversion table
const JAI_MAP_JULIA_C = Dict{DataType, String}(
    Int8    => "int8_t",
    Int16   => "int16_t",
    Int32   => "int32_t",
    Int64   => "int64_t",
    UInt8   => "uint8_t",
    UInt16  => "uint16_t",
    UInt32  => "uint32_t",
    UInt64  => "uint64_t",
    Float32 => "float",
    Float64 => "double"
)


###### START of CODEGEN #######

function code_cpp_header(
        frame       ::JAI_TYPE_CPP,
        apitype     ::JAI_TYPE_API,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, String} where N
    ) ::String

    return ""

end

function code_c_header(
        frame       ::JAI_TYPE_CPP,
        apitype     ::JAI_TYPE_API,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, String} where N
    ) ::String

    return ""

end

function code_c_typedecl(arg::JAI_TYPE_ARG) :: String

    (var, dtype, vname, vinout, addr, vshape, voffset) = arg

    if var isa AbstractArray

        typestr = JAI_MAP_JULIA_C[dtype]
        dimlist = Vector{String}(undef, length(vshape))
        accum = 1

        for (idx, len) in enumerate(reverse(vshape))
            dimlist[idx] = "[" * string(len) * "]"
        end

        dimstr = join(dimlist, "")

    else
        typestr = JAI_MAP_JULIA_C[dtype]
        dimstr = ""
    end

    return typestr * " " * vname * dimstr
end

function code_c_dummyargs(
        args        ::JAI_TYPE_ARGS
    ) ::String

    dargs = Vector{String}()

    for arg in args
        push!(dargs, code_c_typedecl(arg))
    end

    return join(dargs, ", ")
end

function code_c_function(
        prefix      ::String,
        suffix      ::String,
        args        ::JAI_TYPE_ARGS,
        body        ::String
    ) ::String

    name = prefix * suffix
    dargs = code_c_dummyargs(args)

    return jaifmt(C_TEMPLATE_FUNCTION, name=name, dargs=dargs, body=body)
end


###### START of ACCEL #######

function code_c_functions(
        frame       ::JAI_TYPE_CPP,
        apitype     ::JAI_TYPE_ACCEL,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, String} where N
    ) :: String

    funcs = Vector{String}(undef, length(JAI_ACCEL_FUNCTIONS))

    for (i, (name, inout)) in enumerate(JAI_ACCEL_FUNCTIONS)
        funcs[i] = code_c_function(prefix, name, args, "")
    end

    return  join(funcs, "\n\n")

end

###### START of DATA #######

function code_module_specpart(
        frame       ::Union{JAI_TYPE_FORTRAN, JAI_TYPE_FORTRAN_OMPTARGET},
        apitype     ::JAI_TYPE_API_DATA,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, String} where N
    ) :: String

    return "PUBLIC " * prefix * JAI_MAP_API_FUNCNAME[apitype]
end

function code_module_subppart(
        frame       ::JAI_TYPE_FORTRAN,
        apitype     ::JAI_TYPE_API_DATA,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, String} where N
    ) :: String

    apiname  = JAI_MAP_API_FUNCNAME[apitype]

    argnames = Vector{String}(undef, length(args))
    typedecls   = Vector{String}(undef, length(args))

    for (i, arg) in enumerate(args)
        argnames[i] = arg[3]
        typedecls[i] = code_fortran_typedecl(arg)
    end

    dargs = join(argnames, ", ")
    specpart = join(typedecls, "\n")

    return code_fortran_function(prefix, apiname, dargs, specpart, "")

end

###### START of LAUNCH #######

function code_c_functions(
        frame       ::JAI_TYPE_CPP,
        apitype     ::JAI_TYPE_LAUNCH,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, String} where N
    ) :: String

    return code_c_function(prefix, JAI_MAP_API_FUNCNAME[apitype], args, data[1])

end


###### END of CODEGEN #######
