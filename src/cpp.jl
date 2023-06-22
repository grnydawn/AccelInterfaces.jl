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
    Float64 => "double",
    PtrAny  => "void *"
)


###### START of CODEGEN #######

function code_cpp_valstr(
        cvar    ::JAI_TYPE_ARG
    ) ::String

    (var, dtype, vname, vinout, bytes, vshape, voffset, extname) = cvar

    if length(vshape) > 0
        temp = Vector{String}()
        for i in eachindex(var)
            push!(temp, string(var[i]))
        end    
        valstr = "{" * join(temp, ", ") * "}"
    else
        valstr = string(var)
    end

    return valstr
end

function code_cpp_header(
        frame       ::JAI_TYPE_CPP,
        apitype     ::JAI_TYPE_API,
        data_frametype  ::Union{JAI_TYPE_FRAMEWORK, Nothing},
        prefix      ::String,
        cvars       ::JAI_TYPE_ARGS,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) ::String

    consts = Vector{String}()

    for cvar in cvars
        typestr, vname, dimstr = code_c_typedecl(cvar)
        push!(consts, "const $typestr $vname $dimstr = $(code_cpp_valstr(cvar));")
    end

    return join(consts, "\n")

end

function code_c_header(
        frame       ::Union{JAI_TYPE_CPP, JAI_TYPE_CPP_OMPTARGET},
        apitype     ::JAI_TYPE_API,
        data_frametype  ::Union{JAI_TYPE_FRAMEWORK, Nothing},
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) ::String

    return ""

end

function code_c_typedecl(arg::JAI_TYPE_ARG) :: Tuple{String, String, String}

    (var, dtype, vname, vinout, bytes, vshape, voffset, extname) = arg

    typestr = JAI_MAP_JULIA_C[dtype]

    if var isa AbstractArray

        dimlist = Vector{String}(undef, length(vshape))
        accum = 1

        for (idx, len) in enumerate(reverse(vshape))
            dimlist[idx] = "[" * string(len) * "]"
        end

        dimstr = join(dimlist, "")

    elseif var isa Tuple

        dimstr = "[" * string(length(var)) * "]"

    else
        dimstr = ""
    end

    return typestr, vname, dimstr
end

#function code_c_typedecl(arg::JAI_TYPE_ARG) :: String
#
#    (var, dtype, vname, vinout, bytes, vshape, voffset) = arg
#
#    if var isa AbstractArray
#
#        typestr = JAI_MAP_JULIA_C[dtype]
#        dimlist = Vector{String}(undef, length(vshape))
#        accum = 1
#
#        for (idx, len) in enumerate(reverse(vshape))
#            dimlist[idx] = "[" * string(len) * "]"
#        end
#
#        dimstr = join(dimlist, "")
#
#    else
#        typestr = JAI_MAP_JULIA_C[dtype]
#        dimstr = ""
#    end
#
#    return (typestr * " " * vname * dimstr)
#end

function code_c_dummyargs(
        args        ::JAI_TYPE_ARGS
    ) ::String

    dargs = Vector{String}()

    for arg in args
        typestr, vname, dimstr = code_c_typedecl(arg)
        push!(dargs, typestr * " " * vname * dimstr)
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
        data_frametype  ::Union{JAI_TYPE_FRAMEWORK, Nothing},
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) :: String

    funcs = Vector{String}(undef, length(JAI_ACCEL_FUNCTIONS))

    for (i, (name, inout)) in enumerate(JAI_ACCEL_FUNCTIONS)
        funcs[i] = code_c_function(prefix, name, args, "")
    end

    return  join(funcs, "\n\n")

end

###### START of DATA #######

function code_c_functions(
        frame       ::JAI_TYPE_CPP,
        apitype     ::JAI_TYPE_API_DATA,
        data_frametype  ::Union{JAI_TYPE_FRAMEWORK, Nothing},
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) :: String

    return code_c_function(prefix, JAI_MAP_API_FUNCNAME[apitype], args, "")
end

###### START of LAUNCH #######

function code_c_functions(
        frame       ::JAI_TYPE_CPP,
        apitype     ::JAI_TYPE_LAUNCH,
        data_frametype  ::Union{JAI_TYPE_FRAMEWORK, Nothing},
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) :: String

    return code_c_function(prefix, JAI_MAP_API_FUNCNAME[apitype], args, data[1])
end


###### END of CODEGEN #######
