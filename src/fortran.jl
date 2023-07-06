# fortran.jl: implement functions for Fortran framework
# NOTE: (var, dtype, vname, vinout, bytes, vshape, voffset) = arg


const JAI_MAP_JULIA_FORTRAN = Dict{DataType, String}(
    Int8    => "INTEGER (C_INT8_T)",
    Int16   => "INTEGER (C_INT16_T)",
    Int32   => "INTEGER (C_INT32_T)",
    Int64   => "INTEGER (C_INT64_T)",
    Int128  => "INTEGER (C_INT128_T)",
    UInt8   => "INTEGER (C_INT8_T)",
    UInt16  => "INTEGER (C_INT16_T)",
    UInt32  => "INTEGER (C_INT32_T)",
    UInt64  => "INTEGER (C_INT64_T)",
    UInt128 => "INTEGER (C_INT128_T)",
    Float32 => "REAL (C_FLOAT)",
    Float64 => "REAL (C_DOUBLE)"
)

const JAI_MAP_FORTRAN_INOUT = Dict{JAI_TYPE_INOUT, String}(
    JAI_ARG_IN      => "INTENT(IN)",
    JAI_ARG_OUT     => "INTENT(OUT)",
    JAI_ARG_INOUT   => "INTENT(INOUT)",
)


###### START of CODEGEN #######

function code_fortran_function(
        prefix  ::String,
        suffix  ::String,
        dargs   ::String,
        spec    ::String,
        exec    ::String
    ) ::String

    return jaifmt(FORTRAN_TEMPLATE_FUNCTION, prefix=prefix, suffix=suffix,
                          dummyargs=dargs, specpart=spec, execpart=exec)

end

function code_attr_dimension(
        attrs   ::Vector{String},
        var     ::JAI_TYPE_DATA,
        vshape  ::NTuple{N, T} where {N, T<:Integer},
        voffset ::NTuple{N, T} where {N, T<:Integer}
    ) :: Vector{String}

    if var isa OffsetArray
        dimlist = Vector{String}()
        for (length, offset) in zip(vshape, voffset)
            push!(dimlist, string(1+offset) * ":" * string(length+offset))
        end
        push!(attrs, "DIMENSION(" * join(dimlist, ", ") * ")")

    elseif length(vshape) > 0
        dimlist = Vector{String}()
        for length in vshape
            push!(dimlist, "1:" * string(length))
        end
        push!(attrs, "DIMENSION(" * join(dimlist, ", ") * ")")

    end

    attrs
end

function code_fortran_valstr(
        cvar ::JAI_TYPE_ARG;
    ) :: String

    (var, dtype, vname, vinout, bytes, vshape, voffset, extname) = cvar

    if length(vshape) > 0
        temp = Vector{String}()
        for i in eachindex(var)
            push!(temp, string(var[i]))
        end    
        if length(vshape) > 1
            valstr = "&\nRESHAPE((/" * join(temp, ", &\n") * "/), &\n(/" * string(vshape)[2:end-1] * "/))"
        else
            valstr = "(/" * join(temp, ", &\n") * "/)"
        end
    else
        valstr = string(var)
    end

    return valstr
end

function code_parameter_typedecl(
        cvar ::JAI_TYPE_ARG;
    ) :: String

    (var, dtype, vname, vinout, bytes, vshape, voffset, extname) = cvar

    type = JAI_MAP_JULIA_FORTRAN[dtype]

    attrs = Vector{String}()
    code_attr_dimension(attrs, var, vshape, voffset)
    push!(attrs, "PARAMETER") 

    return type * ", " * join(attrs, ", ") * " :: " * vname * " = " * code_fortran_valstr(cvar)
end



function code_fortran_typedecl(
        arg ::JAI_TYPE_ARG;
        inout :: Union{JAI_TYPE_INOUT, Nothing} = nothing
    ) :: String

    (var, dtype, vname, vinout, bytes, vshape, voffset, extname) = arg

    type = JAI_MAP_JULIA_FORTRAN[dtype]

    attrs = Vector{String}()
    code_attr_dimension(attrs, var, vshape, voffset)

    if inout isa JAI_TYPE_INOUT
        vinout = inout
    end
    push!(attrs, JAI_MAP_FORTRAN_INOUT[vinout])

    return type * ", " * join(attrs, ", ") * " :: " * vname
end


###### START of ACCEL #######

function code_module_specpart(
        frametype   ::JAI_TYPE_FORTRAN,
        apitype     ::JAI_TYPE_ACCEL,
        prefix      ::String,
        cvars       ::JAI_TYPE_ARGS,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) :: String

    specs = Vector{String}()

    for cvar in cvars
        push!(specs, code_parameter_typedecl(cvar))
    end

    for (name, inout) in JAI_ACCEL_FUNCTIONS
        funcname = prefix * name
        push!(specs, "PUBLIC " * funcname)
    end

    return join(specs, "\n")
end

function code_module_subppart(
        frametype   ::JAI_TYPE_FORTRAN,
        apitype     ::JAI_TYPE_ACCEL,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) :: String

    arg = args[1]
    funcs = Vector{String}(undef, length(JAI_ACCEL_FUNCTIONS))

    for (i, (name, inout)) in enumerate(JAI_ACCEL_FUNCTIONS)
        specpart = code_fortran_typedecl(arg, inout=inout)
        funcs[i] = code_fortran_function(prefix, name, arg[3], specpart, "")
    end

    return  join(funcs, "\n\n")

end

###### START of DATA #######

function code_module_specpart(
        frametype   ::Union{JAI_TYPE_FORTRAN, JAI_TYPE_FORTRAN_OMPTARGET,
                        JAI_TYPE_FORTRAN_OPENACC},
        apitype     ::JAI_TYPE_API_DATA,
        prefix      ::String,
        cvars       ::JAI_TYPE_ARGS,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) :: String

    specs = Vector{String}()

    for cvar in cvars
        push!(specs, code_parameter_typedecl(cvar))
    end

    push!(specs, "PUBLIC " * prefix * JAI_MAP_API_FUNCNAME[apitype])

    return join(specs, "\n")

end

function code_module_subppart(
        frametype   ::JAI_TYPE_FORTRAN,
        apitype     ::JAI_TYPE_API_DATA,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
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

function code_module_specpart(
        frametype   ::Union{JAI_TYPE_FORTRAN,JAI_TYPE_FORTRAN_OMPTARGET,
                        JAI_TYPE_FORTRAN_OPENACC},
        apitype     ::JAI_TYPE_LAUNCH,
        prefix      ::String,
        cvars       ::JAI_TYPE_ARGS,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) :: String

    specs = Vector{String}()

    for cvar in cvars
        push!(specs, code_parameter_typedecl(cvar))
    end

    push!(specs, "PUBLIC " * prefix * JAI_MAP_API_FUNCNAME[apitype])

    return join(specs, "\n")

end

function code_module_subppart(
        frametype   ::JAI_TYPE_FORTRAN,
        apitype     ::JAI_TYPE_LAUNCH,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) :: String

    apiname = JAI_MAP_API_FUNCNAME[apitype]

    argnames = Vector{String}(undef, length(args))
    typedecls   = Vector{String}(undef, length(args))

    for (i, arg) in enumerate(args)
        argnames[i]  = arg[3]
        typedecls[i] = code_fortran_typedecl(arg)
    end

    dargs = join(argnames, ", ")
    specpart = join(typedecls, "\n")

    return code_fortran_function(prefix, apiname, dargs, specpart, data[1])
end

###### END of CODEGEN #######
