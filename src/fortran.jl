# fortran.jl: implement functions for Fortran framework
# NOTE: (var, dtype, vname, vinout, addr, vshape, voffset) = arg


const FORTRAN_ACCEL_FUNCTIONS = (
        ("get_num_devices", JAI_ARG_OUT),
        ("get_device_num",  JAI_ARG_OUT),
        ("set_device_num",  JAI_ARG_IN ),
        ("device_init",     JAI_ARG_IN ),
        ("device_fini",     JAI_ARG_IN ),
        ("wait",            JAI_ARG_IN )
    )

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

function code_fortran_typedecl(
        arg ::JAI_TYPE_ARG;
        inout :: Union{JAI_TYPE_INOUT, Nothing} = nothing
    ) :: String

    (var, dtype, vname, vinout, addr, vshape, voffset) = arg

    if inout isa JAI_TYPE_INOUT
        vinout = inout
    end

    type = JAI_MAP_JULIA_FORTRAN[dtype]
    attrs = Vector{String}()

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

    push!(attrs, JAI_MAP_FORTRAN_INOUT[vinout])

    return type * ", " * join(attrs, ", ") * " :: " * vname
end


###### START of ACCEL #######

function code_module_specpart(
        frame       ::JAI_TYPE_FORTRAN,
        apitype     ::JAI_TYPE_ACCEL,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, String} where N
    ) :: String

    specs = Vector{String}(undef, length(FORTRAN_ACCEL_FUNCTIONS))

    for (i, (name, inout)) in enumerate(FORTRAN_ACCEL_FUNCTIONS)
        funcname = prefix * name
        specs[i] = "PUBLIC " * funcname
    end

    return join(specs, "\n")
end

function code_module_subppart(
        frame       ::JAI_TYPE_FORTRAN,
        apitype     ::JAI_TYPE_ACCEL,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, String} where N
    ) :: String

    arg = args[1]
    funcs = Vector{String}(undef, length(FORTRAN_ACCEL_FUNCTIONS))

    for (i, (name, inout)) in enumerate(FORTRAN_ACCEL_FUNCTIONS)
        specpart = code_fortran_typedecl(arg, inout=inout)
        funcs[i] = code_fortran_function(prefix, name, arg[3], specpart, "")
    end

    return  join(funcs, "\n\n")

end

###### START of DATA #######

function code_module_specpart(
        frame       ::JAI_TYPE_FORTRAN,
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

function code_module_specpart(
        frame       ::JAI_TYPE_FORTRAN,
        apitype     ::JAI_TYPE_LAUNCH,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, String} where N
    ) :: String

    return "PUBLIC " * prefix * JAI_MAP_API_FUNCNAME[apitype]
end

function code_module_subppart(
        frame       ::JAI_TYPE_FORTRAN,
        apitype     ::JAI_TYPE_LAUNCH,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, String} where N
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
