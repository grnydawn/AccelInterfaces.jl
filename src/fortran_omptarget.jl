# fortran_omptarget.jl: implement functions for Fortran OpenMP Target framework
# NOTE: (var, dtype, vname, vinout, addr, vshape, voffset) = arg


###### START of ACCEL #######

function code_module_specpart(
        frame       ::JAI_TYPE_FORTRAN_OMPTARGET,
        apitype     ::JAI_TYPE_ACCEL,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, String} where N
    ) :: String

    fortran_spec = code_module_specpart(JAI_FORTRAN, apitype, prefix, args, data)

    return "USE OMP_LIB\n" * fortran_spec
end

function code_module_subppart(
        frame       ::JAI_TYPE_FORTRAN_OMPTARGET,
        apitype     ::JAI_TYPE_ACCEL,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, String} where N
    ) :: String

    # One dummy argument
    arg     = args[1]
    dnames  = arg[3]
    funcs   = Vector{String}(undef, length(FORTRAN_ACCEL_FUNCTIONS))

    for (i, (fname, inout)) in enumerate(FORTRAN_ACCEL_FUNCTIONS)

        specpart = code_fortran_typedecl(arg, inout=inout)

        if fname == "device_init"
            execpart = ""

        elseif fname == "device_fini"
            execpart = ""

        elseif fname == "get_device_num"
            execpart = "$dnames(1) = omp_get_device_num()\n"

        elseif fname == "get_num_devices"
            execpart =  "$dnames(1) = omp_get_num_devices()\n"

        elseif fname == "set_device_num"
            execpart =  "CALL omp_set_device_num($dnames(1))\n"

        elseif fname == "wait"
            execpart =  "!$omp taskwait"

        end

        funcs[i] = code_fortran_function(prefix, fname, dnames, specpart, execpart)
    end

    return  join(funcs, "\n\n")

end

###### START of DATA #######

function code_module_specpart(
        frame       ::JAI_TYPE_FORTRAN_OMPTARGET,
        apitype     ::JAI_TYPE_API_DATA,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, String} where N
    ) :: String

    return "PUBLIC " * prefix * JAI_MAP_API_FUNCNAME[apitype]
end

function code_module_subppart(
        frame       ::JAI_TYPE_FORTRAN_OMPTARGET,
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
        frame       ::JAI_TYPE_FORTRAN_OMPTARGET,
        apitype     ::JAI_TYPE_LAUNCH,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, String} where N
    ) :: String

    return "PUBLIC " * prefix * JAI_MAP_API_FUNCNAME[apitype]
end

function code_module_subppart(
        frame       ::JAI_TYPE_FORTRAN_OMPTARGET,
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
