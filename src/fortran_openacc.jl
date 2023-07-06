# fortran_openacc.jl: implement functions for Fortran OpenAcc Target framework
# NOTE: (var, dtype, vname, vinout, addr, vshape, voffset) = arg


###### START of ACCEL #######

function code_module_specpart(
        frame       ::JAI_TYPE_FORTRAN_OPENACC,
        apitype     ::JAI_TYPE_ACCEL,
        prefix      ::String,
        cvars       ::JAI_TYPE_ARGS,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) :: String

    fortran_spec = code_module_specpart(JAI_FORTRAN, apitype, prefix, cvars, args, data)

    # TODO: required compiler-specific openacc use 
    # one idea is to actually test by compiling a small test code
    return "USE OPENACC_LIB\n" * fortran_spec
end

function code_module_subppart(
        frame       ::JAI_TYPE_FORTRAN_OPENACC,
        apitype     ::JAI_TYPE_ACCEL,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) :: String

    # One dummy argument
    arg     = args[1]
    dnames  = arg[3]
    funcs   = Vector{String}(undef, length(JAI_ACCEL_FUNCTIONS))

    for (i, (fname, inout)) in enumerate(JAI_ACCEL_FUNCTIONS)

        specpart = code_fortran_typedecl(arg, inout=inout)

        if fname == "device_init"
            execpart = ""

        elseif fname == "device_fini"
            execpart = ""

        elseif fname == "get_device_num"
            execpart = "$dnames(1) = acc_get_device_num(acc_device_default)\n"

        elseif fname == "get_num_devices"
            execpart =  "$dnames(1) = acc_get_num_devices(acc_device_default)\n"

        elseif fname == "set_device_num"
            execpart =  "CALL acc_set_device_num(INT($dnames(1), 4), acc_device_default)\n"

        elseif fname == "wait"
            execpart =  "CALL acc_wait(0)"

        end

        funcs[i] = code_fortran_function(prefix, fname, dnames, specpart, execpart)
    end

    return  join(funcs, "\n\n")

end

###### START of DATA #######

function code_module_subppart(
        frame       ::JAI_TYPE_FORTRAN_OPENACC,
        apitype     ::JAI_TYPE_API_DATA,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) :: String

    apiname  = JAI_MAP_API_FUNCNAME[apitype]

    argnames    = Vector{String}(undef, length(args))
    typedecls   = Vector{String}(undef, length(args))
    directs     = Vector{String}(undef, length(args))

    for (i, arg) in enumerate(args)
        argname     = arg[3]
        argnames[i] = argname
        typedecls[i] = code_fortran_typedecl(arg)

        if apitype == JAI_ALLOCATE
            directs[i] = "!\$acc enter data create($(argname))"

        elseif apitype == JAI_DEALLOCATE
            directs[i] = "!\$acc exit data delete($(argname))"

        elseif apitype == JAI_UPDATETO
            directs[i] = "!\$acc update device($(argname))"

        elseif apitype == JAI_UPDATEFROM
            directs[i] = "!\$acc update host($(argname))"

        else
            error("Unknown api type: " * string(apitype))

        end
    end

    dargs       = join(argnames, ", ")
    specpart    = join(typedecls, "\n")
    execpart    = join(directs, "\n")

    return code_fortran_function(prefix, apiname, dargs, specpart, execpart)

end

###### START of LAUNCH #######

function code_module_subppart(
        frame       ::JAI_TYPE_FORTRAN_OPENACC,
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
