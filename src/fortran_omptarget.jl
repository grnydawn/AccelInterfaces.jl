# fortran_omptarget.jl: implement functions for Fortran OpenMP Target framework
# NOTE: (var, dtype, vname, vinout, addr, vshape, voffset) = arg


###### START of ACCEL #######

function code_module_specpart(
        frame       ::JAI_TYPE_FORTRAN_OMPTARGET,
        apitype     ::JAI_TYPE_ACCEL,
        data_frametype  ::Union{JAI_TYPE_FRAMEWORK, Nothing},
        prefix      ::String,
        cvars       ::JAI_TYPE_ARGS,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) :: String

    fortran_spec = code_module_specpart(JAI_FORTRAN, apitype,
                        data_frametype, prefix, cvars, args, data)

    return "USE OMP_LIB\n" * fortran_spec
end

function code_module_subppart(
        frame       ::JAI_TYPE_FORTRAN_OMPTARGET,
        apitype     ::JAI_TYPE_ACCEL,
        data_frametype  ::Union{JAI_TYPE_FRAMEWORK, Nothing},
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
            execpart = "$dnames(1) = omp_get_device_num()\n"

        elseif fname == "get_num_devices"
            execpart =  "$dnames(1) = omp_get_num_devices()\n"

        elseif fname == "set_device_num"
            execpart =  "CALL omp_set_device_num($dnames(1))\n"

        elseif fname == "wait"
            execpart =  "!\$omp taskwait"

        end

        funcs[i] = code_fortran_function(prefix, fname, dnames, specpart, execpart)
    end

    return  join(funcs, "\n\n")

end

###### START of DATA #######

function code_module_subppart(
        frame       ::JAI_TYPE_FORTRAN_OMPTARGET,
        apitype     ::JAI_TYPE_API_DATA,
        data_frametype  ::Union{JAI_TYPE_FRAMEWORK, Nothing},
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
            # TODO: check if the following is preferred
            #device = omp_get_default_device();
            #ptr = omp_target_alloc(bytes,device);
            #omp_target_associate_ptr(host_ptr,device_ptr,bytes,0device_offset,0device_num);

            directs[i] = "!\$omp target enter data map(alloc: $(argname))"

        elseif apitype == JAI_DEALLOCATE
            directs[i] = "!\$omp target exit data map(delete: $(argname))"

        elseif apitype == JAI_UPDATETO
            directs[i] = "!\$omp target update to($(argname))"

        elseif apitype == JAI_UPDATEFROM
            directs[i] = "!\$omp target update from($(argname))"

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
        frame       ::JAI_TYPE_FORTRAN_OMPTARGET,
        apitype     ::JAI_TYPE_LAUNCH,
        data_frametype  ::Union{JAI_TYPE_FRAMEWORK, Nothing},
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
