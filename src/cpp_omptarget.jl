# cpp_omptarget.jl: implement functions for C++ OpenMP Target framework



###### START of CODEGEN #######

function code_cpp_header(
        frametype   ::JAI_TYPE_CPP_OMPTARGET,
        apitype     ::JAI_TYPE_API,
        data_frametype  ::Union{JAI_TYPE_FRAMEWORK, Nothing},
        prefix      ::String,
        cvars       ::JAI_TYPE_ARGS,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) ::String

    cpp_hdr = code_cpp_header(JAI_CPP, apitype, data_frametype, prefix,
                                cvars, args, data)

    return "#include <omp.h>\n" * cpp_hdr

end


###### START of ACCEL #######

function code_c_functions(
        frametype   ::JAI_TYPE_CPP_OMPTARGET,
        apitype     ::JAI_TYPE_ACCEL,
        data_frametype  ::Union{JAI_TYPE_FRAMEWORK, Nothing},
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) :: String

    argname   = args[1][3] 
    funcs = Vector{String}(undef, length(JAI_ACCEL_FUNCTIONS))

    for (i, (name, inout)) in enumerate(JAI_ACCEL_FUNCTIONS)

        if name == "device_init"
            body = ""

        elseif name == "device_fini"
            body = ""

        elseif name == "get_device_num"
            body = "$argname[0] = (int64_t) omp_get_device_num();\n"

        elseif name == "get_num_devices"
            body =  "$argname[0] = (int64_t) omp_get_num_devices();\n"

        elseif name == "set_device_num"
            body =  "omp_set_default_device((int) $argname[0]);\n"

        elseif name == "wait"
            body =  "#pragma taskwait"

        end

        funcs[i] = code_c_function(prefix, name, args, body)
    end

    return  join(funcs, "\n\n")

end

###### START of DATA #######

function code_c_functions(
        frametype   ::JAI_TYPE_CPP_OMPTARGET,
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
        frametype   ::JAI_TYPE_CPP_OMPTARGET,
        apitype     ::JAI_TYPE_LAUNCH,
        data_frametype  ::Union{JAI_TYPE_FRAMEWORK, Nothing},
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, String} where N
    ) :: String

    return code_c_function(prefix, JAI_MAP_API_FUNCNAME[apitype], args, data[1])
end


###### END of CODEGEN #######
