# cuda.jl: implement functions for CUDA framework


CUDA_TEMPLATE_KERNEL= """
__global__ void {kname}({kargs}) {{

{kbody}

}}
"""


###### START of CODEGEN #######
function code_cpp_macros(
        frametype   ::JAI_TYPE_CUDA,
        apitype     ::JAI_TYPE_API,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        clauses     ::JAI_TYPE_CONFIG,
        data        ::NTuple{N, String} where N
    ) :: String

    return code_cpp_macros(JAI_CPP, apitype, prefix, args, clauses, data)
end


function code_cpp_header(
        frametype   ::JAI_TYPE_CUDA,
        apitype     ::JAI_TYPE_API,
        prefix      ::String,
        cvars       ::JAI_TYPE_ARGS,
        args        ::JAI_TYPE_ARGS,
        clauses     ::JAI_TYPE_CONFIG,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) ::String

    cpp_hdr = code_cpp_header(JAI_CPP, apitype, prefix, cvars, args, clauses, data)

    lines = Vector{String}()

    push!(lines, "#include <cassert>\n")
    push!(lines, "#include \"cuda.h\"\n")
    push!(lines, "#include \"cuda_runtime.h\"\n")
    push!(lines, "#define CUDA_ASSERT(x) (assert((x)==cudaSuccess))\n")

    return  join(lines, "") * cpp_hdr

end

function code_c_header(
        frametype   ::JAI_TYPE_CUDA,
        apitype     ::JAI_TYPE_API,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        clauses     ::JAI_TYPE_CONFIG,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) ::String

    buf     = Vector{String}()

    for arg in args
        typestr, vname, dimstr = code_c_typedecl(arg)
        ename = arg[end]

        # if arg is not scalar
        if typeof(arg[1]) != arg[2]

            if apitype == JAI_ALLOCATE
                #push!(buf, "$typestr * $(ename)$(dimstr);")
                push!(buf, "$typestr * $(ename);")

            elseif apitype == JAI_UPDATETO
                #push!(buf, "extern $typestr * $(ename)$(dimstr);")
                push!(buf, "extern $typestr * $(ename);")

            elseif apitype == JAI_LAUNCH
                push!(buf, "extern $typestr * $(ename);")

            elseif apitype == JAI_UPDATEFROM
                #push!(buf, "extern $typestr * $(ename)$(dimstr);")
                push!(buf, "extern $typestr * $(ename);")

            elseif apitype == JAI_DEALLOCATE
                push!(buf, "extern $typestr * $(ename);")

            else
            end
        end
    end

    return  join(buf, "\n")

end

#function code_c_function(
#        prefix      ::String,
#        suffix      ::String,
#        args        ::JAI_TYPE_ARGS,
#        clauses     ::JAI_TYPE_CONFIG,
#        body        ::String
#    ) ::String
#
#    name = prefix * suffix
#    dargs = code_c_dummyargs(args)
#
#    return jaifmt(C_TEMPLATE_FUNCTION, name=name, dargs=dargs, body=body)
#end


###### START of ACCEL #######

function code_c_functions(
        frametype   ::JAI_TYPE_CUDA,
        apitype     ::JAI_TYPE_ACCEL,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        clauses     ::JAI_TYPE_CONFIG,
        data        ::NTuple{N, JAI_TYPE_DATA} where N,
    ) :: String

    funcs = Vector{String}(undef, length(JAI_ACCEL_FUNCTIONS))

     # first arg in args should be 1-length integer vector
    vname = args[1][3]

    for (i, (fname, inout)) in enumerate(JAI_ACCEL_FUNCTIONS)
        if fname == "wait"
            funcs[i] = code_c_function(prefix, fname, args, clauses,
                            "CUDA_ASSERT(cudaDeviceSynchronize());")

        elseif fname == "get_num_devices"
            funcs[i] = code_c_function(prefix, fname, args, clauses,
                            "CUDA_ASSERT(cudaGetDeviceCount((int *)$vname));")

        elseif fname == "get_device_num"
            funcs[i] = code_c_function(prefix, fname, args, clauses,
                            "CUDA_ASSERT(cudaGetDevice((int *)$vname));")

        elseif fname == "set_device_num"
            funcs[i] = code_c_function(prefix, fname, args, clauses,
                            "CUDA_ASSERT(cudaSetDevice($vname[0]));")

        else
            funcs[i] = code_c_function(prefix, fname, args, clauses, "")
        end
    end

    return  join(funcs, "\n\n")

end

###### START of DATA #######

function code_c_functions(
        frametype   ::JAI_TYPE_CUDA,
        apitype     ::JAI_TYPE_API_DATA,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        clauses     ::JAI_TYPE_CONFIG,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) :: String

    buf     = Vector{String}()

    for (i, arg) in enumerate(args)
        aname = arg[3]
        asize = arg[5]
        ename = arg[end]

        if apitype == JAI_ALLOCATE
            push!(buf, "CUDA_ASSERT(cudaMalloc((void**)&$ename, $asize));")

        elseif apitype == JAI_UPDATETO
            push!(buf, "CUDA_ASSERT(cudaMemcpy((void *)$ename, (void *)$aname,
                    $asize, cudaMemcpyHostToDevice));")

        elseif apitype == JAI_UPDATEFROM
            push!(buf, "CUDA_ASSERT(cudaMemcpy((void *)$aname, (void *)$ename,
                    $asize, cudaMemcpyDeviceToHost));")

        elseif apitype == JAI_DEALLOCATE
            push!(buf, "CUDA_ASSERT(cudaFree($ename));")

        else
        end
    end

    body = join(buf, "\n")

    return code_c_function(prefix, JAI_MAP_API_FUNCNAME[apitype], args, clauses, body)
end

###### START of LAUNCH #######


function code_cuda_kernel(
        kname       ::String,
        args        ::JAI_TYPE_ARGS,
        clauses     ::JAI_TYPE_CONFIG,
        kbody       ::String
    ) :: String

    kargs = code_c_dummyargs(args)

    return jaifmt(CUDA_TEMPLATE_KERNEL, kname=kname, kargs=kargs, kbody=kbody)
end


function code_cuda_driver_body(
        kname   ::String,
        args    ::JAI_TYPE_ARGS,
        clauses     ::JAI_TYPE_CONFIG,
        config  ::Union{JAI_TYPE_CONFIG, Nothing} = nothing
    ) :: String

    out = Vector{String}()

    shared = 0
    stream = 0
    varnames = join((a[3] for a in args), ", ")

    lcfg = ["1", "1", "0", "0"]

    if config != nothing
        
        if JAI_CUDA in keys(config)
            cudacfg = config[JAI_CUDA]

            if "threads" in keys(cudacfg)
                threads = cudacfg["threads"]

                if threads isa Integer
                    lcfg[2] = string(threads)

                elseif threads isa Tuple && length(threads) > 0

                    for (i, cfg) in enumerate(threads)
                        if cfg isa Tuple
                            lcfg[i] = join((string(c) for c in cfg), ", ")
                        else
                            lcfg[i] = string(cfg)
                        end
                    end

                else
                    error("Wrong launch config syntax: " * string(threads))
                end
            end
        end
    end

    grid, block, shared, stream = lcfg

    nargs   = length(args)
    buf     = fill("", nargs)
    anames  = fill("", nargs)

    for (i, arg) in enumerate(args)

        aname= arg[3] 


        if arg[1] isa AbstractArray
            t, n, d = code_c_typedecl(arg)
            ename = arg[end]

            buf[i] = "$t (*ptr_$n)$d = reinterpret_cast<$t (*)$d>($ename);"
            anames[i] = "(*ptr_$n)" 
        else
            anames[i] = aname
        end
    end

    reintepret  = join(buf, "\n")
    dvarnames   = join(anames, ", \n")

    push!(out, """

$reintepret

$kname<<<dim3($grid), dim3($block), $shared, $stream>>>(
    $dvarnames
);
""")

    return join(out, "\n")
end


function code_c_functions(
        frametype   ::JAI_TYPE_CUDA,
        apitype     ::JAI_TYPE_LAUNCH,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        clauses     ::JAI_TYPE_CONFIG,
        data        ::NTuple{N, JAI_TYPE_DATA} where N,
        launch_config   ::Union{JAI_TYPE_CONFIG, Nothing} = nothing
    ) :: String

    kname = prefix * "device"

    # kernel function
    kfunc = code_cuda_kernel(kname, args, data[1], clauses)

    # driver function
    dbody = code_cuda_driver_body(kname, args, launch_config, clauses)
    dfunc = code_c_function(prefix, JAI_MAP_API_FUNCNAME[apitype], args, clauses, dbody)

    return kfunc * "\n\n" * dfunc
end


###### END of CODEGEN #######
