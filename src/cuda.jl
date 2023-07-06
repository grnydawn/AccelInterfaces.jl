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
        data        ::NTuple{N, String} where N
    ) :: String

    return code_cpp_macros(JAI_CPP, apitype, prefix, args, data)
end


function code_cpp_header(
        frametype   ::JAI_TYPE_CUDA,
        apitype     ::JAI_TYPE_API,
        prefix      ::String,
        cvars       ::JAI_TYPE_ARGS,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) ::String

    cpp_hdr = code_cpp_header(JAI_CPP, apitype, prefix, cvars, args, data)

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

            else
            end
        end
    end

    return  join(buf, "\n")

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
        frametype   ::JAI_TYPE_CUDA,
        apitype     ::JAI_TYPE_ACCEL,
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N,
    ) :: String

    funcs = Vector{String}(undef, length(JAI_ACCEL_FUNCTIONS))

    for (i, (name, inout)) in enumerate(JAI_ACCEL_FUNCTIONS)
        if name == "wait"
            funcs[i] = code_c_function(prefix, name, args, "CUDA_ASSERT(cudaDeviceSynchronize());")
        else
            funcs[i] = code_c_function(prefix, name, args, "")
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
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) :: String

    buf     = Vector{String}()

    for (i, arg) in enumerate(args)
        aname = arg[3]
        asize = arg[5]
        ename = arg[end]

        if apitype == JAI_ALLOCATE
            push!(buf, "CUDA_ASSERT(cudaMalloc((void**)&$ename, $asize));")
            #push!(buf, "printf(\"ALLOC PPPPP $aname = %p\\n\", $aname);")
            #push!(buf, "printf(\"ALLOC QQQQQ $ename = %p\\n\", $ename);")

        elseif apitype == JAI_UPDATETO
            #push!(buf, "printf(\"UPTO PPPPP $aname = %p\\n\", $aname);")
            #push!(buf, "printf(\"UPTO QQQQQ $ename = %p\\n\", $ename);")
            push!(buf, "CUDA_ASSERT(cudaMemcpy((void *)$ename, (void *)$aname,
                    $asize, cudaMemcpyHostToDevice));")

        elseif apitype == JAI_UPDATEFROM
            #push!(buf, "printf(\"UPFR PPPPP $aname = %p\\n\", $aname);")
            #push!(buf, "printf(\"UPFR QQQQQ $ename = %p\\n\", $ename);")
            push!(buf, "CUDA_ASSERT(cudaMemcpy((void *)$aname, (void *)$ename,
                    $asize, cudaMemcpyDeviceToHost));")

        else
        end
    end

    body = join(buf, "\n")

    return code_c_function(prefix, JAI_MAP_API_FUNCNAME[apitype], args, body)
end

###### START of LAUNCH #######


function code_cuda_kernel(
        kname       ::String,
        args        ::JAI_TYPE_ARGS,
        kbody       ::String
    ) :: String

    kargs = code_c_dummyargs(args)

    return jaifmt(CUDA_TEMPLATE_KERNEL, kname=kname, kargs=kargs, kbody=kbody)
end


function code_cuda_driver_body(
        kname   ::String,
        args    ::JAI_TYPE_ARGS,
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
    dbuf    = fill("", nargs)
    anames  = fill("", nargs)
    pf  = Vector{String}()
    #dname   = args[end][3]

    for (i, arg) in enumerate(args)

        aname= arg[3] 


        if arg[1] isa AbstractArray
            t, n, d = code_c_typedecl(arg)
            ename = arg[end]

            buf[i] = "$t (*ptr_$n)$d = reinterpret_cast<$t (*)$d>($ename);"
            anames[i] = "(*ptr_$n)" 
            #push!(pf, "printf(\"KERNEL PPPPP $n = %p\\n\", $n);")
            #push!(pf, "printf(\"KERNEL QQQQQ $ename = %p\\n\", $ename);")
        else
            anames[i] = aname
        end

        #anames[i] = "*ptr_" * arg[3] 
        #anames[i] = "$dname[$(i-1)]"
        #dbuf[i] = "printf(\"AT LAUNCH, $aname: dptr= %p\\n\", (void *)$aname);"

    end

    reintepret  = join(buf, "\n")
    #reintepret  = ""
    dvarnames   = join(anames, ", \n")
    debug       = join(dbuf, "\n")
    printf      = join(pf, "\n")

    push!(out, """

$debug

$reintepret

$printf

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
        data        ::NTuple{N, JAI_TYPE_DATA} where N,
        launch_config   ::Union{JAI_TYPE_CONFIG, Nothing} = nothing
    ) :: String

    kname = prefix * "device"

    # kernel function
    kfunc = code_cuda_kernel(kname, args, data[1])

    # driver function
    dbody = code_cuda_driver_body(kname, args, launch_config)
    dfunc = code_c_function(prefix, JAI_MAP_API_FUNCNAME[apitype], args, dbody)

    return kfunc * "\n\n" * dfunc
end


###### END of CODEGEN #######
