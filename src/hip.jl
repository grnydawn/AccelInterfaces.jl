# hip.jl: implement functions for HIP framework


HIP_TEMPLATE_KERNEL= """
__global__ void {kname}({kargs}) {{

{kbody}

}}
"""


###### START of CODEGEN #######

function code_cpp_header(
        frame       ::JAI_TYPE_HIP,
        apitype     ::JAI_TYPE_API,
        interop_frames  ::Vector{JAI_TYPE_FRAMEWORK},
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) ::String

    cpp_hdr = code_cpp_header(JAI_CPP, apitype, interop_frames, prefix, args, data)

    return "#include \"hip/hip_runtime.h\"\n" * cpp_hdr

end

function code_c_header(
        frame       ::JAI_TYPE_HIP,
        apitype     ::JAI_TYPE_API,
        interop_frames  ::Vector{JAI_TYPE_FRAMEWORK},
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) ::String

    return ""

end

function code_c_typedecl(arg::JAI_TYPE_ARG) :: Tuple{String, String, String}

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

    return typestr, vname, dimstr
end

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
        frame       ::JAI_TYPE_HIP,
        apitype     ::JAI_TYPE_ACCEL,
        interop_frames  ::Vector{JAI_TYPE_FRAMEWORK},
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N,
    ) :: String

    funcs = Vector{String}(undef, length(JAI_ACCEL_FUNCTIONS))

    for (i, (name, inout)) in enumerate(JAI_ACCEL_FUNCTIONS)
        funcs[i] = code_c_function(prefix, name, args, "")
    end

    return  join(funcs, "\n\n")

end

###### START of DATA #######

function code_c_functions(
        frame       ::JAI_TYPE_HIP,
        apitype     ::JAI_TYPE_API_DATA,
        interop_frames  ::Vector{JAI_TYPE_FRAMEWORK},
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N
    ) :: String

    return code_c_function(prefix, JAI_MAP_API_FUNCNAME[apitype], args, "")
end

###### START of LAUNCH #######

#__global__ void 
#vectoradd_float(float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, int width, int height) 
#
#  {
# 
#      int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
#      int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
#
#      int i = y * width + x;
#      if ( i < (width * height)) {
#        a[i] = b[i] + c[i];
#      }
#
#
#
#  }

function code_hip_kernel(
        kname       ::String,
        args        ::JAI_TYPE_ARGS,
        kbody       ::String
    ) :: String

    kargs = code_c_dummyargs(args)

    return jaifmt(HIP_TEMPLATE_KERNEL, kname=kname, kargs=kargs, kbody=kbody)
end


function code_hip_driver_body(
        kname      ::String,
        args        ::JAI_TYPE_ARGS,
        config   ::Union{OrderedDict{String, JAI_TYPE_CONFIG_VALUE}, Nothing} = nothing
    ) :: String

    out = Vector{String}()

    shared = 0
    stream = 0
    varnames = join((a[3] for a in args), ", ")

    lcfg = ["1", "1", "0", "0"]

    if config == nothing
        
    elseif config isa Integer
        lcfg[2] = string(config)

    elseif config isa Tuple && length(config) > 0

        for (i, cfg) in config
            if cfg isa Tuple
                lcfg[i] = join((string(c) for c in cfg), ", ")
            else
                lcfg[i] = string(cfg)
            end
        end

    else
        error("Wrong launch config syntax: " * string(config))
    end

    grid, block, shared, stream = lcfg

    nargs   = length(args) - 1
    buf     = fill("", nargs)
    anames  = fill("", nargs)
    dname   = args[end][3]

    # (var, dtype, vname, vinout, addr, vshape, voffset)
    for (i, arg) in enumerate(args[1:end-1])

        if arg[1] isa AbstractArray
            t, n, d = code_c_typedecl(arg)
            buf[i] = "$t (*ptr_$n)$d = reinterpret_cast<$t (*)$d>($dname[$(i-1)]);"
        end

        anames[i] = "*ptr_" * arg[3] 
    end

    reintepret  = join(buf, "\n")
    dvarnames   = join(anames, ", \n")

    push!(out, """
$reintepret

hipLaunchKernelGGL(
    $kname, 
    dim3($grid), dim3($block), $shared, $stream,
    $dvarnames
);
""")

    return join(out, "\n")
end


function code_c_functions(
        frame       ::JAI_TYPE_HIP,
        apitype     ::JAI_TYPE_LAUNCH,
        interop_frames  ::Vector{JAI_TYPE_FRAMEWORK},
        prefix      ::String,
        args        ::JAI_TYPE_ARGS,
        data        ::NTuple{N, JAI_TYPE_DATA} where N,
        launch_config   ::Union{OrderedDict{String, JAI_TYPE_CONFIG_VALUE}, Nothing} = nothing
    ) :: String

    kname = prefix * "device"

    # kernel function
    kfunc = code_hip_kernel(kname, args[1:end-1], data[1])

    # driver function
    dbody = code_hip_driver_body(kname, args, launch_config)
    dfunc = code_c_function(prefix, JAI_MAP_API_FUNCNAME[apitype], args, dbody)

    return kfunc * "\n\n" * dfunc
end


###### END of CODEGEN #######
