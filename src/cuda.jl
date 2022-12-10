function cuda_genparams(ainfo::AccelInfo) :: String

    params = String[]

    for (name, value) in zip(ainfo.const_names, ainfo.const_vars)

        typestr, dimstr = cpp_typedef(value, name)
        push!(params, "__device__ const " * name * " " * typestr * dimstr *
                " = " * cpp_genvalue(value) * ";")
    end

    return join(params, "\n")
end

function cuda_reinterpret(
            aid::String,
            args::NTuple{N, JaiDataType} where {N},
            names::NTuple{N, String} where {N}
        ) :: String

    reinterpret = String[]

    for (arg, varname) in zip(args, names)

        if arg isa AbstractArray || arg isa Tuple
            typestr, dimstr = cpp_typedef(arg, varname)
            push!(reinterpret,  typestr * " (*ptr_" * varname * ")" * dimstr *
                                " = " * "reinterpret_cast<" * typestr * " (*)" *
                                dimstr * ">(jai_dev_" * varname * "_" * aid * ");\n")
        end
    end

    return join(reinterpret, "\n")
end


function cuda_launchargs(
            args::NTuple{N, JaiDataType} where {N},
            names::NTuple{N, String} where {N}
        ) :: String

    launch = String[]

    for (arg, varname) in zip(args, names)

        if arg isa AbstractArray || arg isa Tuple
            push!(launch,  "*ptr_" * varname)
        end
    end

    return join(launch, ", ")
end

function cuda_genmacros(
            args::NTuple{N, JaiDataType} where {N},
            names::NTuple{N, String} where {N}
        ) :: String

    macros = String[]

    push!(macros, "#define JSHAPE(varname, dim) jai_shape_##varname##dim")
    push!(macros, "#define JSIZE(varname) jai_size_##varname")

    for (name, arg) in zip(names, args)
        if arg isa AbstractArray
            accum = 1
            for (idx, len) in enumerate(reverse(size(arg)))
                strlen = string(len)
                push!(macros, "__device__ const uint32_t jai_shape_" * name * string(idx-1) * " = " * strlen * ";" )
                accum *= len
            end
            push!(macros, "__device__ const uint32_t jai_size_" * name * " = " * string(accum) * ";" )

        elseif arg isa Tuple
            strlen = string(length(arg))
            push!(macros, "__device__ const uint32_t jai_shape_" * name * "0 = " * strlen * ";" )
            push!(macros, "__device__ const uint32_t jai_size_" * name * " = " * strlen * ";" )

        end
    end

    return join(macros, "\n")
end

function cuda_decls(
                aid::String,
                buildtype::BuildType,
                args::NTuple{N, JaiDataType} where {N},
                names::NTuple{N, String} where {N}) :: String

    _decls = String[]

    for (arg, varname) in zip(args, names)

        if arg isa AbstractArray || arg isa Tuple

            typestr = typemap_j2c[eltype(arg)]

            if buildtype == JAI_ALLOCATE
                push!(_decls, "$(typestr) * jai_dev_$(varname)_$(aid);\n")

            else
                push!(_decls, "extern $(typestr) * jai_dev_$(varname)_$(aid);\n")

            end
        end
    end

    return join(_decls, "\n")
end

function gencode_cpp_cuda_kernel(
                kinfo::KernelInfo,
                lid::String,
                cudaopts::Dict{String, T} where T <: Any,
                kernelbody::String,
                inargs::NTuple{N, JaiDataType} where {N},
                outargs::NTuple{M, JaiDataType} where {M},
                innames::NTuple{N, String} where {N},
                outnames::NTuple{M, String} where {M}) :: String

    args, names = merge_args(inargs, outargs, innames, outnames)
    aid = kinfo.accel.accelid[1:_IDLEN]

    params = cuda_genparams(kinfo.accel)
    decls = cuda_decls(aid, JAI_LAUNCH, args, names)
    kernelargs = cpp_genargs(args, names)
    macros = cuda_genmacros(args, names)
    launchargs = cuda_launchargs(args, names)
    reinterpret = cuda_reinterpret(aid, args, names)

    _grid, _block = get(cudaopts, "chevron", (1, 1))
    grid = _grid isa Number ? string(_grid) : join(_grid, ", ")
    block = _block isa Number ? string(_block) : join(_block, ", ")

    if kinfo.accel.acceltype in (JAI_FORTRAN_OPENACC, )
        stream = ""
        launchstmt = "jai_kernel<<<dim3($(grid)), dim3($(block))>>>($(launchargs));"
    else
        stream = "extern cudaStream_t jai_stream_$(aid);"
        launchstmt = "jai_kernel<<<dim3($(grid)), dim3($(block)), 0, jai_stream_$(aid)>>>($(launchargs));"
    end


    return code = """
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

$(params)

$(macros)

extern "C" {

$(stream)

$(decls)

__global__ void jai_kernel(double X[4][3][2], double Y[4][3][2], double Z[4][3][2]) {
    $(kernelbody)
}

int64_t jai_launch_$(lid)($(kernelargs)) {
    int64_t res;

    $(reinterpret)

    $(launchstmt)

    res = 0;

    return res;

}

}
"""
end

function gencode_cpp_cuda_accel(aid::String) :: String

    return code = """
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {

cudaStream_t jai_stream_$(aid);


int64_t jai_device_init_$(aid)(int64_t buf[]) {

    int64_t res;

    cudaStreamCreate(&jai_stream_$(aid));

    res = 0;

    return res;
}

int64_t jai_device_fini_$(aid)(int64_t buf[]) {

    int64_t res;

    cudaStreamDestroy(jai_stream_$(aid));

    res = 0;

    return res;
}

int64_t jai_get_num_devices_$(aid)(int64_t buf[]) {
    int64_t res;
    int count;

    cudaGetDeviceCount(&count);
    buf[0] = (int64_t)count;

    res = 0;

    return res;

}

int64_t jai_get_device_num_$(aid)(int64_t buf[]) {
    int64_t res;
    int num;

    cudaGetDevice(&num);
    buf[0] = (int64_t)num;

    res = 0;

    return res;

}

int64_t jai_set_device_num_$(aid)(int64_t buf[]) {
    int64_t res;

    res = 0;

    cudaSetDevice((int)buf[0]);

    return res;

}

int64_t jai_wait_$(aid)() {
    int64_t res;

    cudaDeviceSynchronize();

    res = 0;

    return res;

}

}
"""
end

function cpp_cuda_apicalls(
                aid::String,
                buildtype::BuildType,
                args::NTuple{N, JaiDataType} where {N},
                names::NTuple{N, String} where {N},
                control::Vector{String}) :: String

    apicalls = String[]


    for (arg, varname) in zip(args, names)
        if arg isa AbstractArray || arg isa Tuple
            typestr = typemap_j2c[eltype(arg)]

            N = prod(size(arg))

            if buildtype == JAI_ALLOCATE
                push!(apicalls, "cudaMalloc((void**)&jai_dev_$(varname)_$(aid), sizeof($(typestr)) * $(N));\n")

            elseif buildtype == JAI_UPDATETO
                if "async" in control
                    push!(apicalls, "cudaMemcpyAsync(jai_dev_$(varname)_$(aid), $(varname), sizeof($(typestr)) * $(N), cudaMemcpyHostToDevice, jai_stream_$(aid));\n")

                else
                    push!(apicalls, "cudaMemcpy(jai_dev_$(varname)_$(aid), $(varname), sizeof($(typestr)) * $(N), cudaMemcpyHostToDevice);\n")
                end

            elseif buildtype == JAI_UPDATEFROM
                if "async" in control
                    push!(apicalls, "cudaMemcpyAsync($(varname), jai_dev_$(varname)_$(aid), sizeof($(typestr)) * $(N), cudaMemcpyDeviceToHost, jai_stream_$(aid));\n")
                else
                    push!(apicalls, "cudaMemcpy($(varname), jai_dev_$(varname)_$(aid), sizeof($(typestr)) * $(N), cudaMemcpyHostToDevice);\n")

                end

            elseif buildtype == JAI_DEALLOCATE
                push!(apicalls, "cudaFree(jai_dev_$(varname)_$(aid));\n")

            else

                error(string("$(buildtype) is not supported yet."))
            end

        end
    end

    return join(apicalls, "\n")

end

function gencode_cpp_cuda_directive(
                ainfo::AccelInfo,
                buildtype::BuildType,
                lid::String,
                args::NTuple{N, JaiDataType} where {N},
                names::NTuple{N, String} where {N},
                control::Vector{String}) :: String

    aid = ainfo.accelid[1:_IDLEN]

    params = cuda_genparams(ainfo)
    decls = cuda_decls(aid, buildtype, args, names)
    funcargs = cpp_genargs(args, names)
    apicalls = cpp_cuda_apicalls(aid, buildtype, args, names, control)
    funcname = LIBFUNC_NAME[ainfo.acceltype][buildtype]


     return code = """
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {

extern cudaStream_t jai_stream_$(aid);

$(decls)

$(params)

int64_t $(funcname)_$(lid)($(funcargs)) {
    int64_t res, JAI_ERRORCODE;
    JAI_ERRORCODE = 0;

    $(apicalls)

    res = JAI_ERRORCODE;

    return res;

}

}
"""
end


