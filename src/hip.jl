
function gencode_cpp_hip_kernel(kinfo::KernelInfo, launchid::String,
                cudaopts::Dict{String, T} where T <: Any,
                kernelbody::String,
                inargs::NTuple{N, JaiDataType} where {N},
                outargs::NTuple{M, JaiDataType} where {M},
                innames::NTuple{N, String} where {N},
                outnames::NTuple{M, String} where {M}) :: String

    args, names = merge_args(inargs, outargs, innames, outnames)

    params = cuda_genparams(kinfo.accel)
    decls = cuda_decls(JAI_LAUNCH, args, names)
    kernelargs = cpp_genargs(args, names)
    macros = cuda_genmacros(args, names)
    launchargs = cuda_launchargs(args, names)
    reinterpret = cuda_reinterpret(args, names)

    _grid, _block = get!(cudaopts, "chevron", (1, 1))
    grid = _grid isa Number ? string(_grid) : join(_grid, ", ")
    block = _block isa Number ? string(_block) : join(_block, ", ")

    return code = """
#include <stdint.h>
#include <hip/hip_runtime.h>

$(params)

$(macros)

extern "C" {

extern hipStream_t jai_stream;

$(decls)

__global__ void jai_kernel(double X[4][3][2], double Y[4][3][2], double Z[4][3][2]) {
    $(kernelbody)
}

int64_t jai_launch($(kernelargs)) {
    int64_t res;

    $(reinterpret)

    hipLaunchKernelGGL(jai_kernel, dim3($(grid)), dim3($(block)), 0, jai_stream, $(launchargs));

    res = 0;

    return res;

}

}
"""
end

function gencode_cpp_hip_accel() :: String

    return code = """
#include <stdint.h>
#include <hip/hip_runtime.h>

extern "C" {

hipStream_t jai_stream;


int64_t jai_device_init(int64_t buf[]) {

    int64_t res;

    hipStreamCreate(&jai_stream);

    res = 0;

    return res;
}

int64_t jai_device_fini(int64_t buf[]) {

    int64_t res;

    hipStreamDestroy(jai_stream);

    res = 0;

    return res;
}

int64_t jai_get_num_devices(int64_t buf[]) {
    int64_t res;
    int count;

    hipGetDeviceCount(&count);
    buf[0] = (int64_t)count;

    res = 0;

    return res;
}

int64_t jai_get_device_num(int64_t buf[]) {
    int64_t res;
    int num;

    hipGetDevice(&num);
    buf[0] = (int64_t)num;

    res = 0;

    return res;

}

int64_t jai_set_device_num(int64_t buf[]) {
    int64_t res;

    res = 0;

    hipSetDevice((int)buf[0]);

    return res;

}

int64_t jai_wait() {
    int64_t res;

    hipDeviceSynchronize();

    res = 0;

    return res;

}

}
"""
end

function cpp_cuda_apicalls(buildtype::BuildType,
                args::NTuple{N, JaiDataType} where {N},
                names::NTuple{N, String} where {N},
                control::Vector{String}) :: String

    apicalls = String[]


    for (arg, varname) in zip(args, names)
        if arg isa AbstractArray || arg isa Tuple
            typestr = typemap_j2c[eltype(arg)]

            N = prod(size(arg))

            if buildtype == JAI_ALLOCATE
                #push!(apicalls, "$(typestr) * d_$(varname);\n")
                push!(apicalls, "hipMalloc((void**)&d_$(varname), sizeof($(typestr)) * $(N));\n")

            elseif buildtype == JAI_UPDATETO
                # hipMemcpyAsync()
                if "async" in control
                    push!(apicalls, "hipMemcpyAsync(d_$(varname), $(varname), sizeof($(typestr)) * $(N), hipMemcpyHostToDevice, jai_stream);\n")

                else
                    push!(apicalls, "hipMemcpyHtoD(d_$(varname), $(varname), sizeof($(typestr)) * $(N));\n")
                end

            elseif buildtype == JAI_UPDATEFROM
                if "async" in control
                    push!(apicalls, "hipMemcpyAsync($(varname), d_$(varname), sizeof($(typestr)) * $(N), hipMemcpyDeviceToHost, jai_stream);\n")
                else
                    push!(apicalls, "hipMemcpyDtoH($(varname), d_$(varname), sizeof($(typestr)) * $(N));\n")
                    #push!(apicalls, "printf(\"####### %f #####\\n\", $(varname)[0][0][0] );")

                end

            elseif buildtype == JAI_DEALLOCATE
                push!(apicalls, "hipFree(d_$(varname));\n")

            else

                error(string("$(buildtype) is not supported yet."))
            end

        end
    end

    return join(apicalls, "\n")

end

function gencode_cpp_hip_directive(ainfo::AccelInfo, buildtype::BuildType,
                launchid::String,
                args::NTuple{N, JaiDataType} where {N},
                names::NTuple{N, String} where {N},
                control::Vector{String}) :: String

    params = cuda_genparams(ainfo)
    decls = cuda_decls(buildtype, args, names)
    funcargs = cpp_genargs(args, names)
    apicalls = cpp_cuda_apicalls(buildtype, args, names, control)
    funcname = LIBFUNC_NAME[ainfo.acceltype][buildtype]


     return code = """
#include <stdint.h>
#include <hip/hip_runtime.h>

extern "C" {

extern hipStream_t jai_stream;

$(decls)

$(params)

int64_t $(funcname)($(funcargs)) {
    int64_t res, JAI_ERRORCODE;
    JAI_ERRORCODE = 0;

    $(apicalls)

    res = JAI_ERRORCODE;

    return res;

}

}
"""
end


