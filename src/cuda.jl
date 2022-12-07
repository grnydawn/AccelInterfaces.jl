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
            args::NTuple{N, JaiDataType} where {N},
            names::NTuple{N, String} where {N}
        ) :: String

    reinterpret = String[]

    for (arg, varname) in zip(args, names)

        if arg isa AbstractArray || arg isa Tuple
            typestr, dimstr = cpp_typedef(arg, varname)
            push!(reinterpret,  typestr * " (*ptr_" * varname * ")" * dimstr *
                                " = " * "reinterpret_cast<" * typestr * " (*)" *
                                dimstr * ">(d_" * varname * ");\n")
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

function cuda_decls(buildtype::BuildType,
                args::NTuple{N, JaiDataType} where {N},
                names::NTuple{N, String} where {N}) :: String

    _decls = String[]

    for (arg, varname) in zip(args, names)

        if arg isa AbstractArray || arg isa Tuple

            typestr = typemap_j2c[eltype(arg)]

            if buildtype == JAI_ALLOCATE
                push!(_decls, "$(typestr) * d_$(varname);\n")

            else
                push!(_decls, "extern $(typestr) * d_$(varname);\n")

            end
        end
    end

    return join(_decls, "\n")
end

function gencode_cpp_cuda_kernel(kinfo::KernelInfo, launchid::String,
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
    launchconf = join(get!(cudaopts, "chevron", (1, 1)), ", ")
    launchargs = cuda_launchargs(args, names)
    reinterpret = cuda_reinterpret(args, names)

    return code = """
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

$(params)

$(macros)

extern "C" {

$(decls)

__global__ void jai_kernel(double X[4][3][2], double Y[4][3][2], double Z[4][3][2]) {
    $(kernelbody)
}

int64_t jai_launch($(kernelargs)) {
    int64_t res;

    $(reinterpret)

    jai_kernel<<<$(launchconf)>>>($(launchargs));

    res = 0;

    return res;

}

}
"""
end

function gencode_cpp_cuda_accel() :: String

    return code = """
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {

int64_t jai_get_num_devices(int64_t buf[]) {
    int64_t res;

    buf[0] = 1;

    res = 0;

    return res;

}

int64_t jai_get_device_num(int64_t buf[]) {
    int64_t res;

    buf[0] = 0;

    res = 0;

    return res;

}

int64_t jai_set_device_num(int64_t buf[]) {
    int64_t res;

    res = 0;

    return res;

}

int64_t jai_wait() {
    int64_t res;

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
                push!(apicalls, "cudaMalloc((void**)&d_$(varname), sizeof($(typestr)) * $(N));\n")

            elseif buildtype == JAI_UPDATETO
                push!(apicalls, "cudaMemcpy(d_$(varname), $(varname), sizeof($(typestr)) * $(N), cudaMemcpyHostToDevice);\n")

            elseif buildtype == JAI_UPDATEFROM
                push!(apicalls, "cudaMemcpy($(varname), d_$(varname), sizeof($(typestr)) * $(N), cudaMemcpyDeviceToHost);\n")

            elseif buildtype == JAI_DEALLOCATE
                push!(apicalls, "cudaFree(d_$(varname));\n")

            else

                error(string("$(buildtype) is not supported yet."))
            end

        end
    end

    return join(apicalls, "\n")

end

function gencode_cpp_cuda_directive(ainfo::AccelInfo, buildtype::BuildType,
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
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {

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

