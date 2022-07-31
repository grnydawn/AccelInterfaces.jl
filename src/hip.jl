
function gencode_hip_kernel(kinfo::KernelInfo, launchid::String,
                kernelbody::String,
                inargs::NTuple{N, JaiDataType} where {N},
                outargs::NTuple{M, JaiDataType} where {M},
                innames::NTuple{N, String} where {N},
                outnames::NTuple{M, String} where {M}) :: String

    macros = String[]

    push!(macros, "#define JSHAPE(varname, dim) jai_shape_##varname##dim")
    push!(macros, "#define JSIZE(varname) jai_size_##varname")

    params = cpp_genparams(kinfo.accel, macros)
    lanuchargs = cpp_genvars(kinfo, macros, inargs, outargs, innames, outnames)
    macrodefs = join(macros, "\n")

    return code = """
#include <stdint.h>
#include <hip/hip_runtime.h>

$(params)

$(macrodefs)

extern "C" {

int64_t jai_kernel($(kernelargs)){

    int64_t res;

    $(kernelbody)

    res = 0;

    return res;
}

int64_t jai_launch($(launchargs)) {
    int64_t res;

    // TODO: how to access device variables

    hipLaunchKernelGGL(jai_kernel, $(actualargs));

    res = 0;

    return res;

}

}
"""
end

function gencode_cpp_accel() :: String

    return code = """
#include <stdint.h>

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

}
"""
end

