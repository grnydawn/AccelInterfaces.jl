
# Julia type to C type conversion table
const typemap_j2c = Dict{DataType, String}(
    Int8    => "int8_t",
    Int16   => "int16_t",
    Int32   => "int32_t",
    Int64   => "int64_t",
    UInt8   => "uint8_t",
    UInt16  => "uint16_t",
    UInt32  => "uint32_t",
    UInt64  => "uint64_t",
    Float32	=> "float",
    Float64	=> "double"
)


function cpp_typedef(arg::JaiDataType, name::String) :: Tuple{String, String}

    dimstr = ""

    if arg isa AbstractArray
        typestr = typemap_j2c[eltype(arg)]
        local dimlist = String[]
        accum = 1

        for (idx, len) in enumerate(reverse(size(arg)))
            strlen = string(len)
            push!(dimlist, "[" * strlen * "]")
        end

        dimstr = join(dimlist, "")

    elseif arg isa Tuple
        strlen = string(length(arg))
        typestr = typemap_j2c[eltype(arg)]
        dimstr = "[" * strlen * "]"

    else
        typestr = typemap_j2c[typeof(arg)]

    end

    return typestr, dimstr
end

function cpp_genvalue(value::JaiConstType) :: String

    local valuelist = String[]

    if value isa NTuple
        for v in value
            push!(valuelist, string(v))
        end

    elseif value isa AbstractArray

        if ndims(value) == 1
            for v in value
                push!(valuelist, string(v))
            end
        else
            error("Multidimensional parameter array is not supported yet.")
        end
    else
        return string(value)
    end

    return "{ " * join(valuelist, ", ") * "}"

end

function cpp_genmacros(
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
                push!(macros, "uint32_t jai_shape_" * name * string(idx-1) * " = " * strlen * ";" )
                accum *= len
            end
            push!(macros, "uint32_t jai_size_" * name * " = " * string(accum) * ";" )

        elseif arg isa Tuple
            strlen = string(length(arg))
            push!(macros, "uint32_t jai_shape_" * name * "0 = " * strlen * ";" )
            push!(macros, "uint32_t jai_size_" * name * " = " * strlen * ";" )

        end
    end

    return join(macros, "\n")
end

function cpp_genparams(ainfo::AccelInfo) :: String

    params = String[]

    for (name, value) in zip(ainfo.const_names, ainfo.const_vars)

        typestr, dimstr = cpp_typedef(value, name)
        push!(params, "const " * name * " " * typestr * dimstr * " = " * cpp_genvalue(value) * ";")
    end

    return join(params, "\n")
end

function cpp_genargs(
                args::NTuple{N, JaiDataType} where {N},
                names::NTuple{N, String} where {N}) :: String

    arguments = String[]

    for (arg, varname) in zip(args, names)

        typestr, dimstr = cpp_typedef(arg, varname)

        push!(arguments, typestr * " " * varname * dimstr)
    end

    return join(arguments, ", ")
end

function gencode_cpp_kernel(kinfo::KernelInfo, launchid::String,
                cppopts::Dict{String, T} where T <: Any,
                kernelbody::String,
                inargs::NTuple{N, JaiDataType} where {N},
                outargs::NTuple{M, JaiDataType} where {M},
                innames::NTuple{N, String} where {N},
                outnames::NTuple{M, String} where {M}) :: String

    args, names = merge_args(inargs, outargs, innames, outnames)
    params = cpp_genparams(kinfo.accel)
    macros = cpp_genmacros(args, names)
    kernelargs = cpp_genargs(args, names)

    return code = """
#include <stdint.h>

$(params)

$(macros)

extern "C" {

int64_t jai_launch($(kernelargs)) {
    int64_t res;

    $(kernelbody)

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

int64_t jai_device_init(int64_t buf[]) {

    int64_t res;

    res = 0;

    return res;
}

int64_t jai_device_fini(int64_t buf[]) {

    int64_t res;

    res = 0;

    return res;
}


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

