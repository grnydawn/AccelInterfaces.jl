
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


function cpp_typedef(arg::JaiDataType, name::String, macros::Vector{String}
            ) :: Tuple{String, String}

    dimstr = ""

    if arg isa AbstractArray
        typestr = typemap_j2c[eltype(arg)]
        local dimlist = String[]
        accum = 1

        for (idx, len) in enumerate(reverse(size(arg)))
            strlen = string(len)
            push!(dimlist, "[" * strlen * "]")
            push!(macros, "uint32_t jai_shape_" * name * string(idx-1) * " = " *
                    strlen * ";" )
            accum *= len
        end
        push!(macros, "uint32_t jai_size_" * name * " = " * string(accum) * ";" )

        dimstr = join(dimlist, "")

    elseif arg isa Tuple
        strlen = string(length(arg))
        typestr = typemap_j2c[eltype(arg)]
        dimstr = "[" * strlen * "]"
        push!(macros, "uint32_t jai_shape_" * name * "0 = " * strlen * ";" )
        push!(macros, "uint32_t jai_size_" * name * " = " * strlen * ";" )

    else
        typestr = typemap_j2c[typeof(arg)]

    end

    typestr, dimstr
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

function cpp_genparams(ainfo::AccelInfo, macros::Vector{String}) :: String

    params = String[]

    for (name, value) in zip(ainfo.const_names, ainfo.const_vars)

        typestr, dimstr = cpp_typedef(value, name, macros)
        push!(params, "const " * name * " " * typestr * dimstr * " = " * cpp_genvalue(value) * ";")
    end

    return join(params, "\n")
end


function cpp_genvars(kinfo::KernelInfo, macros::Vector{String},
                inargs::NTuple{N, JaiDataType} where {N},
                outargs::NTuple{M, JaiDataType} where {M},
                innames::NTuple{N, String} where {N},
                outnames::NTuple{M, String} where {M}) :: String

    onames = String[]

    for (index, oname) in enumerate(outnames)
        if !(oname in innames)
            push!(onames, oname)
        end
    end

    arguments = String[]

    for (arg, varname) in zip(inargs, innames)

        typestr, dimstr = cpp_typedef(arg, varname, macros)

        push!(arguments, typestr * " " * varname * dimstr)
    end

    for (arg, varname) in zip(outargs, outnames)
        if !(varname in innames)
            typestr, dimstr = cpp_typedef(arg, varname, macros)

            push!(arguments, typestr * " " * varname * dimstr)
        end
    end

    return join(arguments, ", ")
end

function gencode_cpp_kernel(kinfo::KernelInfo, launchid::String,
                kernelbody::String,
                inargs::NTuple{N, JaiDataType} where {N},
                outargs::NTuple{M, JaiDataType} where {M},
                innames::NTuple{N, String} where {N},
                outnames::NTuple{M, String} where {M}) :: String

    macros = String[]

    push!(macros, "#define JSHAPE(varname, dim) jai_shape_##varname##dim")
    push!(macros, "#define JSIZE(varname) jai_size_##varname")

    params = cpp_genparams(kinfo.accel, macros)
    kernelargs = cpp_genvars(kinfo, macros, inargs, outargs, innames, outnames)
    macrodefs = join(macros, "\n")

    return code = """
#include <stdint.h>

$(params)

$(macrodefs)

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

int64_t dummy() {
    int64_t res;

    res = 0;

    return res;

}

}
"""
end

