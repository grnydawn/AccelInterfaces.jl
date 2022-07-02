
# Julia type to Fortran type conversion table
const typemap_j2f = Dict(
    Int8    => "INTEGER (C_INT8_T  )",
    Int16   => "INTEGER (C_INT16_T )",
    Int32   => "INTEGER (C_INT32_T )",
    Int64   => "INTEGER (C_INT64_T )",
    Int128  => "INTEGER (C_INT128_T)",
    UInt8   => "INTEGER (C_INT8_T  )",
    UInt16  => "INTEGER (C_INT16_T )",
    UInt32  => "INTEGER (C_INT32_T )",
    UInt64  => "INTEGER (C_INT64_T )",
    UInt128 => "INTEGER (C_INT128_T)",
    Float32	=> "REAL (C_FLOAT  )",
    Float64	=> "REAL (C_DOUBLE )"
)

const kpart1 = """
module kernelmod
USE, INTRINSIC :: ISO_C_BINDING

"""

const kpart2 = """

public launch

contains

"""

const kpart3 = """

    USE, INTRINSIC :: ISO_C_BINDING

"""

const kpart4 = """

    INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

"""

const kpart5 = """

    launch = JAI_ERRORCODE

END FUNCTION

end module

"""

function typedef(arg)

    dimstr = ""

    if arg isa AbstractArray
        typestr = typemap_j2f[eltype(arg)]

        dimlist = []
        if arg isa OffsetArray
            for (offset, length) in zip(arg.offset, size(arg))
                lbound = 1 + offset
                ubound = length + offset 
                push!(dimlist, "$lbound:$ubound")
            end
        else
            for length in size(arg)
                push!(dimlist, string(length))
            end
        end

        dimstr = ", DIMENSION(" * join(dimlist, ", ") * ")"

    else
        typestr = typemap_j2f[typeof(arg)]

    end

    typestr, dimstr
end


function genparams(kinfo::KernelInfo)

    typedecls = []

    for (name, value) in zip(kinfo.accel.constnames, kinfo.accel.constvars)

        typestr, dimstr = typedef(value)
        push!(typedecls, typestr * dimstr * ", PARAMETER :: " * name * " = " * string(value))
    end

    return join(typedecls, "\n")
end

function genvars(kinfo::KernelInfo, hashid::UInt64, inargs::Vector, indtypes::Vector,
                outargs::Vector, outdtypes::Vector, innames::NTuple, outnames::NTuple)

    onames = []
    for (index, ovar) in enumerate(outargs)
        if !(ovar in inargs)
            push!(onames, outnames[index])
        end
    end

    arguments = join((innames...,onames...), ",")

    funcsig = "INTEGER (C_INT64_T) FUNCTION launch($arguments) BIND(C, name=\"launch\")\n"
    typedecls = []

    for (arg, dtype, varname) in zip(inargs, indtypes, innames)

        typestr, dimstr = typedef(arg)

        if arg in outargs
            intentstr = ", INTENT(INOUT)"

        else
            intentstr = ", INTENT(IN)"

        end

        push!(typedecls, typestr * dimstr * intentstr * " :: " * varname)
    end

    for (arg, dtype, varname) in zip(outargs, outdtypes, outnames)
        if !(arg in inargs)
            typestr, dimstr = typedef(arg)
            intentstr = ", INTENT(OUT)"
            push!(typedecls, typestr * dimstr * intentstr * " :: " * varname)
        end
    end

    return funcsig, join(typedecls, "\n")
end

function gencode_fortran(kinfo::KernelInfo, hashid::UInt64, inargs::Vector, indtypes::Vector,
                outargs::Vector, outdtypes::Vector, innames::NTuple, outnames::NTuple)

    open(kinfo.kernelpath, "r") do io
        kernelbody = read(io, String)

        params = genparams(kinfo)
        funcsig, typedecls = genvars(kinfo, hashid, inargs, indtypes,
                                    outargs, outdtypes, innames, outnames)

        return (kpart1 * params * kpart2 * funcsig * kpart3 * typedecls * kpart4 *
                kernelbody * kpart5)
    end
end
