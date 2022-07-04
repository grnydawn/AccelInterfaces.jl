
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

const f_part1 = """
USE, INTRINSIC :: ISO_C_BINDING

"""

const f_part2 = """

public launch

contains

"""

const f_part3 = """

    USE, INTRINSIC :: ISO_C_BINDING

"""

const f_part4 = """

    INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

"""

const f_part5 = """

    launch = JAI_ERRORCODE

END FUNCTION

end module

"""

function dimensions(arg)

    dimlist = []

    if arg isa OffsetArray
        for (offset, length) in zip(arg.offsets, size(arg))
            lbound = 1 + offset
            ubound = length + offset 
            push!(dimlist, "$lbound:$ubound")
        end
    elseif arg isa AbstractArray
        for length in size(arg)
            push!(dimlist, "1:$length")
        end

    else
        error("argument is not array type.")

    end

    join(dimlist, ", ")
end

function typedef(arg)

    dimstr = ""

    if arg isa AbstractArray
        typestr = typemap_j2f[eltype(arg)]

        dimlist = []
        if arg isa OffsetArray
            for (offset, length) in zip(arg.offsets, size(arg))
                lbound = 1 + offset
                ubound = length + offset 
                push!(dimlist, "$lbound:$ubound")
            end
        else
            for length in size(arg)
                push!(dimlist, "1:$length")
            end
        end

        dimstr = ", DIMENSION(" * join(dimlist, ", ") * ")"

    else
        typestr = typemap_j2f[typeof(arg)]

    end

    typestr, dimstr
end


function fortran_genparams(kinfo::KernelInfo)
    return fortran_genparams(kinfo.accel)

end

function fortran_genparams(ainfo::AccelInfo)

    typedecls = []

    for (name, value) in zip(ainfo.constnames, ainfo.constvars)

        typestr, dimstr = typedef(value)
        push!(typedecls, typestr * dimstr * ", PARAMETER :: " * name * " = " * string(value))
    end

    return join(typedecls, "\n")
end

function fortran_genvars(kinfo::KernelInfo, launchid::String, inargs::Vector,
                outargs::Vector, innames::NTuple, outnames::NTuple)

    onames = []
    for (index, ovar) in enumerate(outargs)
        if !(ovar in inargs)
            push!(onames, outnames[index])
        end
    end

    arguments = join((innames...,onames...), ",")

    typedecls = []

    for (arg, varname) in zip(inargs, innames)

        typestr, dimstr = typedef(arg)

        if arg in outargs
            intentstr = ", INTENT(INOUT)"

        else
            intentstr = ", INTENT(IN)"

        end

        push!(typedecls, typestr * dimstr * intentstr * " :: " * varname)
    end

    for (arg, varname) in zip(outargs, outnames)
        if !(arg in inargs)
            typestr, dimstr = typedef(arg)
            intentstr = ", INTENT(OUT)"
            push!(typedecls, typestr * dimstr * intentstr * " :: " * varname)
        end
    end

    return arguments, join(typedecls, "\n")
end

function fortran_typedecls(launchid::String, buildtype::BuildType, inargs::Vector, innames::NTuple)

    typedecls = []

    for (arg, varname) in zip(inargs, innames)

        typestr, dimstr = typedef(arg)
        intent = buildtype in (JAI_DEALLOCATE, JAI_COPYOUT) ? "INOUT" : "INOUT"

        push!(typedecls, typestr * dimstr * ", INTENT(" * intent * ") :: " * varname)
    end

    return join(typedecls, "\n")
end


function gencode_fortran_kernel(kinfo::KernelInfo, launchid::String, kernelbody::String,
                inargs::Vector, outargs::Vector, innames::NTuple, outnames::NTuple)

    params = fortran_genparams(kinfo)
    arguments, typedecls = fortran_genvars(kinfo, launchid, inargs, outargs,
                                innames, outnames)

    return code = """
module mod$(launchid)
USE, INTRINSIC :: ISO_C_BINDING

$(params)

public jai_launch

contains

INTEGER (C_INT64_T) FUNCTION jai_launch($(arguments)) BIND(C, name="jai_launch")
USE, INTRINSIC :: ISO_C_BINDING

$(typedecls)

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

$(kernelbody)

jai_launch = JAI_ERRORCODE

END FUNCTION

end module

"""
end
