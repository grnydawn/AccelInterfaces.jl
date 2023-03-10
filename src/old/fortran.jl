# Julia type to Fortran type conversion table
const typemap_j2f = Dict{DataType, String}(
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

function dimensions(arg::JaiDataType) :: String

    dimlist = String[]

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

function fortran_typedef(arg::JaiDataType) :: Tuple{String, String}

    dimstr = ""

    if arg isa AbstractArray
        typestr = typemap_j2f[eltype(arg)]
        local dimlist = String[]

        if arg isa OffsetArray
            for (offset, length) in zip(arg.offsets, size(arg))
                local lbound = 1 + offset
                local ubound = length + offset 
                push!(dimlist, "$lbound:$ubound")
            end
        else
            for length in size(arg)
                push!(dimlist, "1:$length")
            end
        end

        dimstr = ", DIMENSION(" * join(dimlist, ", ") * ")"

    elseif arg isa Tuple
        typestr = typemap_j2f[eltype(arg)]
        dimstr = ", DIMENSION(" * "1:$(length(arg))" * ")"

    else
        typestr = typemap_j2f[typeof(arg)]

    end

    typestr, dimstr
end


function fortran_genvalue(value::JaiConstType) :: String

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

    return "(/ " * join(valuelist, ", ") * "/)"

end

function fortran_genparams(ainfo::AccelInfo) :: String

    typedecls = String[]

    for (name, value) in zip(ainfo.const_names, ainfo.const_vars)

        typestr, dimstr = fortran_typedef(value)
        push!(typedecls, typestr * dimstr * ", PARAMETER :: " * name * " = " *
                fortran_genvalue(value))
    end

    return join(typedecls, "\n")
end

function fortran_genvars(kinfo::KernelInfo, lid::String,
                inargs::NTuple{N, JaiDataType} where {N},
                outargs::NTuple{M, JaiDataType} where {M},
                innames::NTuple{N, String} where {N},
                outnames::NTuple{M, String} where {M}) :: Tuple{String, String}

    onames = String[]

    for (index, oname) in enumerate(outnames)
        if !(oname in innames)
            push!(onames, oname)
        end
    end

    arguments = join((innames...,onames...), ",")

    typedecls = String[]

    for (arg, varname) in zip(inargs, innames)

        typestr, dimstr = fortran_typedef(arg)

        #if varname in outnames || arg in outargs
        if varname in outnames
            intentstr = ", INTENT(INOUT)"

        else
            intentstr = ", INTENT(IN)"

        end

        push!(typedecls, typestr * dimstr * intentstr * " :: " * varname)
    end

    for (arg, varname) in zip(outargs, outnames)
        if !(varname in innames)
            typestr, dimstr = fortran_typedef(arg)

            #if arg in inargs
            #    intentstr = ", INTENT(INOUT)"
            #else
                intentstr = ", INTENT(OUT)"
            #end

            push!(typedecls, typestr * dimstr * intentstr * " :: " * varname)
        end
    end

    return arguments, join(typedecls, "\n")
end

function fortran_directive_typedecls(
                launchid::String, buildtype::BuildType,
                args::NTuple{N, JaiDataType} where {N},
                names::NTuple{N, String} where {N};
                target::Bool=false
        ) :: String

    typedecls = String[]

    for (arg, varname) in zip(args, names)

        typestr, dimstr = fortran_typedef(arg)
        intent = buildtype in (JAI_DEALLOCATE, JAI_UPDATEFROM) ? "IN" : "IN"

        if target
            push!(typedecls, typestr * dimstr * ", INTENT(" * intent * "), TARGET :: " * varname)
        else
            push!(typedecls, typestr * dimstr * ", INTENT(" * intent * ") :: " * varname)
        end
    end

    return join(typedecls, "\n")
end

function gencode_fortran_kernel(
                kinfo::KernelInfo,
                lid::String,
                fortopts::Dict{String, T} where T <: Any,
                kernelbody::String,
                inargs::NTuple{N, JaiDataType} where {N},
                outargs::NTuple{M, JaiDataType} where {M},
                innames::NTuple{N, String} where {N},
                outnames::NTuple{M, String} where {M}) :: String

    params = fortran_genparams(kinfo.accel)
    arguments, typedecls = fortran_genvars(kinfo, lid, inargs, outargs,
                                innames, outnames)
    devnum = kinfo.accel.device_num
    aid = kinfo.accel.accelid[1:_IDLEN]

    return code = """
module mod_kernel_$(lid)
USE, INTRINSIC :: ISO_C_BINDING

!!INTEGER, PARAMETER :: JAI_DEVICE_NUM = $(devnum)
$(params)

public jai_launch_$(lid)

contains

INTEGER (C_INT64_T) FUNCTION jai_launch_$(lid)($(arguments)) BIND(C, name="jai_launch_$(lid)")
USE, INTRINSIC :: ISO_C_BINDING
IMPLICIT NONE

$(typedecls)

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

$(kernelbody)

jai_launch_$(lid) = JAI_ERRORCODE

END FUNCTION

end module

"""
end


function gencode_fortran_accel(aid::String) :: String

    return code = """
module mod_accel_$(aid)
USE, INTRINSIC :: ISO_C_BINDING

public jai_get_num_devices_$(aid), jai_get_device_num_$(aid), jai_set_device_num_$(aid)
public jai_device_init_$(aid), jai_device_fini_$(aid)
public jai_wait_$(aid)

contains

INTEGER (C_INT64_T) FUNCTION jai_device_init_$(aid)(buf) BIND(C, name="jai_device_init_$(aid)")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T), DIMENSION(1), INTENT(OUT) :: buf
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

jai_device_init_$(aid) = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_device_fini_$(aid)(buf) BIND(C, name="jai_device_fini_$(aid)")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T), DIMENSION(1), INTENT(OUT) :: buf
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

jai_device_fini_$(aid) = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_get_num_devices_$(aid)(buf) BIND(C, name="jai_get_num_devices_$(aid)")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T), DIMENSION(1), INTENT(OUT) :: buf
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

buf(1) = 1

jai_get_num_devices_$(aid) = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_get_device_num_$(aid)(buf) BIND(C, name="jai_get_device_num_$(aid)")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T), DIMENSION(1), INTENT(OUT) :: buf
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

buf(1) = 1

jai_get_device_num_$(aid) = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_set_device_num_$(aid)(buf) BIND(C, name="jai_set_device_num_$(aid)")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T), DIMENSION(1), INTENT(IN) :: buf
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

jai_set_device_num_$(aid) = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_wait_$(aid)() BIND(C, name="jai_wait_$(aid)")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

jai_wait_$(aid) = JAI_ERRORCODE

END FUNCTION

end module

"""
end