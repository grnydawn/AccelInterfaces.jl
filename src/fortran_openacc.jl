
function fortran_openacc_directives(
                aid::String,
                buildtype::BuildType,
                inargs::NTuple{N, JaiDataType} where {N},
                innames::NTuple{N, String} where {N},
                control::Vector{String}) :: String

    directs = String[]
    clauses = join([c for c in control if c in ("async",)], " ")

    for (arg, varname) in zip(inargs, innames)

        # for debug

        if buildtype == JAI_ALLOCATE
            push!(directs, "!\$acc enter data create($(varname)) $(clauses)\n")
            push!(directs, "jai_dev_$(varname)_$(aid) = acc_deviceptr($(varname))\n")

        elseif buildtype == JAI_UPDATETO
            push!(directs, "!\$acc update device($(varname)) $(clauses)\n")

        elseif buildtype == JAI_UPDATEFROM
            push!(directs, "!\$acc update host($(varname)) $(clauses)\n")

        elseif buildtype == JAI_DEALLOCATE
            push!(directs, "!\$acc exit data delete($(varname)) $(clauses)\n")

        else
            error(string(buildtype) * " is not supported.")
        end
    end

    return join(directs, "\n")

end

function gencode_fortran_openacc_directive(ainfo::AccelInfo, buildtype::BuildType,
                lid::String,
                args::NTuple{N, JaiDataType} where {N},
                names::NTuple{N, String} where {N},
                control::Vector{String}) :: String

    aid = ainfo.accelid[1:_IDLEN]

    params = fortran_genparams(ainfo)
    typedecls = fortran_directive_typedecls(lid, buildtype, args, names)
    directives = fortran_openacc_directives(aid, buildtype, args[2:end], names[2:end], control)
    arguments = join(names, ",")
    funcname = LIBFUNC_NAME[ainfo.acceltype][buildtype]

    deviceptrs = ""
    devicevars = ""

    if buildtype == JAI_ALLOCATE
        deviceptrs = join([("TYPE (C_PTR), BIND(C, NAME='jai_dev_$(v)_$(aid)') ::" *
                         " jai_dev_$(v)_$(aid)") for v in names[2:end]], "\n")
        devicevars = "public " * join(["jai_dev_$(v)_$(aid)" for v in names[2:end]], ",")
    end
    
    return """
module mod_$(funcname)_$(lid)
USE, INTRINSIC :: ISO_C_BINDING
USE OPENACC

$(params)

$(deviceptrs)

public $(funcname)_$(lid)
$(devicevars)

contains

INTEGER (C_INT64_T) FUNCTION $(funcname)_$(lid)($(arguments)) BIND(C, name="$(funcname)_$(lid)")
USE, INTRINSIC :: ISO_C_BINDING

$(typedecls)

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

$(directives)

$(funcname)_$(lid) = JAI_ERRORCODE

END FUNCTION

end module

"""
end


function gencode_fortran_openacc_accel(aid::String) :: String

    return code = """
module mod_accel_$(aid)
USE, INTRINSIC :: ISO_C_BINDING
USE OPENACC

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

!buf(1) = acc_get_num_devices(acc_get_device_type())
buf(1) = acc_get_num_devices(acc_device_default)

jai_get_num_devices_$(aid) = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_get_device_num_$(aid)(buf) BIND(C, name="jai_get_device_num_$(aid)")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T), DIMENSION(1), INTENT(OUT) :: buf
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

!buf(1) = acc_get_device_num(acc_get_device_type())
buf(1) = acc_get_device_num(acc_device_default)

jai_get_device_num_$(aid) = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_set_device_num_$(aid)(buf) BIND(C, name="jai_set_device_num_$(aid)")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T), DIMENSION(1), INTENT(IN) :: buf
INTEGER :: device_number
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

device_number = buf(1)
!CALL acc_set_device_num(device_number, acc_get_device_type())
CALL acc_set_device_num(device_number, acc_device_default)

jai_set_device_num_$(aid) = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_wait_$(aid)() BIND(C, name="jai_wait_$(aid)")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

!CALL acc_wait(INTEGER(acc_get_default_async()))
CALL acc_wait(0)

jai_wait_$(aid) = JAI_ERRORCODE

END FUNCTION

end module

"""
end
