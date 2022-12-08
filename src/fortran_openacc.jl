
function fortran_openacc_directives(buildtype::BuildType,
                inargs::NTuple{N, JaiDataType} where {N},
                innames::NTuple{N, String} where {N},
                control::Vector{String}) :: String

    directs = String[]
    clauses = join([c for c in control if c in ("async",)], " ")

    for (arg, varname) in zip(inargs, innames)

        # for debug

        if buildtype == JAI_ALLOCATE
            push!(directs, "!\$acc enter data create($(varname)) $(clauses)\n")

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
                launchid::String,
                args::NTuple{N, JaiDataType} where {N},
                names::NTuple{N, String} where {N},
                control::Vector{String}) :: String

    params = fortran_genparams(ainfo)
    typedecls = fortran_directive_typedecls(launchid, buildtype, args, names)
    directives = fortran_openacc_directives(buildtype, args[2:end], names[2:end], control)
    arguments = join(names, ",")
    funcname = LIBFUNC_NAME[ainfo.acceltype][buildtype]
 
    return """
module mod$(launchid)
USE, INTRINSIC :: ISO_C_BINDING
USE OPENACC

$(params)

public $(funcname)

contains

INTEGER (C_INT64_T) FUNCTION $(funcname)($(arguments)) BIND(C, name="$(funcname)")
USE, INTRINSIC :: ISO_C_BINDING

$(typedecls)

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

$(directives)

$(funcname) = JAI_ERRORCODE

END FUNCTION

end module

"""
end


function gencode_fortran_openacc_accel(accelid::String) :: String

    return code = """
module mod_$(accelid)_accel
USE, INTRINSIC :: ISO_C_BINDING
USE OPENACC

public jai_get_num_devices, jai_get_device_num, jai_set_device_num
public jai_device_init, jai_device_fini
public jai_wait

contains

INTEGER (C_INT64_T) FUNCTION jai_device_init(buf) BIND(C, name="jai_device_init")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T), DIMENSION(1), INTENT(OUT) :: buf
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

jai_device_init = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_device_fini(buf) BIND(C, name="jai_device_fini")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T), DIMENSION(1), INTENT(OUT) :: buf
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

jai_device_fini = JAI_ERRORCODE

END FUNCTION


INTEGER (C_INT64_T) FUNCTION jai_get_num_devices(buf) BIND(C, name="jai_get_num_devices")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T), DIMENSION(1), INTENT(OUT) :: buf
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

!buf(1) = acc_get_num_devices(acc_get_device_type())
buf(1) = acc_get_num_devices(acc_device_default)

jai_get_num_devices = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_get_device_num(buf) BIND(C, name="jai_get_device_num")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T), DIMENSION(1), INTENT(OUT) :: buf
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

!buf(1) = acc_get_device_num(acc_get_device_type())
buf(1) = acc_get_device_num(acc_device_default)

jai_get_device_num = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_set_device_num(buf) BIND(C, name="jai_set_device_num")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T), DIMENSION(1), INTENT(IN) :: buf
INTEGER :: device_number
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

device_number = buf(1)
!CALL acc_set_device_num(device_number, acc_get_device_type())
CALL acc_set_device_num(device_number, acc_device_default)

jai_set_device_num = JAI_ERRORCODE

END FUNCTION

INTEGER (C_INT64_T) FUNCTION jai_wait() BIND(C, name="jai_wait")
USE, INTRINSIC :: ISO_C_BINDING

INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

!CALL acc_wait(INTEGER(acc_get_default_async()))
CALL acc_wait(0)

jai_wait = JAI_ERRORCODE

END FUNCTION

end module

"""
end
