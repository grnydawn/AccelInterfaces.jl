
function fortran_omptarget_directives(buildtype::BuildType,
                inargs::NTuple{N, JaiDataType} where {N},
                innames::NTuple{N, String} where {N}) :: String

    directs = String[]

    for (arg, varname) in zip(inargs, innames)

        # for debug

        if buildtype == JAI_ALLOCATE
            push!(directs, "!\$omp target enter data map(alloc: $(varname))\n")

        elseif buildtype == JAI_UPDATETO
            push!(directs, "!\$omp target update to($(varname))\n")

        elseif buildtype == JAI_UPDATEFROM
            push!(directs, "!\$omp target update from($(varname))\n")

        elseif buildtype == JAI_DEALLOCATE
            push!(directs, "!\$omp target exit data map(delete: $(varname))\n")

        else
            error(string(buildtype) * " is not supported.")
        end
    end

    return join(directs, "\n")

end

function gencode_fortran_omptarget(ainfo::AccelInfo, buildtype::BuildType,
                launchid::String,
                args::NTuple{N, JaiDataType} where {N},
                names::NTuple{N, String} where {N}) :: String

    params = fortran_genparams(ainfo)
    typedecls = fortran_directive_typedecls(launchid, buildtype, args, names)
    directives = fortran_omptarget_directives(buildtype, args[2:end], names[2:end])
    arguments = join(names, ",")
    funcname = LIBFUNC_NAME[ainfo.acceltype][buildtype]
 
    return """
module mod$(launchid)
USE, INTRINSIC :: ISO_C_BINDING
USE OMP_LIB

$(params)

public $(funcname)

contains

INTEGER (C_INT64_T) FUNCTION $(funcname)($(arguments)) BIND(C, name="$(funcname)")
USE, INTRINSIC :: ISO_C_BINDING

$(typedecls)

!INTEGER :: jai_devnum
!INTEGER(ACC_DEVICE_KIND) :: jai_devtype
INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0

!jai_devtype = acc_get_device_type()
!jai_devnum = acc_get_device_num(jai_devtype)

!IF (jai_arg_device_num .GE. 0 .AND. jai_devnum .NE. jai_arg_device_num) THEN
!    CALL acc_set_device_num(INT(jai_arg_device_num, KIND(jai_devnum)), jai_devtype)
!END IF

$(directives)

!IF (jai_arg_device_num .GE. 0 .AND. jai_devnum .NE. jai_arg_device_num) THEN
!    CALL acc_set_device_num(jai_devnum, jai_devtype)
!END IF

$(funcname) = JAI_ERRORCODE

END FUNCTION

end module

"""
end
