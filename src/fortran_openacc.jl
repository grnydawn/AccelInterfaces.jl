
function fortran_openacc_directives(buildtype::BuildType,
                inargs::NTuple{N, JaiDataType} where {N},
                innames::NTuple{N, String} where {N}) :: String

    directs = []

    for (arg, varname) in zip(inargs, innames)

        # for debug
        #push!(directs, "print *, \"LOC " * string(buildtype) * " of " * varname * " = \", LOC(" * varname * ")\n")

        if buildtype == JAI_ALLOCATE
            push!(directs, "!\$acc enter data create($(varname))\n")

        elseif buildtype == JAI_COPYIN
            push!(directs, "!\$acc update device($(varname))\n")

        elseif buildtype == JAI_COPYOUT
            push!(directs, "!\$acc update host($(varname))\n")

        elseif buildtype == JAI_DEALLOCATE
            push!(directs, "!\$acc exit data delete($(varname))\n")

        else
            error(string(buildtype) * " is not supported.")
        end
    end

    return join(directs, "\n")

end

function gencode_fortran_openacc(ainfo::AccelInfo, buildtype::BuildType,
                launchid::String,
                inargs::NTuple{N, JaiDataType} where {N},
                innames::NTuple{N, String} where {N}) :: String

    params = fortran_genparams(ainfo)
    typedecls = fortran_typedecls(launchid, buildtype, inargs, innames)
    directives = fortran_openacc_directives(buildtype, inargs, innames)
    arguments = join((innames...,), ",")
    funcname = LIBFUNC_NAME[ainfo.acceltype][buildtype]
 
    return """
module mod$(launchid)
USE, INTRINSIC :: ISO_C_BINDING

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
