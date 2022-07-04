
function fortran_openacc_directives(buildtype::BuildType, inargs::Vector, innames::NTuple)

    directs = []

    for (arg, varname) in zip(inargs, innames)

        # for debug
        #push!(directs, "print *, \"LOC " * string(buildtype) * " of " * varname * " = \", LOC(" * varname * ")\n")

        if buildtype == JAI_ALLOCATE
            push!(directs, "!\$acc enter data create($(varname))\n")

        elseif buildtype == JAI_COPYIN
            push!(directs, "!\$acc enter data copyin($(varname))\n")

        elseif buildtype == JAI_COPYOUT
            push!(directs, "!\$acc exit data copyout($(varname))\n")

        elseif buildtype == JAI_DEALLOCATE
            push!(directs, "!\$acc exit data delete($(varname))\n")

        else
            error(string(buildtype) * " is not supported.")
        end
    end

    return join(directs, "\n")

end

function gencode_fortran_openacc(ainfo::AccelInfo, buildtype::BuildType, launchid::String,
                inargs::Vector, innames::NTuple)

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