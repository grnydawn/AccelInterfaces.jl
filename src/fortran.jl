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
    INTEGER (C_INT64_T) :: res

    res = 0

"""

const kpart5 = """
    launch = res

END FUNCTION

end module
"""

function typeconv(value)

    return "VALUE"
end

function genparams(constants::Dict)

    params = []
    for (name, value) in kinfo.accel.constants
        push!(params, "$(typeconv(value)), PARAMETER :: $name = $value")
    end

    return join(params, "\n")
end

function genvars(kinfo::KernelInfo, hashid::UInt64, inargs::Vector, indtypes::Vector,
                inoutargs::Vector, inoutdtypes::Vector, outargs::Vector, outdtypes::Vector,
                innames::NTuple, outnames::NTuple)

#INTEGER (C_INT64_T) FUNCTION launch($arguments) BIND(C, name="launch")

    return "", ""
end

function gencode_fortran(kinfo::KernelInfo, hashid::UInt64, inargs::Vector, indtypes::Vector,
                inoutargs::Vector, inoutdtypes::Vector, outargs::Vector, outdtypes::Vector,
                innames::NTuple, outnames::NTuple)

    open(kinfo.kernelpath, "r") do io
           kernelbody = read(io)
    end

    params = genparams(kinfo.accel.constants)
    args, typedecls = getvars(kinfo, hashid, inargs, indtypes, inoutargs, inoutdtypes,
                                outargs, outdtypes, innames, outnames)

    return (kpart1 * params * kpart2 * args * kpart3 * typedecls * kpart4 *
            kernelbody * kpart5)

end
