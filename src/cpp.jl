c_part1 = """

#include <stdint.h>

"""


function gencode_cpp_kernel(kinfo::KernelInfo, launchid::String,
                kernelbody::String,
                inargs::NTuple{N, JaiDataType} where {N},
                outargs::NTuple{M, JaiDataType} where {M},
                innames::NTuple{N, String} where {N},
                outnames::NTuple{M, String} where {M}) :: String

#    params = fortran_genparams(kinfo)
#    arguments, typedecls = fortran_genvars(kinfo, launchid, inargs, outargs,
#                                innames, outnames)
#
#    return code = """
##include <stdint.h>
#
#module mod$(launchid)
#USE, INTRINSIC :: ISO_C_BINDING
#
#$(params)
#
#public jai_launch
#
#contains
#
#INTEGER (C_INT64_T) FUNCTION jai_launch($(arguments)) BIND(C, name="jai_launch")
#USE, INTRINSIC :: ISO_C_BINDING
#
#$(typedecls)
#
#INTEGER (C_INT64_T) :: JAI_ERRORCODE  = 0
#
#$(kernelbody)
#
#jai_launch = JAI_ERRORCODE
#
#END FUNCTION
#
#end module
#
#"""
end

