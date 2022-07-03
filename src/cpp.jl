c_part1 = """

#include <stdint.h>

"""


function gencode_cpp(kinfo::KernelInfo, hashid::UInt64, kernelbody::String,
                inargs::Vector, outargs::Vector, innames::NTuple, outnames::NTuple)


    #params = genparams(kinfo)
    #funcsig, typedecls = genvars(kinfo, hashid, inargs, outargs,
    #                            innames, outnames)

    #return (kpart1 * params * kpart2 * funcsig * kpart3 * typedecls * kpart4 *
    #        kernelbody * kpart5)

end
