c_part1 = """

#include <stdint.h>

"""


function gencode_cpp_kernell(kinfo::KernelInfo, launchid::String, kernelbody::String,
                inargs::Vector, outargs::Vector, innames::NTuple, outnames::NTuple)


    #params = genparams(kinfo)
    #funcsig, typedecls = genvars(kinfo, launchid, inargs, outargs,
    #                            innames, outnames)

    #return (kpart1 * params * kpart2 * funcsig * kpart3 * typedecls * kpart4 *
    #        kernelbody * kpart5)

end
