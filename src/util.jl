# util.jl: implement utility functions
#
#
#

extract_name_from_frametype(x) = lowercase(split(string(x), ".")[end][10:end])
