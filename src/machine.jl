# machine.jl: implement functions for specific OS and machine support

struct JAI_TYPE_OS
    name    ::String
end

struct JAI_TYPE_MACHINE
    name    ::String
    prerun  ::String
end

