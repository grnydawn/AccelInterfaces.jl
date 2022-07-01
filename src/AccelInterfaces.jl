module AccelInterfaces

# Write your package code here.
export AccelType, FLANG, CLANG, KernelInfo

@enum AccelType FLANG CLANG ANYACCEL

struct KernelInfo
end

end
