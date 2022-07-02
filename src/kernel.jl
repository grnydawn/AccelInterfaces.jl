
# accel string to AccelType conversion
const acceltypemap = Dict(
    "fortran" => JAI_FORTRAN,
    "cpp"   => JAI_CPP
)

struct KernelSection
    acceltype::AccelType
    params::Dict
    body::String
end

function _parse_header(hdr)

    acctypes = Vector{AccelType}()
    params = Dict()

    pos = findfirst(isequal(':'), hdr)
    if pos == nothing
        hdrtype = hdr
        hdrparam = ""
    else
        hdrtype = hdr[1:pos-1]
        hdrparam = hdr[pos+1:end]
    end

    for acctype in split(hdrtype, ",")
        push!(acctypes, acceltypemap[lowercase(strip(acctype))])
    end
    
    # TODO: eval params in a custom environment
    #@getparams params $hdrparam

    return (acctypes, params)
end

struct KernelDef

    rawdata::String
    sections::Vector

    function KernelDef(kerneldef::String)

        sections = []

        acctypes = [JAI_HEADER]
        params = Dict()
        body = []

        for line in split(kerneldef, "\n")

            s = rstrip(line)

            # comments start with # or ;
            if length(s) < 1 || s[1] == '#'
                continue

            elseif s[1] == '[' && s[end] == ']'

                bodystr = join(body)

                for acctype in acctypes
                    push!(sections, KernelSection(acctype, params, bodystr))
                end
                    
                acctypes, params = _parse_header(s[2:end-1])
                body = []

            else
                push!(body, s)
            end
        end

        if length(acctypes) > 0
            bodystr = join(body, "\n")
            for acctype in acctypes
                push!(sections, KernelSection(acctype, params, bodystr))
            end
        end

       new(kerneldef, sections)
    end

end

function get_kernelbody(kdef::KernelDef, acceltype::AccelType)

    for section in kdef.sections
        if section.acceltype == acceltype
            return section.body
        end
    end
end

struct KernelInfo

    accel::AccelInfo
    kerneldef::KernelDef
    sharedlibs::Dict

    function KernelInfo(accel::AccelInfo, kerneldef::String)

        kdef = kerneldef

        if isfile(kerneldef)
            open(kerneldef, "r") do io
                kdef = read(io, String)
            end
        end

        new(accel, KernelDef(kdef), Dict())
    end

    function KernelInfo(accel::AccelInfo, kerneldef::IOStream)
        KernelInfo(accel, read(kerneldef, String))
    end

    function KernelInfo(accel::AccelInfo, kerneldef::KernelDef)
        new(accel, kerneldef, Dict())
    end

end


