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

    specid::String
    params::Dict
    body::String

    function KernelDef(acceltype::AccelType, kerneldef::String)

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

                bodystr = join(body, "\n")

                for acctype in acctypes
                    if acctype == acceltype
                        push!(sections, KernelSection(acctype, params, bodystr))
                    end
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
                if acctype == acceltype
                    push!(sections, KernelSection(acctype, params, bodystr))
                end
            end
        end

        if length(sections) > 0
            section = sections[end]
            specid = bytes2hex(sha1(string(section.body, section.params))[1:4])

            new(specid, section.params, section.body)
        else
            error("No valid accelerator type is defined in kernel specification file")
        end
    end

end

struct KernelInfo

    kernelid::String
    accel::AccelInfo
    kerneldef::KernelDef

    function KernelInfo(accel::AccelInfo, kerneldef::String)

        kdef = kerneldef

        if isfile(kerneldef)
            open(kerneldef, "r") do io
                kdef = read(io, String)
            end
        end

        kdefobj = KernelDef(accel.acceltype, kdef)
        kernelid = bytes2hex(sha1(string(accel.accelid, kdefobj.specid))[1:4])

        new(kernelid, accel, kdefobj)
    end

    function KernelInfo(accel::AccelInfo, kerneldef::IOStream)
        KernelInfo(accel, read(kerneldef, String))
    end

    function KernelInfo(accel::AccelInfo, kerneldef::KernelDef)
        new(accel, kerneldef, Dict())
    end

end


