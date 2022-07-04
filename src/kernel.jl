
@enum BuildType JAI_LAUNCH JAI_ALLOCATE JAI_DEALLOCATE JAI_COPYIN JAI_COPYOUT

const ACCEL_CODE = Dict(
    JAI_FORTRAN => "FO",
    JAI_FORTRAN_OPENACC => "FA",
    JAI_FORTRAN_OMPTARGET => "FM",
    JAI_CPP => "CP",
    JAI_CPP_OPENACC => "CA",
    JAI_CPP_OMPTARGET => "FM"
)

const BUILD_CODE = Dict(
    JAI_LAUNCH => "L",
    JAI_ALLOCATE => "A",
    JAI_COPYIN => "I",
    JAI_COPYOUT => "O",
    JAI_DEALLOCATE => "D"
)

# accel string to AccelType conversion
const acceltypemap = Dict(
    "fortran" => JAI_FORTRAN,
    "fortran_openacc" => JAI_FORTRAN_OPENACC,
    "cpp"   => JAI_CPP,
    "cpp_openacc" => JAI_CPP_OPENACC
)

struct KernelSection
    acceltype::AccelType
    params::Dict
    body::String
end

const LIBFUNC_NAME = Dict(
    JAI_FORTRAN => Dict(
        JAI_LAUNCH => "jai_launch",
    ),
    JAI_FORTRAN_OPENACC => Dict(
        JAI_LAUNCH => "jai_launch",
        JAI_ALLOCATE => "jai_allocate",
        JAI_COPYIN => "jai_copyin",
        JAI_COPYOUT => "jai_copyout",
        JAI_DEALLOCATE => "jai_deallocate"
    ),
    JAI_CPP => Dict(
        JAI_LAUNCH => "jai_launch",
    ),
    JAI_CPP_OPENACC => Dict(
        JAI_LAUNCH => "jai_launch",
        JAI_ALLOCATE => "jai_allocate",
        JAI_COPYIN => "jai_copyin",
        JAI_COPYOUT => "jai_copyout",
        JAI_DEALLOCATE => "jai_deallocate"
    )
)


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


