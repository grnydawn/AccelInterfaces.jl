
const ACCEL_CODE = Dict{AccelType, String}(
    JAI_FORTRAN => "FO",
    JAI_FORTRAN_OPENACC => "FA",
    JAI_FORTRAN_OMPTARGET => "FM",
    JAI_CPP => "CP",
    JAI_CPP_OPENACC => "CA",
    JAI_CPP_OMPTARGET => "FM"
)

const BUILD_CODE = Dict{BuildType, String}(
    JAI_ACCEL => "C",
    JAI_LAUNCH => "L",
    JAI_ALLOCATE => "A",
    JAI_UPDATETO => "I",
    JAI_UPDATEFROM => "O",
    JAI_DEALLOCATE => "D"
)

# accel string to AccelType conversion
const acceltypemap = Dict{String, AccelType}(
    "fortran" => JAI_FORTRAN,
    "fortran_openacc" => JAI_FORTRAN_OPENACC,
    "fortran_omptarget" => JAI_FORTRAN_OMPTARGET,
    "cpp"   => JAI_CPP,
    "cpp_openacc" => JAI_CPP_OPENACC
)

struct KernelSection
    acceltype::AccelType
    params::Dict{String, String}
    body::String
    secenv::Module

    function KernelSection(acceltype::AccelType, accelid::String,
                    params::Dict{String, String}, body::String, secnum::Int)

        modsym = Symbol("ksmod_$(accelid)_$(length(_kernelcache))_$(secnum)")

        if acceltype == JAI_HEADER

            bodyexpr = Meta.parse(body) 

            mod = @eval module $modsym
                        $bodyexpr
                        end
        else
            mod = @eval module $modsym
                        end
        end

        new(acceltype, params, body, mod)
    end
end

const LIBFUNC_NAME = Dict{AccelType, Dict}(
    JAI_FORTRAN => Dict{BuildType, String}(
        JAI_LAUNCH => "jai_launch",
    ),
    JAI_FORTRAN_OPENACC => Dict{BuildType, String}(
        JAI_LAUNCH => "jai_launch",
        JAI_ALLOCATE => "jai_allocate",
        JAI_UPDATETO => "jai_updateto",
        JAI_UPDATEFROM => "jai_updatefrom",
        JAI_DEALLOCATE => "jai_deallocate"
    ),
    JAI_FORTRAN_OMPTARGET => Dict{BuildType, String}(
        JAI_LAUNCH => "jai_launch",
        JAI_ALLOCATE => "jai_allocate",
        JAI_UPDATETO => "jai_updateto",
        JAI_UPDATEFROM => "jai_updatefrom",
        JAI_DEALLOCATE => "jai_deallocate"
    ),

    JAI_CPP => Dict{BuildType, String}(
        JAI_LAUNCH => "jai_launch",
    ),
    JAI_CPP_OPENACC => Dict{BuildType, String}(
        JAI_LAUNCH => "jai_launch",
        JAI_ALLOCATE => "jai_allocate",
        JAI_UPDATETO => "jai_updateto",
        JAI_UPDATEFROM => "jai_updatefrom",
        JAI_DEALLOCATE => "jai_deallocate"
    )
)


function _parse_header(hdr) :: Tuple{Vector{AccelType}, Dict{String, String}}

    acctypes = Vector{AccelType}()
    params = Dict{String, String}()

    local pos = findfirst(isequal(':'), hdr)
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

    acctypes, params
end

# keep the only matching section
struct KernelDef

    specid::String
    params::Dict
    body::String

    function KernelDef(accel::AccelInfo, kerneldef::String)

        sections = KernelSection[]
        acceltype = accel.acceltype
        accelid = accel.accelid

        acctypes = Vector{AccelType}()
        push!(acctypes, JAI_HEADER)
        #acctypes = AccelType[JAI_HEADER]
        params = Dict{String, String}()
        body = String[]

        for line in split(kerneldef, "\n")

            s = rstrip(line)

            # comments start with # or ;
            if length(s) < 1 || s[1] == '#'
                continue

            elseif s[1] == '[' && s[end] == ']'

                bodystr = join(body, "\n")

                for atype in acctypes
                    if atype == acceltype || atype == JAI_HEADER
                        push!(sections, KernelSection(atype, accelid, params, bodystr, length(sections)))
                    end
                end
                    
                acctypes, params = _parse_header(s[2:end-1])
                body = String[]

            else
                push!(body, s)
            end
        end

        # activate the only matching section
        if length(acctypes) > 0
            bodystr = join(body, "\n")
            for atype in acctypes
                if atype == acceltype || atype == JAI_HEADER
                    push!(sections, KernelSection(atype, accelid, params, bodystr, length(sections)))
                end
            end
        end

        if length(sections) > 0
            section = sections[end]
            #specid = bytes2hex(sha1(string(section.body, section.params))[1:4])

            io = IOBuffer()
            ser = serialize(io, (section.body, section.params))
            specid = bytes2hex(sha1(String(take!(io)))[1:4])

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

    function KernelInfo(accel::AccelInfo, kerneldef::KernelDef;
            _lineno_::Union{Int64, Nothing}=nothing,
            _filepath_::Union{String, Nothing}=nothing)

        io = IOBuffer()
        ser = serialize(io, (accel.accelid, kerneldef.specid, _lineno_, _filepath_))
        kernelid = bytes2hex(sha1(String(take!(io)))[1:4])

        new(kernelid, accel, kerneldef)
    end

end

const _kernelcache = Dict{String, KernelInfo}()

function jai_kernel_init(kname::String, aname::String,
            kspec::Union{String, KernelDef};
            _lineno_::Union{Int64, Nothing}=nothing,
            _filepath_::Union{String, Nothing}=nothing)

    accel = _accelcache[aname]

    if kspec isa String

        kdef = kspec

        if isfile(kspec)
            open(kspec, "r") do io
                kdef = read(io, String)
            end
        end
    end

    kernel = KernelInfo(accel, KernelDef(accel, kdef),
                    _lineno_=_lineno_, _filepath_=_filepath_)

    global _kernelcache[kname] = kernel
end

