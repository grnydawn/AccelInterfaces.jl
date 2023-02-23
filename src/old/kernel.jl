
const ACCEL_CODE = Dict{AccelType, String}(
    JAI_FORTRAN => "F_",
    JAI_FORTRAN_OPENACC => "FA",
    JAI_FORTRAN_OMPTARGET => "FM",
    JAI_CPP => "C_",
    JAI_CPP_CUDA => "CU",
    JAI_CPP_HIP => "HI",
    JAI_CPP_OPENACC => "CA",
    JAI_CPP_OMPTARGET => "CM"
)

const BUILD_CODE = Dict{BuildType, String}(
    JAI_ACCEL => "I",
    JAI_LAUNCH => "K",
    JAI_ALLOCATE => "A",
    JAI_UPDATETO => "T",
    JAI_UPDATEFROM => "F",
    JAI_DEALLOCATE => "D"
)

# accel string to AccelType conversion
const acceltypemap = Dict{String, AccelType}(
    "fortran" => JAI_FORTRAN,
    "fortran_openacc" => JAI_FORTRAN_OPENACC,
    "fortran_omptarget" => JAI_FORTRAN_OMPTARGET,
    "cpp"   => JAI_CPP,
    "cuda"   => JAI_CPP_CUDA,
    "hip"   => JAI_CPP_HIP,
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
        JAI_DEALLOCATE => "jai_deallocate",
        JAI_WAIT => "jai_wait"
    ),
    JAI_FORTRAN_OMPTARGET => Dict{BuildType, String}(
        JAI_LAUNCH => "jai_launch",
        JAI_ALLOCATE => "jai_allocate",
        JAI_UPDATETO => "jai_updateto",
        JAI_UPDATEFROM => "jai_updatefrom",
        JAI_DEALLOCATE => "jai_deallocate",
        JAI_WAIT => "jai_wait"
    ),

    JAI_CPP => Dict{BuildType, String}(
        JAI_LAUNCH => "jai_launch",
    ),
    JAI_CPP_CUDA => Dict{BuildType, String}(
        JAI_LAUNCH => "jai_launch",
        JAI_ALLOCATE => "jai_allocate",
        JAI_UPDATETO => "jai_updateto",
        JAI_UPDATEFROM => "jai_updatefrom",
        JAI_DEALLOCATE => "jai_deallocate",
        JAI_WAIT => "jai_wait"
    ),
    JAI_CPP_HIP => Dict{BuildType, String}(
        JAI_LAUNCH => "jai_launch",
        JAI_ALLOCATE => "jai_allocate",
        JAI_UPDATETO => "jai_updateto",
        JAI_UPDATEFROM => "jai_updatefrom",
        JAI_DEALLOCATE => "jai_deallocate",
        JAI_WAIT => "jai_wait"
    ),
    JAI_CPP_OPENACC => Dict{BuildType, String}(
        JAI_LAUNCH => "jai_launch",
        JAI_ALLOCATE => "jai_allocate",
        JAI_UPDATETO => "jai_updateto",
        JAI_UPDATEFROM => "jai_updatefrom",
        JAI_DEALLOCATE => "jai_deallocate",
        JAI_WAIT => "jai_wait"
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

    function KernelDef(accel::AccelInfo, acceltype::AccelType, kerneldef::String)

        sections = KernelSection[]
        #acceltype = accel.acceltype
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
    acceltype::AccelType
    compile::String

    function KernelInfo(accel::AccelInfo, kerneldef::KernelDef,
            acceltype::AccelType, compile::String;
            _lineno_::Union{Int64, Nothing}=nothing,
            _filepath_::Union{String, Nothing}=nothing)

        io = IOBuffer()
        ser = serialize(io, (accel.accelid, kerneldef.specid, _lineno_, _filepath_))
        kernelid = bytes2hex(sha1(String(take!(io)))[1:4])

        new(kernelid, accel, kerneldef, acceltype, compile)
    end

end

const _kernelcache = Dict{String, KernelInfo}()

function jai_kernel_init(
            aname::String,
            kname::String,
            kspec::Union{String, KernelDef};
            framework::Union{NTuple{N, Tuple{String, Union{NTuple{M, Tuple{String,
                        Union{String, Nothing}}}, String, Nothing}}}, Nothing} where {N, M}=nothing,
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

    acceltype = accel.acceltype
    compile = accel.compile

    # TODO: add more framework that should be available as a kernel

    if framework != nothing
        for (frameworkname, frameconfig) in framework
            acceltype = _accelmap[frameworkname]

            if frameconfig isa Nothing
                if startswith(frameworkname, "fortran")
                    compile = get(ENV, "JAI_FC", get(ENV, "FC", "")) * " " *
                                get(ENV, "JAI_FFLAGS", get(ENV, "FFLAGS", ""))

                elseif startswith(frameworkname, "cpp")
                    compile = get(ENV, "JAI_CXX", get(ENV, "CXX", "")) * " " *
                                get(ENV, "JAI_CXXFLAGS", get(ENV, "CXXFLAGS", ""))
                else
                    error(string(frameworkname * " is not supported."))
                end

            elseif frameconfig isa String
                compile = frameconfig

            else
                for (cfgname, cfg) in frameconfig
                    if cfgname == "compile"
                        compile = cfg
                    end
                end
            end

            if compile == nothing
                error("No compile information is available.")
            end

        end
    end

    kernel = KernelInfo(accel, KernelDef(accel, acceltype, kdef),
                    acceltype, compile, _lineno_=_lineno_, _filepath_=_filepath_)

    global _kernelcache[aname * kname] = kernel

end

function merge_args(inargs::NTuple{N, JaiDataType} where {N},
                outargs::NTuple{M, JaiDataType} where {M},
                innames::NTuple{N, String} where {N},
                outnames::NTuple{M, String} where {M}) :: Tuple{
                    NTuple{M, JaiDataType} where {M},
                    NTuple{N, String} where {N}
                } Tuple{NTuple{M, JaiDataType}, NTuple{M, String}} where {M}

    #args = collect(JaiDataType, inargs)
    #names = collect(String, innames)
    N1 = length(inargs)
    N = N1 + length(outargs)
    args = Vector{JaiDataType}(undef, N)
    names = Vector{String}(undef, N)

    for i in range(1, stop=N1)
        args[i] = inargs[i]
        names[i] = innames[i]
    end

    ptr = N1

    for (index, oname) in enumerate(outnames)
        if !(oname in innames)
            #push!(args, outargs[index])
            #push!(names, oname)
            ptr += 1
            args[ptr] = outargs[index]
            names[ptr] = oname

        end
    end

    #return (args...,), (names...,)
    return (args[1:ptr]...,), (names[1:ptr]...,)

end

