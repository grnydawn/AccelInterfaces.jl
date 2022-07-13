module AccelInterfaces

using Serialization

import Libdl.dlopen,
       Libdl.RTLD_LAZY,
       Libdl.RTLD_DEEPBIND,
       Libdl.RTLD_GLOBAL,
       Libdl.dlsym

import SHA.sha1
import Dates.now,
       Dates.Millisecond

import OffsetArrays.OffsetArray,
       OffsetArrays.OffsetVector

        #JAI_ANYACCEL, JAI_CPP_OPENACC, AccelInfo, KernelInfo, get_accel, get_kernel,
export AccelType, JAI_VERSION, JAI_FORTRAN, JAI_CPP, JAI_FORTRAN_OPENACC,
        jai_directive, jai_accel_init, jai_accel_fini, jai_kernel_init,
        @jenterdata, @jexitdata, @jlaunch, @jaccel, @jkernel, @jdecel


const JAI_VERSION = "0.0.1"
const TIMEOUT = 10

#@enum DeviceType JAI_HOST JAI_DEVICE

@enum BuildType::Int64 begin
    JAI_ALLOCATE    = 10
    JAI_UPDATETO    = 20
    JAI_LAUNCH      = 30
    JAI_UPDATEFROM  = 40
    JAI_DEALLOCATE  = 50
end

@enum AccelType begin
        JAI_FORTRAN
        JAI_FORTRAN_OPENACC
        JAI_FORTRAN_OMPTARGET
        JAI_CPP
        JAI_CPP_OPENACC
        JAI_CPP_OMPTARGET
        JAI_ANYACCEL
        JAI_HEADER
end

const _accelmap = Dict{String, AccelType}(
    "fortran" => JAI_FORTRAN,
    "fortran_openacc" => JAI_FORTRAN_OPENACC,
    "fortran_omptarget" => JAI_FORTRAN_OMPTARGET,
    "cpp" => JAI_CPP,
    "cpp_openacc" => JAI_CPP_OPENACC,
    "cpp_omptarget" => JAI_CPP_OMPTARGET,
    "any" => JAI_ANYACCEL
)

const JaiConstType = Union{Number, NTuple{N, T}, AbstractArray{T, N}} where {N, T<:Number}
const JaiDataType = JaiConstType

struct AccelInfo

    accelid::String
    acceltype::AccelType
    ismaster::Bool
    device_type::Int64
    device_num::Int64
    const_vars::NTuple{N,JaiConstType} where {N}
    const_names::NTuple{N, String} where {N}
    compile::Union{String, Nothing}
    sharedlibs::Dict{String, Ptr{Nothing}}
    workdir::Union{String, Nothing}
    debugdir::Union{String, Nothing}
    dtypecache::Dict{T, String} where T<:DataType
    directcache::Dict{Tuple{BuildType, Int64, Int64, String}, Tuple{Ptr{Nothing}, Expr}}

    function AccelInfo(;master::Bool=true,
            const_vars::NTuple{N,JaiConstType} where {N}=(),
            const_names::NTuple{N, String} where {N}=(),
            framework::NTuple{N, String} where {N}=(),
            device::NTuple{N, Integer} where {N}=(),
            compile::NTuple{N, String} where {N}=(),
            workdir::Union{String, Nothing}=nothing,
            debugdir::Union{String, Nothing}=nothing,
            _lineno_::Union{Int64, Nothing}=nothing,
            _filepath_::Union{String, Nothing}=nothing)

        # TODO: check if acceltype is supported in this system(h/w, compiler, ...)
        #     : detect available acceltypes according to h/w, compiler, flags, ...

        io = IOBuffer()
        ser = serialize(io, (Sys.STDLIB, JAI_VERSION, framework, const_vars,
                        const_names, compile, _lineno_, _filepath_))
        accelid = bytes2hex(sha1(String(take!(io)))[1:4])

        if workdir == nothing
            workdir = joinpath(pwd(), ".jaitmp")
        end

        if master && !isdir(workdir)
            mkdir(workdir)
        end

        if debugdir != nothing
            debugdir = abspath(debugdir)
            if master && !isdir(debugdir)
                mkdir(debugdir)
            end
        end

        # TODO: support multiple framework arguments
        acceltype = _accelmap[lowercase(framework[1])]

        if length(device) == 0
            device_type = -1
            device_num = -1

        elseif length(device) == 1
            device_type = -1
            device_num = device[1]

        else
            device_type = device[1]
            device_num = device[2]

        end

        # TODO: support multiple optional compiler commands
        compile = length(compile) == 0 ? nothing : compile[1]

        new(accelid, acceltype, master, device_type, device_num, const_vars,
            const_names, compile, Dict{String, Ptr{Nothing}}(),
            workdir, debugdir, Dict{DataType, String}(),
            Dict{Tuple{BuildType, Int64, Int64, String},
            Tuple{Ptr{Nothing}, Expr}}())
    end
end

const _accelcache = Dict{String, AccelInfo}()

function jai_accel_init(name::String; master::Bool=true,
            const_vars::NTuple{N,JaiConstType} where {N}=(),
            const_names::NTuple{N, String} where {N}=(),
            framework::NTuple{N, String} where {N}=(),
            device::NTuple{N, Integer} where {N}=(),
            compile::NTuple{N, String} where {N}=(),
            workdir::Union{String, Nothing}=nothing,
            debugdir::Union{String, Nothing}=nothing,
            _lineno_::Union{Int64, Nothing}=nothing,
            _filepath_::Union{String, Nothing}=nothing)

    accel = AccelInfo(master=master, const_vars=const_vars, const_names=const_names, 
                    framework=framework, device=device, compile=compile,
                    workdir=workdir, debugdir=debugdir, _lineno_=_lineno_,
                    _filepath_=_filepath_)

    global _accelcache[name] = accel
end

function jai_accel_fini(name::String;
            _lineno_::Union{Int64, Nothing}=nothing,
            _filepath_::Union{String, Nothing}=nothing)

    delete!(_accelcache, name)
end

# NOTE: keep the order of includes
include("./kernel.jl")
include("./fortran.jl")
include("./fortran_openacc.jl")
include("./cpp.jl")

function timeout(libpath::String, duration::Real; waittoexist::Bool=true) :: Nothing

    local tstart = now()

    while true
        local check = waittoexist ? ispath(libpath) : ~ispath(libpath)

        if check
            break

        elseif ((now() - tstart)/ Millisecond(1000)) > duration
            error("Timeout occured while waiting for shared library")

        else
            sleep(0.1)
        end
    end
end

function get_accel(acceltype::AccelType; ismaster::Bool=true,
                    const_vars::NTuple{N,JaiConstType} where {N}=(),
                    compile::Union{String, Nothing}=nothing,
                    const_names::NTuple{N, String} where {N}=()) :: AccelInfo

    return AccelInfo(acceltype, ismaster=ismaster, const_vars=const_vars,
                    compile=compile, const_names=const_names)
end

function get_kernel(accel::AccelInfo, path::String) :: KernelInfo
    return KernelInfo(accel, path)
end

function jai_directive(accel::String, buildtype::BuildType,
            buildtypecount::Int64,
            data::Vararg{JaiDataType, N} where {N};
            names::NTuple{N, String} where {N}=(),
            _lineno_::Union{Int64, Nothing}=nothing,
            _filepath_::Union{String, Nothing}=nothing) :: Int64

    accel = _accelcache[accel]

    data = (accel.device_num, data...)
    names = ("jai_arg_device_num", names...)

    if accel.acceltype in (JAI_FORTRAN, JAI_CPP)
        return 0::Int64
    end

    cachekey = (buildtype, buildtypecount, _lineno_, _filepath_)

    if _lineno_ isa Int64 && _filepath_ isa String
        if haskey(accel.directcache, cachekey)
            dfunc, argtypes = accel.directcache[cachekey]
            ccallexpr = :(ccall($dfunc, Int64, $argtypes, $(data...)))
            @eval return $ccallexpr
        end
    end

    dtypes, sizes = argsdtypes(accel, data...)

    io = IOBuffer()
    ser = serialize(io, (buildtype, accel.accelid, dtypes, sizes))
    launchid = bytes2hex(sha1(String(take!(io)))[1:4])

    local libpath = joinpath(accel.workdir, "SL$(launchid).so")

    # load shared lib
    if haskey(accel.sharedlibs, launchid)
        dlib = accel.sharedlibs[launchid]

    else
        build!(accel, buildtype, launchid, libpath, data, names)
        dlib = dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        accel.sharedlibs[launchid] = dlib
    end

    if buildtype == JAI_ALLOCATE
        local dfunc = dlsym(dlib, :jai_allocate)

    elseif buildtype == JAI_UPDATETO
        local dfunc = dlsym(dlib, :jai_updateto)

    elseif buildtype == JAI_UPDATEFROM
        local dfunc = dlsym(dlib, :jai_updatefrom)

    elseif buildtype == JAI_DEALLOCATE
        local dfunc = dlsym(dlib, :jai_deallocate)

    else
        error(string(buildtype) * " is not supported.")

    end

    local argtypes = Meta.parse("("*join(dtypes, ",")*",)")
    local ccallexpr = :(ccall($dfunc, Int64, $argtypes, $(data...)))

    #println("************************")
    #dtypestr = "("*join(dtypes, ",")*",)"
    #println(dtypestr)
    #println(names)

    if _lineno_ isa Int64 && _filepath_ isa String
        accel.directcache[cachekey] = (dfunc, argtypes)
    end

    @eval return $ccallexpr

end

function argsdtypes(ainfo::AccelInfo,
            data::Vararg{JaiDataType, N} where {N};
        ) :: Tuple{Vector{String}, Vector{NTuple{M, T} where {M, T<:Integer}}}
        #) :: Tuple{Vector{DataType}, Vector{NTuple{M, T} where {M, T<:Integer}}}

    local N = length(data)

    dtypes = Vector{String}(undef, N)
    sizes = Vector{NTuple{M, T} where {M, T<:Integer}}(undef, N)

    for (index, arg) in enumerate(data)
        local arg = data[index]

        sizes[index] = size(arg)

        if typeof(arg) <: AbstractArray
            dtype = Ptr{typeof(arg)}

        elseif ainfo.acceltype in (JAI_CPP, JAI_CPP_OPENACC)
            dtype = typeof(arg)

        elseif ainfo.acceltype in (JAI_FORTRAN, JAI_FORTRAN_OPENACC)
            dtype = Ref{typeof(arg)}
        end

        if haskey(ainfo.dtypecache, dtype)
            dtypes[index] = ainfo.dtypecache[dtype]

        else
            dtypes[index] = string(dtype)
            ainfo.dtypecache[dtype] = dtypes[index]
        end
    end

    dtypes, sizes
end

# kernel launch
function launch_kernel(kname::String,
            invars::Vararg{JaiDataType, N} where {N};
            innames::NTuple{N, String} where {N}=(),
            outnames::NTuple{N, String} where {N}=(),
            output::NTuple{N,JaiDataType} where {N}=(),
            _lineno_::Union{Int64, Nothing}=nothing,
            _filepath_::Union{String, Nothing}=nothing) :: Int64

    kinfo = _kernelcache[kname]

    invars = (kinfo.accel.device_num, invars...)
    innames = ("jai_arg_device_num", innames...)

    args = (invars..., output...)
    cachekey = (_lineno_, _filepath_)

    if _lineno_ isa Int64 && _filepath_ isa String
        if haskey(kinfo.launchcache, cachekey)
            kfunc, argtypes = kinfo.launchcache[cachekey]
            ccallexpr = :(ccall($kfunc, Int64, $argtypes, $(args...)))
            @eval return $ccallexpr
        end
    end

    indtypes, insizes = argsdtypes(kinfo.accel, invars...)
    outdtypes, outsizes = argsdtypes(kinfo.accel, output...)
    dtypes = vcat(indtypes, outdtypes)

    io = IOBuffer()
    ser = serialize(io, (JAI_LAUNCH, kinfo.kernelid, indtypes, insizes, outdtypes, outsizes))
    launchid = bytes2hex(sha1(String(take!(io)))[1:4])

    libpath = joinpath(kinfo.accel.workdir, "SL$(launchid).so")

    # load shared lib
    if haskey(kinfo.accel.sharedlibs, launchid)
        dlib = kinfo.accel.sharedlibs[launchid]

    else
        build!(kinfo, launchid, libpath, invars, output, innames, outnames)
        dlib = dlopen(libpath, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        kinfo.accel.sharedlibs[launchid] = dlib
    end

    kfunc = dlsym(dlib, :jai_launch)

    #println("#####################")
    #dtypestr = "("*join(dtypes, ",")*",)"
    #println(dtypestr)
    #println((innames..., outnames...))

    local argtypes = Meta.parse(dtypestr)
    local ccallexpr = :(ccall($kfunc, Int64, $argtypes, $(args...)))

    if _lineno_ isa Int64 && _filepath_ isa String
        kinfo.launchcache[cachekey] = (kfunc, argtypes)
    end

    @eval return $ccallexpr
end

function setup_build(acceltype::AccelType, buildtype::BuildType, launchid::String,
                compile::Union{String, Nothing})

    prefix = ACCEL_CODE[acceltype] * BUILD_CODE[buildtype]
 
    if acceltype == JAI_FORTRAN
        ext = ".F90"
        if compile == nothing
            compile = "gfortran -fPIC -shared -g -ffree-line-length-none"
        end

    elseif  acceltype == JAI_CPP
        ext = ".cpp"
        if compile == nothing
            compile = "g++ -fPIC -shared -g"
        end

    elseif  acceltype == JAI_FORTRAN_OPENACC
        ext = ".F90"
        if compile == nothing
            compile = "gfortran -fPIC -shared -fopenacc -g -ffree-line-length-none"
        end

    else
        error(string(acceltype) * " is not supported yet.")

    end

    (prefix*launchid*ext, compile)
end


# kernel build
function build!(kinfo::KernelInfo, launchid::String, outpath::String,
                inargs::NTuple{N, JaiDataType} where {N},
                outargs::NTuple{M, JaiDataType} where {M},
                innames::NTuple{N, String} where {N},
                outnames::NTuple{M, String} where {M}) :: String

    compile = kinfo.accel.compile

    srcfile, compile = setup_build(kinfo.accel.acceltype, JAI_LAUNCH, launchid,
                                    compile)

    srcpath = joinpath(kinfo.accel.workdir, srcfile)
    pidfile = outpath * ".pid"

    # generate source code
    if !ispath(outpath)

        code = generate!(kinfo, launchid, inargs, outargs, innames, outnames)

        if !ispath(outpath)

            curdir = pwd()

            try
                procdir = mktempdir()
                cd(procdir)

                open(srcfile, "w") do io
                    write(io, code)
                end

                if kinfo.accel.debugdir != nothing
                    debugfile = joinpath(kinfo.accel.debugdir, srcfile)
                    open(debugfile, "w") do io
                        write(io, code)
                    end
                end

                outfile = basename(outpath)

                if !ispath(outpath)
                    compilelog = read(run(`$(split(compile)) -o $outfile $(srcfile)`), String)

                    if !ispath(outpath)
                        open(pidfile, "w") do io
                            write(io, string(getpid()))
                        end

                        if !ispath(outpath)
                            cp(outfile, outpath)
                        end

                        rm(pidfile)
                    end
                end
            catch err

            finally
                cd(curdir)
            end
        end
    end

    timeout(pidfile, TIMEOUT, waittoexist=false)

    outpath
end


# non-kernel build
function build!(ainfo::AccelInfo, buildtype::BuildType, launchid::String,
                outpath::String,
                args::NTuple{N, JaiDataType} where {N},
                names::NTuple{N, String} where {N}) :: String

    srcfile, compile = setup_build(ainfo.acceltype, buildtype,
                launchid, ainfo.compile)

    srcpath = joinpath(ainfo.workdir, srcfile)
    pidfile = outpath * ".pid"

    # generate source code
    if !ispath(outpath)
        code = generate!(ainfo, buildtype, launchid, args, names)

        if !ispath(outpath)

            curdir = pwd()

            try
                procdir = mktempdir()
                cd(procdir)

                open(srcfile, "w") do io
                    write(io, code)
                end

                if ainfo.debugdir != nothing
                    debugfile = joinpath(ainfo.debugdir, srcfile)
                    open(debugfile, "w") do io
                        write(io, code)
                    end
                end

                outfile = basename(outpath)

                if !ispath(outpath)
                    compilelog = read(run(`$(split(compile)) -o $outfile $(srcfile)`), String)

                    if !ispath(outpath)
                        open(pidfile, "w") do io
                            write(io, string(getpid()))
                        end

                        if !ispath(outpath)
                            cp(outfile, outpath)
                        end

                        rm(pidfile)
                    end
                end
            catch err

            finally
                cd(curdir)
            end
        end
    end

    timeout(pidfile, TIMEOUT, waittoexist=false)

    outpath
end

# kernel generate
function generate!(kinfo::KernelInfo, launchid::String,
                inargs::NTuple{N, JaiDataType} where {N},
                outargs::NTuple{M, JaiDataType} where {M},
                innames::NTuple{N, String} where {N},
                outnames::NTuple{M, String} where {M}) :: String

    body = kinfo.kerneldef.body

    if kinfo.accel.acceltype == JAI_FORTRAN
        code = gencode_fortran_kernel(kinfo, launchid, body, inargs, outargs, innames, outnames)

    elseif kinfo.accel.acceltype == JAI_CPP
        code = gencode_cpp_kernel(kinfo, launchid, body, inargs, outargs, innames, outnames)

    elseif kinfo.accel.acceltype == JAI_FORTRAN_OPENACC
        code = gencode_fortran_kernel(kinfo, launchid, body, inargs, outargs, innames, outnames)

    end

    code
end

# accel generate
function generate!(ainfo::AccelInfo, buildtype::BuildType,
                launchid::String,
                args::NTuple{N, JaiDataType} where {N},
                names::NTuple{N, String} where {N}) :: String


    if ainfo.acceltype == JAI_FORTRAN_OPENACC
        code = gencode_fortran_openacc(ainfo, buildtype, launchid, args, names)

    else
        error(string(ainfo.acceltype) * " is not supported for allocation.")

    end

    code
end


macro jenterdata(accel, directs...)

    tmp = Expr(:block)
    allocs = Expr[]
    nonallocs = Expr[]
    alloccount = 1
    updatetocount = 1
    allocnames = String[]
    updatenames = String[]

    for direct in directs
        insert!(direct.args, 2, string(accel))

        if direct.args[1] == :allocate
            for uvar in direct.args[3:end]
                push!(allocnames, String(uvar))
            end
            insert!(direct.args, 3, JAI_ALLOCATE)
            insert!(direct.args, 4, alloccount)
            alloccount += 1
            push!(allocs, direct)

        elseif direct.args[1] == :update
            for dvar in direct.args[3:end]
                push!(updatenames, String(dvar))
            end
            insert!(direct.args, 3, JAI_UPDATETO)
            insert!(direct.args, 4, updatetocount)
            updatetocount += 1
            push!(nonallocs, direct)

        else
            error(string(direct.args[1]) * " is not supported.")

        end

    end

    for direct in (allocs..., nonallocs...)

        if direct.args[1] == :update
            kwupdatenames = Expr(:kw, :names, Expr(:tuple, updatenames...))
            push!(direct.args, kwupdatenames)

        elseif direct.args[1] == :allocate

            kwallocnames = Expr(:kw, :names, Expr(:tuple, allocnames...))
            push!(direct.args, kwallocnames)

        end

        kwline = Expr(:kw, :_lineno_, __source__.line)
        push!(direct.args, kwline)

        kwfile = Expr(:kw, :_filepath_, string(__source__.file))
        push!(direct.args, kwfile)

        direct.args[1] = :jai_directive

        push!(tmp.args, esc(direct))
    end

    #dump(tmp)
    return(tmp)
end

macro jexitdata(accel, directs...)


    tmp = Expr(:block)
    deallocs = Expr[]
    nondeallocs = Expr[]
    updatefromcount = 1
    dealloccount = 1
    deallocnames = String[]
    updatenames = String[]

    for direct in directs

        insert!(direct.args, 2, string(accel))

        if direct.args[1] == :update
            for uvar in direct.args[3:end]
                push!(updatenames, String(uvar))
            end
            insert!(direct.args, 3, JAI_UPDATEFROM)
            insert!(direct.args, 4, updatefromcount)
            updatefromcount += 1
            push!(nondeallocs, direct)

        elseif direct.args[1] == :deallocate
            for dvar in direct.args[3:end]
                push!(deallocnames, String(dvar))
            end
            insert!(direct.args, 3, JAI_DEALLOCATE)
            insert!(direct.args, 4, dealloccount)
            dealloccount += 1
            push!(deallocs, direct)

        else
            error(string(direct.args[1]) * " is not supported.")

        end
    end

    for direct in (nondeallocs..., deallocs...)

        if direct.args[1] == :update
            kwupdatenames = Expr(:kw, :names, Expr(:tuple, updatenames...))
            #kwupdatenames = Expr(:kw, :names, :($((updatenames...),)))
            push!(direct.args, kwupdatenames)

        elseif direct.args[1] == :deallocate

            kwdeallocnames = Expr(:kw, :names, Expr(:tuple, deallocnames...))
            #kwdeallocnames = Expr(:kw, :names, :($((deallocnames...),)))
            push!(direct.args, kwdeallocnames)

        end

        kwline = Expr(:kw, :_lineno_, __source__.line)
        push!(direct.args, kwline)

        kwfile = Expr(:kw, :_filepath_, string(__source__.file))
        push!(direct.args, kwfile)

        direct.args[1] = :jai_directive

        push!(tmp.args, esc(direct))
    end

    #dump(tmp)
    return(tmp)
end

macro jkernel(kernelname, accelname, kernelspec)

    tmp = Expr(:call)
    push!(tmp.args, :jai_kernel_init)

    push!(tmp.args, String(kernelname))
    push!(tmp.args, String(accelname))
    push!(tmp.args, esc(kernelspec))

    kwline = Expr(:kw, :_lineno_, __source__.line)
    push!(tmp.args, kwline)

    kwfile = Expr(:kw, :_filepath_, string(__source__.file))
    push!(tmp.args, kwfile)

    #dump(tmp)
    return(tmp)

end


macro jlaunch(largs...)
    tmp = Expr(:call)
    push!(tmp.args, :launch_kernel)
    innames = String[]
    outnames = String[]

    flag = true

    for larg in largs
        if larg isa Symbol
            if flag
                push!(tmp.args, string(larg))
                flag = false
            else
                push!(innames, String(larg))
                push!(tmp.args, esc(larg))
            end

        elseif larg.head == :parameters
            for param in larg.args
                if param.head  == :kw && param.args[1] == :output
                    for ovar in param.args[2].args
                        push!(outnames, String(ovar))
                    end
                end
            end
            push!(tmp.args, esc(larg))
        end
    end

    kwinnames = Expr(:kw, :innames, Expr(:tuple, innames...))
    push!(tmp.args, kwinnames)

    kwoutnames = Expr(:kw, :outnames, Expr(:tuple, outnames...))
    push!(tmp.args, kwoutnames)

    kwline = Expr(:kw, :_lineno_, __source__.line)
    push!(tmp.args, kwline)

    kwfile = Expr(:kw, :_filepath_, string(__source__.file))
    push!(tmp.args, kwfile)

    #dump(tmp)
    return(tmp)

end

macro jaccel(accel, clauses...)

    tmp = Expr(:block)

    eval(quote import AccelInterfaces.jai_directive   end)
    eval(quote import AccelInterfaces.jai_kernel_init end)
    eval(quote import AccelInterfaces.jai_accel_init  end)
    eval(quote import AccelInterfaces.jai_accel_fini  end)

    init = Expr(:call)
    push!(init.args, :jai_accel_init)
    push!(init.args, string(accel))

    for clause in clauses

        if clause.args[1] == :constant
            const_vars = clause.args[2:end]
            const_names = [String(n) for n in const_vars]
            const_vars = (esc(c) for c in const_vars)

            push!(init.args, Expr(:kw, :const_vars, Expr(:tuple, const_vars...))) 
            push!(init.args, Expr(:kw, :const_names, Expr(:tuple, const_names...)))

        elseif clause.args[1] == :device
            device = (esc(d) for d in clause.args[2:end])

            push!(init.args, Expr(:kw, :device, Expr(:tuple, device...))) 

        elseif clause.args[1] == :framework
            framework = (String(f) for f in clause.args[2:end])

            push!(init.args, Expr(:kw, :framework, Expr(:tuple, framework...))) 

        elseif clause.args[1] == :compile
            compile = (esc(c) for c in clause.args[2:end])

            push!(init.args, Expr(:kw, :compile, Expr(:tuple, compile...))) 

        elseif clause.args[1] == :set

            for kwarg in clause.args[2:end]
                if kwarg.head == :kw
                    push!(init.args, Expr(:kw, kwarg.args[1], esc(kwarg.args[2]))) 
                else
                    error("set clause allows keyword argument only.")
                end
            end

        else
            error(string(clause.args[1]) * " is not supported.")

        end
    end

    kwline = Expr(:kw, :_lineno_, __source__.line)
    push!(init.args, kwline)

    kwfile = Expr(:kw, :_filepath_, string(__source__.file))
    push!(init.args, kwfile)

    push!(tmp.args, init)

    #dump(tmp)
    return(tmp)
end


macro jdecel(accel, clauses...)

    tmp = Expr(:block)

    fini = Expr(:call)
    push!(fini.args, :jai_accel_fini)
    push!(fini.args, string(accel))

    for clause in clauses
    end

    kwline = Expr(:kw, :_lineno_, __source__.line)
    push!(fini.args, kwline)

    kwfile = Expr(:kw, :_filepath_, string(__source__.file))
    push!(fini.args, kwfile)

    push!(tmp.args, fini)

    #dump(tmp)
    return(tmp)
end



end
