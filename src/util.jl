# util.jl: implement utility functions
#
import Pidfile: mkpidlock, LockMonitor
import Serialization.serialize
import SHA.sha1
import Random.randstring
import DataStructures.Stack

function locked_filetask(pidfile::String, target::String, fn::Function,
            args...; delete=false)

    check = (d, t) -> (d ? ispath(t) : !ispath(t))

    if check(delete, target)
        lock = nothing

        try
            lock = mkpidlock(pidfile, stale_age=3)

            if check(delete, target)
                fn(args...)
            end

        catch err
            rethrow(err)

        finally

            if lock isa LockMonitor
                close(lock)
            end
        end
    end

end

function generate_jid(args...) ::UInt32

    ret = 0x00000000::UInt32

	tempid = UInt64(0)
	for arg in args
		tempid = hash(arg, tempid)
	end

	if tempid in keys(JAI["jids"])
		return JAI["jids"][tempid]
	end

    io = IOBuffer()
    serialize(io, args)

    for n in sha1(String(take!(io)))[1:4]
        ret |= n
        ret <<= 8
    end

	JAI["jids"][tempid] = ret

    return ret
end

function vector_to_uint32(uid::Vector{UInt8}) :: UInt32
    ret = 0x00000000::UInt32
    for n in uid[1:4]
        ret |= n
        ret <<= 8
    end
    return ret
end

function generate_prefix(name :: String, jid :: UInt32)
    return "jai_" * name * "_" * string(jid, base=16) * "_"
end

name_from_frame(x) = lowercase(split(string(typeof(x)), ".")[end][10:end])

# TODO: customize for Jai
macro jdebug(e) :(@info  sprint(showerror, $(esc(e)))) end
macro jinfo(e)  :(@info  sprint(showerror, $(esc(e)))) end
macro jwarn(e)  :(@warn  sprint(showerror, $(esc(e)))) end
macro jerror(e) :(@error sprint(showerror, $(esc(e)))) end

function pack_arg(
        arg     ::JAI_TYPE_DATA,
        externs ::Union{Nothing, Dict{UInt64, String}},
        apitype ::Union{Nothing, JAI_TYPE_API};
        name    ::String="",
        inout   ::Union{Nothing, JAI_TYPE_INOUT}=nothing
    ) :: JAI_TYPE_ARG

    jid = UInt64(0)

    try
        try
            jid = convert(UInt64, pointer(parent(arg)))
        catch err
            jid = convert(UInt64, pointer(arg))
        end
    catch err
        jid = hash(arg)
    end

    extname = ""

    if externs isa Dict{UInt64, String}
        if apitype isa JAI_TYPE_ALLOCATE
            lenext = length(externs)
            extname = "jai_extern_$(lenext)_$(name)"
            externs[jid] = extname
        elseif apitype isa JAI_TYPE_API_DATA || apitype isa JAI_TYPE_LAUNCH
            if haskey(externs, jid)
                extname = externs[jid]
            else
            end
        else
        end
    end

    jid = hash(apitype, jid)
    jid = hash(name, jid)
    jid = hash(inout, jid)

    if haskey(JAI["pack_jids"], jid)
        s = JAI["pack_jids"][jid]
        if extname != s[8]
            JAI["pack_jids"][jid] = (s[1], s[2], s[3], s[4], s[5], s[6], s[7], extname)
        end

        return JAI["pack_jids"][jid]
    end

    if inout == nothing
        if apitype isa JAI_TYPE_API
            inout=JAI_MAP_APITYPE_INOUT[apitype]
        else
            inout = JAI_ARG_IN
        end
    end

    if name == ""
        name = "var_" * randstring(4) 
    end

    bytes   = sizeof(arg)

    if arg isa AbstractArray

        shape   = size(arg)
        dtype   = eltype(arg)

        if arg isa OffsetArray
            pobj = parent(arg)
            bytes   = sizeof(pobj)
            offsets = arg.offsets
        else
            offsets = Tuple(1 for _ in 1:length(shape))
        end

    elseif arg isa Tuple

        shape   = Tuple(length(arg))
        dtype   = eltype(arg)
        offsets = Tuple(1)

    else
        shape   = ()
        offsets = ()
        dtype   = typeof(arg)
    end
    
    JAI["pack_jids"][jid] = (arg, dtype, name, inout, bytes, shape, offsets, extname)
    return JAI["pack_jids"][jid]
end


function pack_args(
        innames     ::Vector{String},
        indata      ::NTuple{N, JAI_TYPE_DATA} where N,
        outnames    ::Vector{String},
        outdata     ::NTuple{N, JAI_TYPE_DATA} where N,
        externs     ::Union{Nothing, Dict{UInt64, String}},
        apitype     ::Union{Nothing, JAI_TYPE_API}
    ) :: JAI_TYPE_ARGS

    buf = OrderedDict()

    for (n, d) in zip(innames, indata)
        buf[n] = [d, JAI_ARG_IN]
    end

    for (n, d) in zip(outnames, outdata)
        if n in keys(buf)
            buf[n] = [d, JAI_ARG_INOUT]
        else
            buf[n] = [d, JAI_ARG_OUT]
        end
    end

    args = JAI_TYPE_ARGS()

    for (n, (d, inout)) in buf
        arg = pack_arg(d, externs, apitype, name=n, inout=inout)
        push!(args, arg)
    end

    return args
end


function check_retval(retval::Int64)
    if retval != 0
        throw(JAI_ERROR_NONZERO_RETURN())
    end
end


function get_context(
        contexts    ::Vector{JAI_TYPE_CONTEXT_ACCEL},
        aid         ::UInt32
    ) :: Union{JAI_TYPE_CONTEXT_ACCEL, Nothing}

    for ctx in contexts
        if ctx.aid == aid
            return ctx
        end
    end

    return nothing
end


function get_context(
        contexts    ::Vector{JAI_TYPE_CONTEXT_KERNEL},
        kid         ::UInt32
    ) :: Union{JAI_TYPE_CONTEXT_KERNEL, Nothing}

    for ctx in contexts
        if ctx.kid == kid
            return ctx
        end
    end

    return nothing
end

function get_context(
        contexts    ::Vector{JAI_TYPE_CONTEXT_ACCEL},
        name         ::String
    ) :: Union{JAI_TYPE_CONTEXT_ACCEL, Nothing}

    if name == ""
        if length(contexts) > 0
            return contexts[end]
        end
        return nothing
    end

    for ctx in contexts
        if ctx.aname == name
            return ctx
            break
        end
    end

    return nothing
end


function get_context(
        contexts    ::Vector{JAI_TYPE_CONTEXT_KERNEL},
        name         ::String
    ) :: Union{JAI_TYPE_CONTEXT_KERNEL, Nothing}

    if name == ""
        if length(contexts) > 0
            return contexts[end]
        end
        return nothing
    end

    for ctx in contexts
        if ctx.kname == name
            return ctx
            break
        end
    end

    return nothing
end

get_accel(arg)          = get_context(JAI["ctx_accels"], arg)
get_kernel(actx, arg)   = get_context(actx.ctx_kernels, arg)

function jaifmt(_T::String; kwargs...)

    #T = replace(_T, "{{"=>"__JAI1__", "}}"=>"__JAI2__")
    _T = replace(_T, "{{" => "__JAI1__")
    T  = replace(_T, "}}" => "__JAI2__")

    names = Dict{Symbol, Any}(kwargs)
    output = Vector{String}()

    ostack = Stack{Int64}()
    prev   = 1
    N      = length(T)

    for i in 1:N
        if T[i] == '{'
            if i < N && T[i+1] == ' '
                error("ERROR: blank after { near: " * T[i:end])

            else
                push!(ostack, i)
                push!(output, T[prev:i-1])
                prev = nothing
            end

        elseif T[i] == '}'
            if T[i-1] == ' '
                error("ERROR: wrong Jai template formatting near: " * T[i-1:end])

            elseif length(ostack) == 0
                error("ERROR: pop from empty stack")
            else
                pidx = pop!(ostack)
                name = T[pidx+1:i-1]
                push!(output, string(names[Symbol(name)]))
                prev = i+1
            end

        elseif i == N

            if length(ostack) > 0
                error("ERROR: unused stack: " * string(ostack))

            elseif prev != nothing
                push!(output, T[prev:N])
            end
        end
    end

    return replace(replace(join(output), "__JAI1__"=>"{"), "__JAI2__"=>"}")
end
