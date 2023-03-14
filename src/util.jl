# util.jl: implement utility functions
#
import Pidfile.mkpidlock
import Serialization.serialize
import SHA.sha1
import Random.randstring
import DataStructures.Stack

function locked_filetask(pidfile::String, target::String, fn::Function, args...) ::Nothing

    if !ispath(target)
        lock = nothing

        try
            lock = mkpidlock(pidfile, stale_age=3)

            if !ispath(target)
                fn(args...)
            end

        catch err
            rethrow(err)

        finally

            if lock != nothing
                close(lock)
            end
        end
    end

    return nothing
end


function generate_jid(args...) ::UInt32

    ret = 0x00000000::UInt32

    io = IOBuffer()
    serialize(io, args)

    for n in sha1(String(take!(io)))[1:4]
        ret |= n
        ret <<= 8
    end

    return ret
end

function generate_prefix(name :: String, jid :: UInt32)
    return "jai_" * name * string(jid, base=16)
end

name_from_frame(x) = lowercase(split(string(typeof(x)), ".")[end][10:end])

# TODO: customize for Jai
macro jdebug(e) :(@info  sprint(showerror, $(esc(e)))) end
macro jinfo(e)  :(@info  sprint(showerror, $(esc(e)))) end
macro jwarn(e)  :(@warn  sprint(showerror, $(esc(e)))) end
macro jerror(e) :(@error sprint(showerror, $(esc(e)))) end

function pack_arg(
        arg::JAI_TYPE_DATA;
        name="",
        inout=JAI_ARG_IN
    ) :: JAI_TYPE_ARG

    if name == ""
        name = "var_" * randstring(4) 
    end

    if arg isa OffsetArray
        shape   = size(arg)
        offsets = arg.offsets
        addr    = Base.unsafe_convert(Ptr{Clonglong}, arg)
        dtype   = eltype(arg)

    elseif arg isa AbstractArray
        shape   = size(arg)
        offsets = Tuple(1 for _ in 1:length(arg))
        addr    = Base.unsafe_convert(Ptr{Clonglong}, arg)
        dtype   = eltype(arg)

    else
        shape   = ()
        offsets = ()
        addr    = nothing
        dtype   = typeof(arg)
    end

    return (arg, dtype, name, inout, addr, shape, offsets)
end


function pack_args(
        innames     ::Vector{String},
        indata      ::NTuple{N, JAI_TYPE_DATA} where N,
        outnames    ::Vector{String},
        outdata     ::NTuple{N, JAI_TYPE_DATA} where N
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
        arg = pack_arg(d, name=n, inout=inout)
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
        if length(contexts) == 1
            return contexts[1]
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
        if length(contexts) == 1
            return contexts[1]
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

function jaifmt(_T; kwargs...)

    T = replace(_T, "{{"=>"__JAI1__", "}}"=>"__JAI2__")

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

    return replace(join(output), "__JAI1__"=>"{", "__JAI2__"=>"}")
end
