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

name_from_frame(x) = lowercase(split(string(typeof(x)), ".")[end][10:end])

# TODO: customize for Jai
macro jdebug(e) :(@info  sprint(showerror, $(esc(e)))) end
macro jinfo(e)  :(@info  sprint(showerror, $(esc(e)))) end
macro jwarn(e)  :(@warn  sprint(showerror, $(esc(e)))) end
macro jerror(e) :(@error sprint(showerror, $(esc(e)))) end

function pack_arg(
        arg::JAI_TYPE_DATA;
        name="",
        argtype=JAI_ARG_IN,
        shape=()
    ) :: JAI_TYPE_ARG

    if name == ""
        name = "var_" * randstring(4) 
    end

    if arg isa AbstractArray
        shape = size(arg)
    end

    return (arg, name, argtype, shape)
end


function check_retval(retval::Int64)
    if retval != 0
        throw(JAI_ERROR_NONZERO_RETURN())
    end
end

function get_context(
        contexts    ::Vector{JAI_TYPE_CONTEXT_ACCEL},
        aid         ::UInt32
    )

    for ctx in contexts
        if ctx.aid == aid
            return ctx
        end
    end

    return nothing
end


function get_context(
        contexts    ::Vector{JAI_TYPE_CONTEXT_ACCEL},
        name        ::String
    )

    for ctx in contexts
        if ctx.aname == name
            return ctx
            break
        end
    end

    if name == "" && length(contexts) == 1
        return contexts[1]
    else
        return nothing
    end
end

get_accel(arg)     = get_context(JAI["ctx_accels"], arg)
get_kernel(arg)    = get_context(JAI["ctx_kernels"], arg)

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
