# util.jl: implement utility functions
#

import Pidfile.mkpidlock
import Serialization.serialize
import SHA.sha1
import Random.randstring

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

# Jai global configurations
JAI = Dict{String, Union{String, Number, Bool, Nothing}}(
        "debug" => true
    )

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
