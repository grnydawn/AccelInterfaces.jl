# util.jl: implement utility functions
#

import Pidfile.mkpidlock
import Serialization.serialize
import SHA.sha1

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

extract_name_from_frametype(x) = lowercase(split(string(x), ".")[end][10:end])

