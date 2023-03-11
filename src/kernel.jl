# kernel.jl: implement functions for handling Jai KNL file

function parse_header(hdr::Union{String, SubString}) :: JAI_TYPE_KERNELHDR
    return JAI_TYPE_KERNELHDR(hdr)
end

function parse_body(buf::Vector{String}) :: JAI_TYPE_KERNELBODY
    return JAI_TYPE_KERNELBODY(join(buf, "\n"))
end

function parse_kerneldef(kdef::String) :: JAI_TYPE_KERNELDEF

    sbuf = Vector{Vector{Union{JAI_TYPE_KERNELHDR, JAI_TYPE_KERNELBODY}}}()

    is_hdr = false

    
    if isfile(kdef)
        open(kdef, "r") do io
            kdef = read(io, String)
        end
    end

    buf = Vector{String}()

    for line in split(kdef, "\n")
        s = strip(line)

        if length(s) < 1
            continue

        elseif is_hdr
            if s[end] == ']'
                hdr = parse_header(join(buf, "\n") * s[1:end-1])
                if hdr isa JAI_TYPE_KERNELHDR
                    push!(sbuf, [hdr])
                    is_hdr = false
                    empty!(buf)
                else
                    push!(buf, line)
                end
            else
                push!(buf, line)
            end
        elseif s[1] == '['
            if length(buf) > 0 && length(sbuf) > 0
                push!(sbuf[end], parse_body(buf))
            end

            if s[end] == ']'
                hdr = parse_header(s[2:end-1])
                if hdr isa JAI_TYPE_KERNELHDR
                    push!(sbuf, [hdr])
                else
                    push!(buf, line)
                    is_hdr = true
                end
            else
                push!(buf, line)
                is_hdr = true
            end
        else
            push!(buf, line)
        end
    end

    if length(buf) > 0 && length(sbuf) > 0
        push!(sbuf[end], parse_body(buf))
    end

    secs = Vector{JAI_TYPE_KERNELSEC}()
    ksids = Vector{UInt32}()

    for (hdr, body) in sbuf
        ksid = generate_jid(hdr.header, body.body)
        push!(secs, JAI_TYPE_KERNELSEC(ksid, hdr, body))
        push!(ksids, ksid)
    end

    kdid = generate_jid(ksids...)
    return JAI_TYPE_KERNELDEF(kdid, secs)
end
