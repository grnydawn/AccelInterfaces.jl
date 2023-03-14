# kernel.jl: implement functions for handling Jai KNL file


# TODO eval kernel section

function parse_header(hdr::Union{String, SubString}) :: Vector{JAI_TYPE_KERNELHDR}

    frames = Vector{Union{JAI_TYPE_FRAMEWORK, Symbol}}()
    params = nothing
    names  = Vector{String}()

    hdr = strip(hdr)
    
    if length(hdr) < 3
        error("Knl file header syntax error: " * hdr)
    end

    hdr = hdr[2:end-1]

    colons = Vector{Int32}()
    for r in findall(":", hdr)
        push!(colons, r.start)
    end

    if length(colons) == 0
        push!(colons, length(hdr)+1)
    end

    for colon in colons
        expr = Meta.parse(hdr[1:(colon-1)])

        if expr isa Symbol
            push!(frames, get(JAI_MAP_SYMBOL_FRAMEWORK, expr, expr))
        elseif all(a isa Symbol for a in expr.args)
            for arg in expr.args
                push!(frames, get(JAI_MAP_SYMBOL_FRAMEWORK, arg, arg))
            end
        end

        if length(frames) > 0
            if colon < length(hdr)
                params = Meta.parse("("*hdr[colon+1:end]*",)")
                while length(params.args) > 0 && params.args[1] isa Symbol
                    push!(names, string(popfirst!(params.args)))
                end
            end
            break
        end
    end


    output = Vector{JAI_TYPE_KERNELHDR}()
    for frame in frames
        push!(output, JAI_TYPE_KERNELHDR(frame, names, params))
    end

    return output
end

function parse_body(buf::Vector{String}) :: JAI_TYPE_KERNELBODY
    return JAI_TYPE_KERNELBODY(join(buf, "\n"))
end

function parse_kerneldef(kdef::String) :: JAI_TYPE_KERNELDEF

    sbuf = Vector{Vector{Union{Vector{JAI_TYPE_KERNELHDR}, JAI_TYPE_KERNELBODY}}}()

    doc = nothing
    is_hdr = false

    
    try
        if isfile(kdef)
            open(kdef, "r") do io
                kdef = read(io, String)
            end
        end
    catch err
    end

    buf = Vector{String}()

    for line in split(kdef, "\n")
        s = strip(line)

        if length(s) < 1
            continue

        elseif is_hdr
            if s[end] == ']'
                push!(buf, s)
                hdrs = parse_header(join(buf, "\n"))
                if hdrs isa Vector{JAI_TYPE_KERNELHDR}
                    push!(sbuf, [hdrs])
                    is_hdr = false
                    empty!(buf)
                else
                    push!(buf, line)
                end
            else
                push!(buf, line)
            end
        elseif s[1] == '['
            if length(sbuf) > 0
                push!(sbuf[end], parse_body(buf))
            elseif doc == nothing
                doc = join(buf, "\n")
            end
            empty!(buf)

            if s[end] == ']'
                hdrs = parse_header(s)
                if hdrs isa Vector{JAI_TYPE_KERNELHDR}
                    push!(sbuf, [hdrs])
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

    if length(buf) > 0
        if length(sbuf) > 0
            push!(sbuf[end], parse_body(buf))
        elseif doc == nothing
            doc = join(buf, "\n")
        end
    end

    secs = Vector{JAI_TYPE_KERNELSEC}()
    ksids = Vector{UInt32}()
    ksecs = Vector{JAI_TYPE_KERNELINITSEC}()

    for (hdrs, body) in sbuf
        for hdr in hdrs
            ksid = generate_jid(hdr, body.body)
            if hdr.frame == :kernel
                modname = Symbol(ksid)
                modbody = Meta.parse(body.body)
                env = @eval baremodule $modname $modbody end
                push!(ksecs, JAI_TYPE_KERNELINITSEC(ksid, hdr.argnames, env))
            else
                push!(secs, JAI_TYPE_KERNELSEC(ksid, hdr, body))
            end
            push!(ksids, ksid)
        end
    end

    kdid = generate_jid(ksids...)
    return JAI_TYPE_KERNELDEF(kdid, doc, ksecs, secs)
end

function select_section(
        frame::JAI_TYPE_FRAMEWORK,
        kdef::JAI_TYPE_KERNELDEF
    ) :: JAI_TYPE_KERNELSEC

    for sec in kdef.sections
        if sec.header.frame == frame
            expr = sec.header.params
            if length(kdef.init) > 0
                for isec in kdef.init
                    params = @eval isec.env $expr
                    if (params isa Nothing || !haskey(params, :enable) ||
                        getproperty(params, :enable))
                        return sec
                    end
                end
            else
                return sec
            end
        end
    end

    throw(JAI_ERROR_NOVALID_SECTION())
end

function get_knlbody(ctx::JAI_TYPE_CONTEXT_KERNEL) :: String
    ksec = select_section(ctx.framework.type, ctx.kdef)
    return ksec.body.body
end

