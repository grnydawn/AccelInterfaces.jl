# compiler.jl: implement functions for compiler support

struct JAI_TYPE_COMPILER
    name            ::String
    path            ::String
    opt_shared      ::String
    opt_frame       ::String
    opt_append      ::String
end

const JAI_TYPE_COMPILERS = OrderedDict{JAI_TYPE_FRAMEWORK, Vector{JAI_TYPE_COMPILER}}
const JAI_COMPILERS = JAI_TYPE_COMPILERS()

function append_compiler(
        frame   ::JAI_TYPE_FRAMEWORK,
        comps   ::JAI_TYPE_COMPILERS,
        ctx     ::JAI_TYPE_COMPILER
    )
            
    if frame in keys(comps)
        idx = findfirst(f -> f.name == ctx.name, comps[frame])
        if idx == nothing
            push!(comps[frame], ctx)
        else
            comps[frame][idx] = ctx
        end
    else
        c = Vector{JAI_TYPE_COMPILER}()
        push!(c, ctx)
        comps[frame] = c
    end
end

function get_compiles(
        compiles::Vector{String},
        comps::Vector{JAI_TYPE_COMPILER}
    ) :: Nothing

    for comp in comps
        c = (comp.path * " " * comp.opt_shared * " " *
             comp.opt_frame * " " * comp.opt_append)
        push!(compiles, c)
    end
end

function get_compiles(
        frame   ::JAI_TYPE_FRAMEWORK,
        compiler::JAI_TYPE_CONFIG
    ) :: Vector{String}


    compiles= Vector{String}()
    tcomps  = JAI_TYPE_COMPILERS()

    for (name, cinfo) in compiler
        path    = cinfo[:path]
        shared  = cinfo[:opt_shared]
        frames  = cinfo[:opt_frameworks]
        append  = get(cinfo, :opt_append, "")

        for (fsym, fopt) in frames
            f = JAI_MAP_SYMBOL_FRAMEWORK[fsym]
            ctx = JAI_TYPE_COMPILER(name, path, shared, fopt, append)
            append_compiler(f, tcomps, ctx)
        end
    end

    if frame in keys(tcomps)
        get_compiles(compiles, tcomps[frame])
    end

    if frame in keys(JAI_COMPILERS)
        get_compiles(compiles, JAI_COMPILERS[frame])
    end

    for (f, ctxs) in tcomps
        for ctx in ctxs
            append_compiler(f, JAI_COMPILERS, ctx)
        end
    end

    return compiles
end
