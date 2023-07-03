# api.jl: implement user interface macros
#


function _jaccel_clause_handler(output, clauses)

    # parse clauses
    for clause in clauses
        if clause.args[1] == :constant
            const_vars = :(())
            const_names = :(())
            for cvar in clause.args[2:end]
                if cvar isa Symbol
                    push!(const_vars.args, esc(cvar))
                    push!(const_names.args, string(cvar))
                else
                    error("Wrong jaccel clause: " * string(clause.args))
                end
            end

            push!(output.args, Expr(:kw, :const_vars, const_vars))
            push!(output.args, Expr(:kw, :const_names, const_names))

        elseif clause.args[1] in (:device, :machine)
            t = :(())
            for d in clause.args[2:end]
                push!(t.args, esc(d))
            end
            push!(output.args, Expr(:kw, clause.args[1], t))

        #elseif clause.args[1] in (:framework, :set, :compiler)
        elseif clause.args[1] in (:set, :compiler)

            d = :(JAI_TYPE_CONFIG())

            for item in clause.args[2:end]

                if item isa Symbol
                    key = item
                    value = :nothing

                elseif item.head == :kw
                    key = item.args[1]
                    value = length(item.args)>1 ? esc(item.args[2]) : nothing

                else
                    error("Wrong syntax: " * string(clause))
                end

                #if clause.args[1]== :framework
                #    push!(d.args, Expr(:call, :(=>), JAI_MAP_SYMBOL_FRAMEWORK[key], value))
                #else
                    push!(d.args, Expr(:call, :(=>), string(key), value))
                #end
            end

            push!(output.args, Expr(:kw, clause.args[1], d))

        else
            error(string(clause.args[1]) * " is not supported.")

        end
    end
end


"""
    @jconfig [clauses...]

Configure Jai environment


# Arguments

See also [`@jaccel`](@jaccel), [`@jdecel`](@jdecel)

# Examples
```julia-repl
julia> @jconfig framework(fortran="gfortran -fPIC -shared -g")
```

# Implementation
T.B.D.

"""
macro jconfig(clauses...)

    config = Expr(:call)
    push!(config.args, :jai_config)
    
    # handle jconfig specific inputs

    # collect jaccel inputs
    _jaccel_clause_handler(config, clauses)

    # exclude jaccel specific inputs

    push!(config.args, __source__.line)
    push!(config.args, string(__source__.file))

    return(config)
end



"""
    @jaccel [name][, clauses...]

Create accelerator context.

If `name` is not specified, this context can be accessed only as the currently active context.

# Arguments
- `name`::String: a unique name for this accelerator context
- `framework`::NamedTuple: 
- `device`::Integer: 
- `compiler`::Integer: 
- `machine`::Integer: 
- `constant`::Tuple of Variable literal :
- `set`::Named Tuple: 

See also [`@jdecel`](@jdecel), [`@jkernel`](@jkernel)

# Examples
```julia-repl
julia> @jaccel myacc framework(fortran="gfortran -fPIC -shared -g")
AccelInfo
```

# Implementation
T.B.D.

"""
macro jaccel(clauses...)

    init = Expr(:call)
    push!(init.args, :jai_accel)

    nclauses = length(clauses)

    # parse accelname
    if nclauses > 0 && clauses[1] isa Symbol
        push!(init.args, string(clauses[1]))
        start_index = 2
    else
        push!(init.args, "")
        start_index = 1
    end

    _jaccel_clause_handler(init, clauses[start_index:end])

    push!(init.args, __source__.line)
    push!(init.args, string(__source__.file))

    #dump(clauses)
    
    return(init)
end


function _jdata(symalloc, jai_alloctype, symupdate, jai_updatetype, directs, sline, sfile)

    tmp = Expr(:block)

    ndirects = length(directs)

    # parse accelname
    if ndirects > 0 && directs[1] isa Symbol
        stracc = string(directs[1])
        start_index = 2
    else
        stracc = ""
        start_index = 1
    end

    allocs = Expr[]
    nonallocs = Expr[]
    config = :(JAI_TYPE_CONFIG())
    alloccount = 1
    updatetocount = 1
    allocnames = String[]
    updatenames = String[]
    control = String[]

    # parse directs
    for direct in directs[start_index:end]

        if direct isa Symbol
            push!(control, string(direct))

        elseif direct.args[1] == symalloc

            for idx in range(2, stop=length(direct.args))
                push!(allocnames, string(direct.args[idx]))
                direct.args[idx] = esc(direct.args[idx])
            end

            insert!(direct.args, 2, stracc)
            insert!(direct.args, 3, jai_alloctype)
            insert!(direct.args, 4, alloccount)
            alloccount += 1
            push!(allocs, direct)

        elseif direct.args[1] == symupdate

            for idx in range(2, stop=length(direct.args))
                push!(updatenames, string(direct.args[idx]))
                direct.args[idx] = esc(direct.args[idx])
            end

            insert!(direct.args, 2, stracc)
            insert!(direct.args, 3, jai_updatetype)
            insert!(direct.args, 4, updatetocount)
            updatetocount += 1
            push!(nonallocs, direct)

        elseif direct.args[1] in (:enable_if,)

            key = direct.args[1]
            value = direct.args[2]
            push!(config.args, Expr(:call, :(=>), string(key), esc(value)))
            #push!(config.args, esc(direct.args[2]))

        # for later use
        elseif direct.args[1] in ()

            config = :(JAI_TYPE_CONFIG())

            for item in direct.args[2:end]

                if item isa Symbol
                    key = item
                    value = :nothing

                elseif item.head == :kw
                    key = item.args[1]
                    value = length(item.args)>1 ? esc(item.args[2]) : nothing

                else
                    error("Wrong syntax: " * string(direct))
                end

                #if direct.args[1]== :framework
                #    push!(d.args, Expr(:call, :(=>), JAI_MAP_SYMBOL_FRAMEWORK[key], value))
                #else
                    push!(config.args, Expr(:call, :(=>), string(key), value))
                #end
            end

        elseif direct.args[1] in (:async,)
            push!(control, string(direct.args[1]))

        else
            error(string(direct.args[1]) * " is not supported.")

        end

    end

    if symalloc == :alloc
        _buffer = (allocs..., nonallocs...)
    elseif symalloc == :delete
        _buffer = (nonallocs..., allocs...)
    else
        error("unknown symbol in jai data macro: " * string(symalloc))
    end

    for direct in _buffer

        if direct.args[1] == symupdate
            insert!(direct.args, 5, updatenames)

        elseif direct.args[1] == symalloc
            insert!(direct.args, 5, allocnames)

        else
            error("unknown data directive in jai data macro: " *
                  string(direct.args[1]))

        end

        insert!(direct.args, 6, config)
        insert!(direct.args, 7, control)
        insert!(direct.args, 8, sline)
        insert!(direct.args, 9, string(sfile))

        direct.args[1] = :jai_data

        push!(tmp.args, direct)
    end

    #dump(tmp)
    return(tmp)
end


"""
    @jenterdata [name][, clauses...]

Allocate device memory or copy data to device memory.

If `name` is not specified, the currently active accel context will be used.

# Arguments
- `name`::String: a unique name for the accelerator context
- `alloc`::NTuple: 
- `updateto`::NTuple: 
- `async`::Keyword :

See also [`@jaccel`](@jaccel), [`@jkernel`](@jkernel)

# Examples
```julia-repl
julia> @jenterdata myacc alloc(X, Y, Z), updateto(X, Y, Z)
0
```

# Implementation
T.B.D.

"""
macro jenterdata(directs...)
    return _jdata(:alloc, JAI_ALLOCATE, :updateto, JAI_UPDATETO,
                  directs, __source__.line, __source__.file)
end


"""
    @jexitdata [name][, clauses...]

Dealloc device memory or copy data from device memory.

If `name` is not specified, the currently active accel context will be used.

# Arguments
- `name`::String: a unique name for the accelerator context
- `delete`::NTuple: 
- `updatefrom`::NTuple: 
- `async`::Keyword :

See also [`@jaccel`](@jaccel), [`@jkernel`](@jkernel)

# Examples
```julia-repl
julia> @jexitdata myacc delete(X, Y, Z), updatefrom(X, Y, Z)
0
```

# Implementation
T.B.D.

"""
macro jexitdata(directs...)
    return _jdata(:delete, JAI_DEALLOCATE, :updatefrom, JAI_UPDATEFROM,
                  directs, __source__.line, __source__.file)
end


"""
    @jkernel kerneldef, [kernelname, [accelname, ]][clauses...]

Create kernel context.

If `kernelname` or `accelname` is not specified, the currently active accel or kernel context will be used.

# Arguments
- `kerneldef`::String: Jai kernel definition
- `kernelname`::String: Kernel context name
- `accelname`::String: Accel context name

See also [`@jaccel`](@jaccel), [`@jenterdata`](@jenterdata)

# Examples
```julia-repl
julia> @jkernel knlfilepath mykernel myaccel
0
```

# Implementation
T.B.D.

"""
macro jkernel(kdef, clauses...)

    expr = Expr(:call)
    push!(expr.args, :jai_kernel)

    push!(expr.args, esc(kdef)) # kernel definition

    nclauses = length(clauses)
    start_index = 1

    knlname = ""
    accname = ""

    for clause in clauses
        if clause isa Symbol
            name = string(clause)
            if knlname == ""
                knlname = name
            elseif accname == ""
                accname = name
            else
                error("Wrong jkernel symbol: " * name)
            end
            start_index += 1
        else
            break
        end
    end

    push!(expr.args, knlname) # kernel context name
    push!(expr.args, accname) # accel context name
    push!(expr.args, __source__.line)
    push!(expr.args, string(__source__.file))

    for clause in clauses[start_index:end]

        if clause.args[1] in (:framework, :compiler)

            d = :(JAI_TYPE_CONFIG())

            for item in clause.args[2:end]

                if item isa Symbol
                    key = item
                    value = :nothing

                elseif item.head == :kw
                    key = item.args[1]
                    value = length(item.args)>1 ? esc(item.args[2]) : nothing

                else
                    error("Wrong jaccel syntax: " * string(clause))
                end

                if clause.args[1]== :framework
                    push!(d.args, Expr(:call, :(=>), JAI_MAP_SYMBOL_FRAMEWORK[key], value))
                else
                    push!(d.args, Expr(:call, :(=>), string(key), value))
                end
            end

            push!(expr.args, Expr(:kw, clause.args[1], d))

        else
            error(string(clause.args[1]) * " is not supported.")

        end

    end

    #dump(expr)
    return(expr)

end


"""
    @jlaunch [kernelname, [accelname, ]][clauses...]

Launch a kernel on an accelerator.

If `kernelname` or `accelname` is not specified, the currently active accel or kernel context will be used.

# Arguments
- `kernelname`::String: Kernel context name
- `accelname`::String: Accel context name

See also [`@jaccel`](@jaccel), [`@jkernel`](@jkernel)

# Examples
```julia-repl
julia> @jlaunch mykernel myaccel input(X, Y) output(Z)
0
```

# Implementation
T.B.D.

"""
macro jlaunch(clauses...)

    tmp = Expr(:call)
    push!(tmp.args, :jai_launch)
    input = :(())
    output = :(())
    innames = String[]
    outnames = String[]

    nclauses = length(clauses)

    knlname = ""
    accname = ""
    start_index = 1

    for clause in clauses
        if clause isa Symbol
            if knlname == ""
                knlname = string(clause)
            elseif accname == ""
                accname = string(clause)
            else
                error("Wrong jlaunch symbol")
            end
            start_index += 1
        else
            break
        end
    end

    push!(tmp.args, string(knlname))
    push!(tmp.args, string(accname))

    frames = :(JAI_TYPE_CONFIG())

    for clause in clauses[start_index:end]
        if clause.head == :call
            if clause.args[1] == :input
                for invar in clause.args[2:end]
                    push!(innames, String(invar))
                    push!(input.args, esc(invar))
                end
            elseif clause.args[1] == :output
                for outvar in clause.args[2:end]
                    push!(outnames, String(outvar))
                    push!(output.args, esc(outvar))
                end
            elseif clause.args[1] in keys(JAI_MAP_SYMBOL_FRAMEWORK)
                
                if length(clause.args) > 1
                    value = :(OrderedDict{String, JAI_TYPE_CONFIG_VALUE}())

                    for item in clause.args[2:end]
                        push!(value.args, Expr(:call, :(=>),
                            String(item.args[1]), esc(item.args[2])))
                    end
                else
                    value = nothing
                end

                push!(frames.args, Expr(:call, :(=>),
                        JAI_MAP_SYMBOL_FRAMEWORK[clause.args[1]], value))

            else
                error("Wrong jlaunch clause: " * string(clause.args[1]))
            end
        end
    end

    push!(tmp.args, input)
    push!(tmp.args, output)
    push!(tmp.args, innames)
    push!(tmp.args, outnames)
    push!(tmp.args, __source__.line)
    push!(tmp.args, string(__source__.file))
    push!(tmp.args, frames)

    #dump(tmp)
    return(tmp)

end


"""
    @jwait [kernelname[ accelname]]

Wait to finish device operation

If `kernelname` or `accelname` is not specified, the currently active accel or kernel context will be used.

# Arguments

See also [`@jaccel`](@jaccel), [`@jkernel`](@jkernel)

# Examples
```julia-repl
julia> @jwait mykernel
0
```

# Implementation
T.B.D.

"""
macro jwait(clauses...)

    expr = Expr(:call)

    nclauses = length(clauses)

    if nclauses > 0 && clauses[1] isa Symbol
        name = string(clauses[1])
    else
        name = ""
    end

    push!(expr.args, :jai_wait)
    push!(expr.args, name)

    push!(expr.args, __source__.line)
    push!(expr.args, string(__source__.file))

    return(expr)
end

"""
    @jdecel [name][, clauses...]

Destroy accelerator context.

If `name` is not specified, this context can be accessed only as the currently active context.

# Arguments
- `name`::String: a unique name for this accelerator context

See also [`@jaccel`](@jaccel), [`@jkernel`](@jkernel)

# Examples
```julia-repl
julia> @jdecel myacc
```

# Implementation
T.B.D.

"""
macro jdecel(clauses...)

    fini = Expr(:call)

    nclauses = length(clauses)

    # parse accelname
    if nclauses > 0 && clauses[1] isa Symbol
        accname = string(clauses[1])
        start_index = 2
    else
        accname = ""
        start_index = 1
    end

    push!(fini.args, :jai_decel)
    push!(fini.args, accname)
    push!(fini.args, __source__.line)
    push!(fini.args, string(__source__.file))

    return(fini)
end


"""
    @jdiff [name] A B [clauses...] begin ... end

Analize difference between A and B

If `name` is not specified, this context can be accessed only as the currently active context.

# Arguments
- `name`::String: a unique name for this accelerator context
- `A`, `B`::Test cases

# Examples
```julia-repl
julia> @jdiff myacc fort_impl(USE_HIP=false) hip_impl(USE_HIP=true) begin
...
end
```

# Implementation
T.B.D.

"""
macro jdiff(items...)

    block   = Expr(:block)
    diff    = Expr(:call)
    diffA   = Expr(:call)
    diffB   = Expr(:call)
    diffend = Expr(:call)

    nitems = length(items)

    # parse accelname
    if nitems > 0 && items[1] isa Symbol
        accname = string(items[1])
        idx = 2
    else
        accname = ""
        idx = 1
    end

    if nitems < idx + 2 # includes body block
        error("Not enough diff cases.")
    elseif items[idx].head != :call || items[idx+1].head != :call 
        error("Wrong case syntax in @jdiff.")
    end

    #dump(items[idx])
    Acase = items[idx]
    Bcase = items[idx+1]

    Aname = string(Acase.args[1])
    Bname = string(Bcase.args[1])

    push!(diff.args, :jai_diff)
    push!(diff.args, accname)
    push!(diff.args, (Aname, Bname))
    push!(diff.args, __source__.line)
    push!(diff.args, string(__source__.file))
    push!(block.args, diff)

    push!(diffA.args, :_jai_diffA)
    push!(diffA.args, accname)
    push!(diffA.args, (Aname, Bname))
    push!(diffA.args, __source__.line)
    push!(diffA.args, string(__source__.file))
    push!(block.args, diffA)

    for kwargs in Acase.args[2:end]
        if kwargs.head != :kw
            error("Wrong case syntax in @jdiff.")
        end

        push!(block.args, Expr(:(=), kwargs.args[1], esc(kwargs.args[2])))
    end

    body = esc(items[end])

    push!(block.args, body)

    push!(diffB.args, :_jai_diffB)
    push!(diffB.args, accname)
    push!(diffB.args, (Aname, Bname))
    push!(diffB.args, __source__.line)
    push!(diffB.args, string(__source__.file))
    push!(block.args, diffB)

    for kwargs in Bcase.args[2:end]
        if kwargs.head != :kw
            error("Wrong case syntax in @jdiff.")
        end

        push!(block.args, Expr(:(=), kwargs.args[1], esc(kwargs.args[2])))
    end

    push!(block.args, body)

    push!(diffend.args, :_jai_diffend)
    push!(diffend.args, accname)
    push!(diffend.args, (Aname, Bname))
    push!(diffend.args, __source__.line)
    push!(diffend.args, string(__source__.file))

    push!(block.args, diffend)

    return(block)
end
