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

        elseif clause.args[1] in (:set, :compiler, :framework)

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

                if clause.args[1]== :framework
                    push!(d.args, Expr(:call, :(=>), JAI_MAP_SYMBOL_FRAMEWORK[key], value))
                else
                    push!(d.args, Expr(:call, :(=>), string(key), value))
                end
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
"""

macro jconfig(clauses...)

    config = Expr(:call)
    push!(config.args, :jai_config)
    
    # handle jconfig specific inputs

    # collect jaccel inputs
    _jaccel_clause_handler(config, clauses)

    # exclude jaccel specific inputs

	callsite = sha1(string(__source__.line)*string(__source__.file))
    push!(config.args, vector_to_uint32(callsite))

    return(config)
end



"""
    @jaccel [name][, clauses...]

Create accelerator context.

If `name` is not specified, this context can be accessed only as the currently active context.

# Arguments
- `name`: a unique name for this accelerator context
- `framework`: a list the framework names and compiler-command for the frameworks
- `device`: device number 
- `constant`: list of constant variables to be available in all kernels
- `set`: set several options for the accelerator context including "workdir" for working directory, "debug" for enabling debugging feature.

See also [`@jdecel`](@jdecel), [`@jkernel`](@jkernel)

# Examples
```julia-repl
julia> @jaccel myacc framework(fortran="gfortran -fPIC -shared -g")
AccelInfo
```
"""
macro jaccel(clauses...)

    init = Expr(:call)
    push!(init.args, :jai_accel)

    nclauses = length(clauses)

    # TODO: support stream clause

    # parse accelname
    if nclauses > 0 && clauses[1] isa Symbol
        push!(init.args, string(clauses[1]))
        start_index = 2
    else
        push!(init.args, "")
        start_index = 1
    end

    _jaccel_clause_handler(init, clauses[start_index:end])

	callsite = sha1(string(__source__.line)*string(__source__.file))
    push!(init.args, vector_to_uint32(callsite))

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
    #control = String[]

    # parse directs
    for direct in directs[start_index:end]

        if direct isa Symbol
            push!(config.args, Expr(:call, :(=>), string(direct), true))
            #push!(control, string(direct))

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

        elseif direct.args[1] in (:enable_if, :async)

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

        #elseif direct.args[1] in (:async,)
        #    push!(control, string(direct.args[1]))

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

	callsite = sha1(string(sline)*string(sfile))

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
        insert!(direct.args, 7, vector_to_uint32(callsite))

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
- `name`: a unique name for the accelerator context
- `alloc`: a list of variable names to be allocated in an accelerator memory
- `updateto`: a list of variable names whose content will be copied to an accelerator 

See also [`@jaccel`](@jaccel), [`@jkernel`](@jkernel)

# Examples
```julia-repl
julia> @jenterdata myacc alloc(X, Y, Z) updateto(X, Y, Z)
```
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
- `name`: a unique name for the accelerator context
- `delete`: a list of variable names to be deallocated in an accelerator memory
- `updatefrom`: a list of variable names whose GPU content will be copied to host

See also [`@jaccel`](@jaccel), [`@jkernel`](@jkernel)

# Examples
```julia-repl
julia> @jexitdata myacc delete(X, Y, Z) updatefrom(X, Y, Z)
```
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
- `kerneldef`: Jai kernel definition
- `kernelname`: Kernel context name
- `accelname`: Accelerator context name
- `framework`: a list the framework names and compiler-command for the frameworks

See also [`@jaccel`](@jaccel), [`@jenterdata`](@jenterdata)

# Examples
```julia-repl
julia> @jkernel filepath mykernel myaccel framework(fortran="gfortran -fPIC -shared -g")
```
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

	callsite = sha1(string(__source__.line)*string(__source__.file))
    push!(expr.args, vector_to_uint32(callsite))

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
- `kernelname`: Kernel context name
- `accelname`: Accel context name
- `input`: a list of input variables to the kernel
- `output`: a list of output variables to the kernel
- `hip`: a list of hip launch configurations
- `cuda`: a list of cuda launch configurations

See also [`@jaccel`](@jaccel), [`@jkernel`](@jkernel)

# Examples
```julia-repl
julia> @jlaunch mykernel myaccel input(X, Y) output(Z) hip(threads=((1,1,1), (256,1,1)),enable_if=true)
```
"""
macro jlaunch(clauses...)

    tmp = Expr(:call)
    push!(tmp.args, :jai_launch)
    input = :(())
    output = :(())
    innames = String[]
    outnames = String[]
    config = :(JAI_TYPE_CONFIG())

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

        if clause in (:async,)

            push!(config.args, Expr(:call, :(=>),
                string(clause), true))

        elseif clause.head == :call
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

            elseif clause.args[1] in (:async,)

                push!(config.args, Expr(:call, :(=>),
                    string(clause.args[1]), esc(clause.args[2])))

            else
                error("Wrong jlaunch clause: " * string(clause.args[1]))
            end
        end
    end

    push!(tmp.args, input)
    push!(tmp.args, output)
    push!(tmp.args, innames)
    push!(tmp.args, outnames)
    push!(tmp.args, frames)
    push!(tmp.args, config)

	callsite = sha1(string(__source__.line)*string(__source__.file))
    push!(tmp.args, vector_to_uint32(callsite))

    #dump(tmp)
    return(tmp)

end


"""
    @jwait [accelname]

Wait to finish device operation

If `accelname` is not specified, the currently active accel context will be used.

# Arguments
- `accelname`: Accel context name

See also [`@jaccel`](@jaccel), [`@jkernel`](@jkernel)

# Examples
```julia-repl
julia> @jwait myaccel
```
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

    callsite = sha1(string(__source__.line)*string(__source__.file))
    push!(expr.args, vector_to_uint32(callsite))

    return(expr)
end

"""
    @jdecel [name]

Delete an accelerator context.

If `name` is not specified, this context can be accessed only as the currently active context.

# Arguments
- `name`: a unique name for this accelerator context

See also [`@jaccel`](@jaccel), [`@jkernel`](@jkernel)

# Examples
```julia-repl
julia> @jdecel myacc
```
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
    callsite = sha1(string(__source__.line)*string(__source__.file))
    push!(fini.args, vector_to_uint32(callsite))

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

    callsite = vector_to_uint32(sha1(
					string(__source__.line)*string(__source__.file)))

    push!(diff.args, :jai_diff)
    push!(diff.args, accname)
    push!(diff.args, (Aname, Bname))
    push!(diff.args, callsite)
    push!(block.args, diff)

    push!(diffA.args, :_jai_diffA)
    push!(diffA.args, accname)
    push!(diffA.args, (Aname, Bname))
    push!(diffA.args, callsite)
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
    push!(diffB.args, callsite)
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
    push!(diffend.args, callsite)

    push!(block.args, diffend)

    return(block)
end
