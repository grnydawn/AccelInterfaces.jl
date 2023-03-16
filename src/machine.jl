# machine.jl: implement functions for specific OS and machine support

function set_machine(mach::String)

    for (machname, machdata) in TOML.parsefile(mach)

        skeys = 
        selected = false

        if !(machdata isa Dict{String, Any})
            continue
        end


        if "regex_bindir" in keys(machdata)
            selected = occursin(Regex(machdata["regex_bindir"]), Sys.BINDIR)
        end

        if !selected && "regex_env" in keys(machdata)
            evar, epat = split(machdata["regex_env"], ":")
            selected = occursin(Regex(epat), get(ENV, evar, ""))
        end

        if selected
            d_module    = Vector{String}()
            d_desc      = ""

            for (secname, secdata) in machdata
                if startswith(secname, "regex_")
                    continue

                elseif secname == "module"
                    append!(d_module, secdata)

                elseif secname == "desc"
                    d_desc = secdata
                else
                    println("Unused section: " * secname)
                end
            end

            JAI["machine"] = JAI_TYPE_MACHINE(d_desc, d_module)
        end
    end
end

function get_prerun() :: Vector{Cmd}

    if JAI["machine"] isa JAI_TYPE_MACHINE
        out = Vector{Cmd}()
        for mod in JAI["machine"].modules
            push!(out, `module $(split(mod))`)
        end
        return out
    end

    return ""
end
