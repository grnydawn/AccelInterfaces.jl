# compiler.jl: implement functions for compiler support

struct JAI_TYPE_COMPILER
    lang            ::Char
    path            ::String
    opt_debug       ::String
    opt_sharedlib   ::String
    opt_frame       ::Dict{JAI_TYPE_FRAMEWORK, String}
    opt_append      ::String
end

const JAI_AVAILABLE_COMPILERS = OrderedDict{String, JAI_TYPE_COMPILER}()

function get_compiles(
        frame   ::JAI_TYPE_FRAMEWORK,
        compiler::JAI_TYPE_CONFIG
    )

end
