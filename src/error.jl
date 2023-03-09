# fortran.jl: implement functions for Fortran framework

struct JAI_ERROR_NOTIMPLEMENTED_FRAMEWORK <: Exception
    frame ::JAI_TYPE_FRAMEWORK
end
Base.showerror(io::IO, e::JAI_ERROR_NOTIMPLEMENTED_FRAMEWORK) = print(io,
    "Framework-$(extract_name_from_frametype(typeof(e.frame))) " *
    "should implement 'genslib_accel' function.")

struct JAI_ERROR_NOAVAILABLE_FRAMEWORK <: Exception end
Base.showerror(io::IO, e::JAI_ERROR_NOAVAILABLE_FRAMEWORK) = print(io,
    "None of the framework is available on this system.")

struct JAI_ERROR_NOVALID_FRAMEWORK <: Exception end
Base.showerror(io::IO, e::JAI_ERROR_NOVALID_FRAMEWORK) = print(io,
    "None of the framework is selected.")

struct JAI_ERROR_COMPILE_NOSHAREDLIB <: Exception
    compile ::String
    output  ::String
end
Base.showerror(io::IO, e::JAI_ERROR_COMPILE_NOSHAREDLIB) = print(io,
    "Compilation failed:\n" * e.compile * "\n\nOUTPUT:\n" * e.output)

struct JAI_ERROR_NONZERO_RETURN <: Exception
    retval ::Int64
end
Base.showerror(io::IO, e::JAI_ERROR_NONZERO_RETURN) = print(io,
    "Non-zero return value: $(e.retval)")
