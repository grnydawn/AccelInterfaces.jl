# fortran.jl: implement functions for Fortran framework

struct JAI_ERROR_NOTIMPLEMENTED_FRAMEWORK <: Exception
    frame ::JAI_TYPE_FRAMEWORK
end
Base.showerror(io::IO, e::JAI_ERROR_NOTIMPLEMENTED_FRAMEWORK) = print(io,
    "ERROR: framework-$(extract_name_from_frametype(typeof(e.frame))) " *
    "should implement 'genslib_accel' function.")

struct JAI_ERROR_NOAVAILABLE_FRAMEWORK <: Exception end
Base.showerror(io::IO, e::JAI_ERROR_NOAVAILABLE_FRAMEWORK) = print(io,
    "ERROR: none of the framework is available on this system.")

