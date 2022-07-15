using Documenter
using AccelInterfaces

makedocs(
    sitename = "AccelInterfaces",
    format = Documenter.HTML(),
    modules = [AccelInterfaces]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
