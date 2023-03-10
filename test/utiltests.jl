import AccelInterfaces as Jai

T = """
begin
{x} = 1
{{{{y}} = 2
{z} = 3
{x} == 1
end
"""
Jai.jaifmt(T, x=1, z=2)
