# First, install MSVC with the optional Clang and UCRT features
# Then you need to have sundials 6.6.1 and tbb >= 2021 installed.
# I used Conda, hence the below path as C:\ProgramData\Miniconda3\envs\clangtest\Library .

$cxxflags = '-std=c++1y -O3  -D_REENTRANT -DBOOST_DISABLE_ASSERTS -D_BOOST_LGAMMA -DTBB_INTERFACE_NEW -Wno-sign-compare -Wno-deprecated-builtins -Wno-ignored-attributes'
$cppflags = '-I C:\ProgramData\Miniconda3\envs\clangtest\Library\include\ -I src -I stan/src -I stan/lib/rapidjson_1.1.0/ -I lib/CLI11-1.9.1/ -I stan/lib/stan_math/ -I stan/lib/stan_math/lib/eigen_3.3.9 -I stan/lib/stan_math/lib/boost_1.78.0'

$linkflags = '-Wl",/LIBPATH:C:\ProgramData\Miniconda3\envs\clangtest\Library\lib\" -ltbb -lsundials_nvecserial -lsundials_cvodes -lsundials_idas -lsundials_kinsol'


Invoke-Expression "clang++ $cppflags $cxxflags -c -o src/bridgestan.o src/bridgestan.cpp"

.\bin\stanc.exe  --o=test_models/multi/multi.cpp test_models/multi/multi.stan

Invoke-Expression "clang++ $cppflags $cxxflags -c -o test_models/multi/multi.o test_models/multi/multi.cpp"

Invoke-Expression "clang++ $cppflags $cxxflags -shared test_models/multi/multi.o src/bridgestan.o $linkflags -o test_models/multi/multi_model.so"

