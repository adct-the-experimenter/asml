enable_language(CUDA)
add_library(kernels
kernel.cu
kernel.cuh
)

target_compile_features(kernels PUBLIC cxx_std_11)
set_target_properties(kernels
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(kernels)
