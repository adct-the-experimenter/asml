cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG} -Wdelete-non-virtual-dtor")

project( nei_gui LANGUAGES CXX CUDA)
enable_language(CUDA)

find_package(Boost COMPONENTS filesystem regex)
find_package(OpenCV REQUIRED)
find_package(CGAL QUIET COMPONENTS Core )
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(CUDA REQUIRED)

# CUDA specific flags

# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --verbose")

# For some MPs it may be necessary to specify the compute capability of your
# NVIDIA GPU. In that case, simply uncomment the following two lines that start
# with 'set' and change the value of COMPUTE_CAPABILITY to one of the following
# hardware architectures: Tesla-class '11', '12', or '13'; Fermi-class '20'
# or '21'; Kepler-class '30', '32', '35'; or Maxwell-class '50' or '52'
set(COMPUTE_CAPABILITY 60)
# set(CUDA_SEPARABLE_COMPILATION TRUE)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --gpu-architecture;sm_${COMPUTE_CAPABILITY}")
# list(APPEND CUDA_NVCC_FLAGS -arch=sm_35)
enable_language(CUDA)

# Set warning levels for host compilation
if (UNIX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-function -Wno-unused-parameter -Wextra -std=c++11")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unused-function -Wno-unused-parameter -Wextra")
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -Wno-unused-function,-Wno-unused-parameter")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
	# Add debugging to CUDA NVCC flags (for NVidia's NSight tools)
	set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")
endif ()

# set CU FILES
file(GLOB CUDA_FILES "src/" *.cu)
CUDA_COMPILE(CU_O ${CUDA_FILES})

#CPP specific compile

#INCLUDE(${QT_USE_FILE})
include( ${CGAL_USE_FILE} )
include( CGAL_CreateSingleSourceCGALProgram )

# DLP 20230310 Force CXX 14 standard
set(CMAKE_CXX_STANDARD 14)
#include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
#CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
#if(COMPILER_SUPPORTS_CXX14)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#elseif(COMPILER_SUPPORTS_CXX0X)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#else()
#        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
#endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(EXECUTABLE_OUTPUT_PATH ../bin)
file(GLOB_RECURSE objectannotationtool_src "src/*.cpp")

file(GLOB_RECURSE cu_objectannotationtool_src "src/*.cu")

QT5_WRAP_UI(UI_INTERFACE 
			gui/mainwindow.ui 
)

include_directories(/contrib/ece569/libs/include)
link_directories(/contrib/ece569/libs/lib64)

set(CUDA_MANUAL_INCLUDE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${Boost_INCLUDE_DIRS} ${CMAKE_BINARY_DIR} ${CUDA_MANUAL_INCLUDE} src)

#add CUDA (.cu) sources to programs directly in calls to add_library() and add_executable().

set_property(TARGET ${CU_O} 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_executable( nei_gui ${objectannotationtool_src} ${UI_INTERFACE} )
add_subdirectory(src/kernels)

target_link_libraries( nei_gui ${Boost_LIBRARIES} ${OpenCV_LIBS} Qt5::Core
    Qt5::Gui
    Qt5::Widgets 
	kernels)
