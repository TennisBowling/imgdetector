cmake_minimum_required(VERSION 3.0.0)

# set(CMAKE_TOOLCHAIN_FILE "C:\\Users\\FASTS\\random\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake")
project(imgdetector VERSION 0.1.0)

# include(CTest)
# enable_testing()
add_executable(imgdetector main.cpp)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(CURL_ZLIB "use curl with zlib" OFF)

# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)
add_subdirectory(boost-cmake)
target_link_libraries(imgdetector PUBLIC Boost::boost)
target_link_libraries(imgdetector PUBLIC Boost::program_options)

add_subdirectory(SQLiteCpp)
target_link_libraries(imgdetector PUBLIC SQLiteCpp sqlite3)

# option(BUILD_opencv_python3 "Build OpenCV Python3 bindings" OFF)
# option(BUILD_opencv_python2 "Build OpenCV Python2 bindings" OFF)
# option(BUILD_JAVA "Build Java bindings" OFF)
# option(BUILD_opencv_js "Build JavaScript bindings" OFF)
find_package(OpenCV REQUIRED)
message("OpenCV version: ${OpenCV_VERSION}")
message("OpenCV location: ${OpenCV_INCLUDE_DIRS}")
target_link_libraries(imgdetector PUBLIC ${OpenCV_LIBS})

include(FetchContent)
FetchContent_Declare(cpr GIT_REPOSITORY https://github.com/libcpr/cpr.git
    GIT_TAG db351ffbbadc6c4e9239daaa26e9aefa9f0ec82d)
FetchContent_MakeAvailable(cpr)
target_link_libraries(imgdetector PRIVATE cpr::cpr)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.10.5/json.tar.xz)
FetchContent_MakeAvailable(json)
target_link_libraries(imgdetector PRIVATE nlohmann_json::nlohmann_json)

# get spdlog through FetchContent
FetchContent_Declare(spdlog GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.10.0)
FetchContent_MakeAvailable(spdlog)
target_link_libraries(imgdetector PRIVATE spdlog::spdlog_header_only)