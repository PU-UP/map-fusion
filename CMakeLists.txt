cmake_minimum_required(VERSION 3.5)
project(slam_architect_ws)

if(BUILD_IN_SLAM_CORE)
    add_subdirectory(./map_fusion)
    add_library(map_fusion_component SHARED map_fusion_component.cpp)
    target_link_libraries(map_fusion_component PUBLIC 
        slam_common
        ${MAP_FUSION_LIBS}
    )
else()
    add_subdirectory(./map_fusion)

endif()
