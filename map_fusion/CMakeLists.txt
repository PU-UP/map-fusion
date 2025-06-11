cmake_minimum_required(VERSION 3.5)

# 设置项目名称和版本
project(map_fusion VERSION 1.0)

# 添加源文件
set(SOURCES
    map_fusion.cpp
    grid_map.cpp
    finished_submap.cpp
    local_map.cpp
)


# 添加库
add_library(${PROJECT_NAME} SHARED ${SOURCES})
set(MAP_FUSION_LIBS ${MAP_FUSION_LIBS} ${PROJECT_NAME} CACHE INTERNAL "list of all needed algorithm libraries" FORCE)
target_include_directories(${PROJECT_NAME} PUBLIC
    ./map_fusion
)
target_link_libraries(
  ${PROJECT_NAME} PUBLIC
  ${OpenCV_LIBS} 
  ${CERES_LIBRARIES} 
)
