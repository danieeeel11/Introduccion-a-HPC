cmake_minimum_required(VERSION 3.5)

project(LinearProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#Se crea una variable para las fuentes
set(project_sources
    main.cpp
    ClassExtraction/extractiondata.cpp
    Regression/linearregression.cpp)
#Se crea una variable para las interfaces
set(project_headers
    ClassExtraction/extractiondata.h
    Regression/linearregression.h)

add_executable(${PROJECT_NAME}
    ${project_sources}
    ${project_headers})

#add_executable(LinearProject main.cpp)

install(TARGETS LinearProject
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
