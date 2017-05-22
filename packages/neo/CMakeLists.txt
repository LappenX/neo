
NEO_BEGIN_TARGET(tuple_test TEST_EXE)
  NEO_ADD_INCLUDE_DIRECTORIES(src)
  NEO_ADD_SOURCES(src/test/TestTuple.cpp src/util/Assert.cpp)
  NEO_ADD_DEPENDENCIES(Boost)
  SET(NEO_CXX_STANDARD 11)
NEO_END_TARGET()

NEO_BEGIN_TARGET(property_test TEST_EXE)
  NEO_ADD_INCLUDE_DIRECTORIES(src)
  NEO_ADD_SOURCES(src/test/TestProperty.cpp src/util/Assert.cpp)
  NEO_ADD_DEPENDENCIES(Boost)
  SET(NEO_CXX_STANDARD 11)
NEO_END_TARGET()

NEO_BEGIN_TARGET(main EXE)
  NEO_ADD_INCLUDE_DIRECTORIES(src)
  NEO_ADD_SOURCES(src/main.cpp src/util/Assert.cpp src/util/Logging.cpp src/util/Math.cpp
    src/gl/core/VertexArrayObject.cpp src/resource/TextResource.cpp src/gl/core/Shader.cpp
    src/gl/target/GlfwWindow.cpp src/gl/GlError.cpp src/gl/GlLibrary.cpp src/gl/Glm.cpp)
  NEO_ADD_DEPENDENCIES(Boost GLFW GLM OPENGL GLEW FreeImage Threads)
  SET(NEO_CXX_STANDARD 11)
NEO_END_TARGET()

# TODO: make macro
ADD_CUSTOM_TARGET(copy-runtime-files ALL COMMAND cmake -E copy_directory ${CMAKE_SOURCE_DIR}/packages/neo/res ${CMAKE_BINARY_DIR}/res)