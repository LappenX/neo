NEO_BEGIN_TARGET(tuple_test TEST_EXE)
  NEO_ADD_INCLUDE_DIRECTORIES(src)
  NEO_ADD_SOURCES(src/test/TestTuple.cpp)
  NEO_ADD_DEPENDENCIES(Boost)
  SET(NEO_CXX_STANDARD 11)
NEO_END_TARGET()

NEO_BEGIN_TARGET(property_test TEST_EXE)
  NEO_ADD_INCLUDE_DIRECTORIES(src)
  NEO_ADD_SOURCES(src/test/TestProperty.cpp)
  NEO_ADD_DEPENDENCIES(Boost)
  SET(NEO_CXX_STANDARD 11)
NEO_END_TARGET()





NEO_BEGIN_TARGET(tensor_non_cuda_test TEST_EXE)
  NEO_ADD_INCLUDE_DIRECTORIES(src)
  NEO_ADD_SOURCES(src/test/TestTensor.cpp)
  NEO_ADD_DEPENDENCIES(Boost)
  SET(NEO_CXX_STANDARD 11)
NEO_END_TARGET()

NEO_BEGIN_TARGET(tensor_cuda_test CUDA_TEST_EXE)
  NEO_ADD_INCLUDE_DIRECTORIES(src)
  NEO_ADD_SOURCES(src/test/TestTensor.cu src/util/DeviceCout.cu)
  NEO_ADD_DEPENDENCIES(Boost CUDA)
  SET(NEO_CXX_STANDARD 11)
  SET(NEO_CUDA_COMPUTE_CAPABILITY 50)
NEO_END_TARGET()

ADD_DEPENDENCIES(tensor_cuda_test tensor_non_cuda_test)





NEO_BEGIN_TARGET(main_ogl EXE)
  NEO_ADD_INCLUDE_DIRECTORIES(src)
  NEO_ADD_SOURCES(src/main_ogl.cpp src/util/Logging.cpp src/gl/core/VertexArrayObject.cpp src/resource/TextResource.cpp src/gl/core/Shader.cpp
    src/gl/target/GlfwWindow.cpp src/gl/GlError.cpp src/gl/GlLibrary.cpp src/resource/ImageResource.cpp)
  NEO_ADD_DEPENDENCIES(Boost GLFW OPENGL GLEW FreeImage Threads)
  SET(NEO_CXX_STANDARD 11)
NEO_END_TARGET()

# TODO: make macro
ADD_CUSTOM_TARGET(copy-runtime-files ALL COMMAND cmake -E copy_directory ${CMAKE_SOURCE_DIR}/packages/neo/res ${CMAKE_BINARY_DIR}/res)