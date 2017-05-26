#ifndef VIEW_GL_H
#define VIEW_GL_H

#include <Common.h>

// TODO: define own opengl matrix type instead of this hack fix
#if defined(TENSOR_H)
#error "Has to include gl.h before Tensor.h"
#endif

#define DEFAULT_TENSOR_INDEX_STRATEGY ColMajorIndexStrategy
#include <tensor/Tensor.h>

#include "core/BufferObject.h"
#include "core/ClearBuffer.h"
#include "core/Shader.h"
#include "core/VertexArrayObject.h"
#include "core/Viewport.h"

// TODO: #include "parameters/ModelTransformation.h"
#include "parameters/Projection.h"
#include "parameters/Camera.h"

#include "target/GlfwWindow.h"
#include "target/RenderTarget.h"

#include "uniform/Uniform.h"
#include "uniform/RenderStack.h"

#include "GlError.h"
#include "GlLibrary.h"
#include "RenderContext.h"
#include "RenderStep.h"
#include "GlConsts.h"

#endif // VIEW_GL_H