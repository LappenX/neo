#ifndef VIEW_GL_H
#define VIEW_GL_H

#include <Common.h>

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
#include "Glm.h"
#include "RenderContext.h"
#include "RenderStep.h"

#endif // VIEW_GL_H