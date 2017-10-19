#include "Shader.h"

#include <gl/GlError.h>
#include <gl/core/VertexArrayObject.h>

namespace gl {

ShaderStage::ShaderStage(const std::string& source, GLenum type)
  : m_type(type)
  , m_handle(0)
{
  // Create handle
  m_handle = glCreateShader(m_type);
  GL_CHECK_ERROR("Failed to create handle for " + this->toString());
  if (!m_handle)
  {
    throw GlException("Failed to create handle for " + this->toString());
  }

  // Compile
  const char* source_c_str = source.c_str();
  glShaderSource(m_handle, 1, &source_c_str, NULL);
  glCompileShader(m_handle);
  GL_CHECK_ERROR("Failed to create " + this->toString());
  GLint compile_result;
  glGetShaderiv(m_handle, GL_COMPILE_STATUS, &compile_result);
  if (compile_result == GL_FALSE)
  {
    throw GlException("Failed to compile " + this->toString() + ". Infolog: " + getInfoLog());
  }

  LOG(debug, "gl") << "Created and compiled " << this->toString();
}

ShaderStage::~ShaderStage()
{
  glDeleteShader(m_handle);
  m_handle = 0;
  GL_CHECK_ERROR("Failed to delete handle for " + this->toString());
  LOG(debug, "gl") << "Destroyed " << this->toString();
}

std::string ShaderStage::getInfoLog() const
{
  GLint len;
  glGetShaderiv(m_handle, GL_INFO_LOG_LENGTH, &len);

  char log[len];
  glGetShaderInfoLog(m_handle, len, &len, &log[0]);

  return std::string(&log[0]);
}

std::string ShaderStage::name(GLenum type)
{
  switch(type)
  {
    case GL_COMPUTE_SHADER: return "compute";
    case GL_VERTEX_SHADER: return "vertex";
    case GL_TESS_CONTROL_SHADER: return "tess control";
    case GL_GEOMETRY_SHADER: return "geometry";
    case GL_FRAGMENT_SHADER: return "fragment";
    default: return "unknown";
  }
}





void Shader::init()
{
  VertexArrayObject::unbind(); // TODO: necessary?

  // Create handle
  m_handle = glCreateProgram();
  GL_CHECK_ERROR("Failed to create handle for " + this->toString());
  if (!m_handle)
  {
    throw GlException("Failed to create handle for " + this->toString());
  }

  // Attach shaders
  for (ShaderStage* stage : m_stages)
  {
    glAttachShader(m_handle, stage->m_handle);
    GL_CHECK_ERROR("Failed to attach " + stage->toString() + " to " + this->toString());
  }

  // Bind inputs and outputs
  for (const auto& pair : m_attribute_mapping.getInputs())
  {
    glBindAttribLocation(m_handle, pair.second, pair.first.c_str());
  }
  GL_CHECK_ERROR("Failed to bind attribute locations for " + this->toString());
  for (const auto& pair : m_attribute_mapping.getOutputs())
  {
    glBindFragDataLocation(m_handle, pair.second, pair.first.c_str());
  }
  GL_CHECK_ERROR("Failed to bind fragdata locations for " + this->toString());

  // Link
  glLinkProgram(m_handle);
  GL_CHECK_ERROR("Failed to link " + this->toString());
  glValidateProgram(m_handle);
  GL_CHECK_ERROR("Failed to validate " + this->toString() + " after linking");
  GLint link_status;
  glGetProgramiv(m_handle, GL_LINK_STATUS, &link_status);
  if (link_status == GL_FALSE)
  {
    throw GlException("Failed to link " + this->toString() + ". Infolog: " + getInfoLog());
  }

  LOG(debug, "gl") << "Created and linked " << this->toString();
}

Shader::~Shader()
{
  glDeleteProgram(m_handle);
  m_handle = 0;
  GL_CHECK_ERROR("Failed to delete handle for " + this->toString());
  LOG(debug, "gl") << "Destroyed " << this->toString();
}

void Shader::activate()
{
  glUseProgram(this->m_handle);
  GL_CHECK_ERROR("Failed to activate " + this->toString());
}

void Shader::deactivate()
{
  VertexArrayObject::unbind(); // TODO: necessary? already in vao render func
  glUseProgram(0);
  GL_CHECK_ERROR("Failed to deactivate " + this->toString());
}

std::string Shader::getInfoLog() const
{
  GLint len;
  glGetProgramiv(m_handle, GL_INFO_LOG_LENGTH, &len);

  char log[len];
  glGetProgramInfoLog(m_handle, len, &len, &log[0]);

  return std::string(&log[0]);
}

} // end of ns gl