#ifndef VIEW_GL_UNIFORM_UNIFORM_H
#define VIEW_GL_UNIFORM_UNIFORM_H

#include <Common.h>

#include "../core/Shader.h"
#include <util/Property.h>
#include <tensor/Tensor.h>

#include <memory>

namespace gl {

namespace detail {

template <typename T>
class GlUniform;

#define UNIFORM_MACRO(type, ...) \
  template <> \
  class GlUniform<type> \
  { \
  public: \
    static void glUniform(GLint location, const type& value) \
    { \
      __VA_ARGS__; \
      GL_CHECK_ERROR("Failed to set uniform variable in shader"); \
    } \
  }

// https://www.khronos.org/opengles/sdk/docs/man/xhtml/glUniform.xml

UNIFORM_MACRO(tensor::Vector1f, glUniform1fv(location, 1, value.storage().ptr()));
UNIFORM_MACRO(tensor::Vector2f, glUniform2fv(location, 1, value.storage().ptr()));
UNIFORM_MACRO(tensor::Vector3f, glUniform3fv(location, 1, value.storage().ptr()));
UNIFORM_MACRO(tensor::Vector4f, glUniform4fv(location, 1, value.storage().ptr()));

UNIFORM_MACRO(tensor::Matrix1f, glUniform1fv(location, 1, value.storage().ptr()));
UNIFORM_MACRO(tensor::Matrix2f, glUniformMatrix2fv(location, 1, false, value.storage().ptr()));
UNIFORM_MACRO(tensor::Matrix3f, glUniformMatrix3fv(location, 1, false, value.storage().ptr()));
UNIFORM_MACRO(tensor::Matrix4f, glUniformMatrix4fv(location, 1, false, value.storage().ptr()));

} // end of ns detail

template <typename T>
void glUniform(GLint location, const T& value)
{
  detail::GlUniform<T>::glUniform(location, value);
}





template <typename T>
class Uniform
{
public:
  Uniform(Shader* shader, std::string name, T value)
    : m_shader(shader)
    , m_name(name)
    , m_location(-1)
    , m_value(value)
  {
    m_location = glGetUniformLocation(m_shader->m_handle, m_name.c_str());
    if (m_location < 0)
    {
      throw GlException("Failed to acquire location of uniform '" + m_name + "' in " + m_shader->toString());
    }
  }

  void set() const
  {
    gl::glUniform(m_location, m_value.get());
  }

  property::Container<T>& Value()
  {
    return m_value;
  }

private:
  Shader* m_shader;
  std::string m_name;
  GLint m_location;

  property::Container<T> m_value;
};

} // end of ns gl

#endif // VIEW_GL_UNIFORM_UNIFORM_H