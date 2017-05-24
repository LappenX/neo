#ifndef VIEW_GL_UNIFORM_UNIFORM_H
#define VIEW_GL_UNIFORM_UNIFORM_H

#include <Common.h>

#include "../core/Shader.h"
#include "../RenderStep.h"
#include "../Glm.h"
#include <util/Property.h>

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
UNIFORM_MACRO(float, glUniform1f(location, value));
UNIFORM_MACRO(glm::vec2, glUniform2fv(location, 1, &value[0]));
UNIFORM_MACRO(glm::vec3, glUniform3fv(location, 1, &value[0]));
UNIFORM_MACRO(glm::vec4, glUniform4fv(location, 1, &value[0]));
UNIFORM_MACRO(glm::mat2, glUniformMatrix2fv(location, 1, GL_FALSE, &value[0][0]));
UNIFORM_MACRO(glm::mat3, glUniformMatrix3fv(location, 1, GL_FALSE, &value[0][0]));
UNIFORM_MACRO(glm::mat4, glUniformMatrix4fv(location, 1, GL_FALSE, &value[0][0]));
UNIFORM_MACRO(GLint, glUniform1i(location, value));
UNIFORM_MACRO(glm::ivec2, glUniform2iv(location, 1, &value[0]));
UNIFORM_MACRO(glm::ivec3, glUniform3iv(location, 1, &value[0]));
UNIFORM_MACRO(glm::ivec4, glUniform4iv(location, 1, &value[0]));

} // end of ns detail

template <typename T>
void glUniform(GLint location, const T& value)
{
  detail::GlUniform<T>::glUniform(location, value);
}





template <typename T, typename TPropertyCRTP = void>
class UniformSetter;

template <typename T>
class Uniform
{
public:
  Uniform(Shader* shader, std::string name)
    : m_shader(shader)
    , m_name(name)
    , m_location(-1)
  {
    m_location = glGetUniformLocation(m_shader->m_handle, m_name.c_str());
    if (m_location < 0)
    {
      throw GlException("Failed to acquire location of uniform '" + m_name + "' in " + m_shader->toString());
    }
  }

  void glUniform(const T& value)
  {
    gl::glUniform(m_location, value);
  }

  template <typename TProperty>
  UniformSetter<T, TProperty> makeSetStep(const TProperty* property);

private:
  Shader* m_shader;
  std::string m_name;
  GLint m_location;
};

template <typename T, typename TProperty>
class UniformSetter : public UnRenderStep
{
public:
  UniformSetter(Uniform<T>* uniform, const TProperty* property)
    : m_uniform(uniform)
    , m_property(property)
  {
  }

  void render(RenderContext& context)
  { // TODO: specialize subclass for T = const T&, single line for this:
    T prop = m_property->get();
    m_uniform->glUniform(prop);
  }

private:
  Uniform<T>* m_uniform;
  const TProperty* m_property;
};

template <typename T>
template <typename TProperty>
UniformSetter<T, TProperty> Uniform<T>::makeSetStep(const TProperty* property)
{
  return UniformSetter<T, TProperty>(this, property);
}

} // gl

#endif // VIEW_GL_UNIFORM_UNIFORM_H