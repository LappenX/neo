#ifndef VIEW_GL_CORE_SHADER_H
#define VIEW_GL_CORE_SHADER_H

#include <Common.h>

#include <util/Property.h>
#include <util/Tuple.h>
#include "../core/VertexArrayObject.h"
#include "../RenderStep.h"

#include <string>
#include <functional> // hash
#include <vector>
#include <unordered_map>
#include <memory>

#include <GL/glew.h>
#include <GL/gl.h>

namespace gl {

class ShaderStage
{
public:
  ShaderStage(const std::string& source, GLenum type);
  ~ShaderStage();

  static std::string name(GLenum type);

  std::string getInfoLog() const;

  std::string toString() const
  {
    return ShaderStage::name(m_type) + " shader stage";
  }

  friend class Shader;

private:
  GLenum m_type;
  GLuint m_handle;
};



template <typename T>
class Uniform;
class AttributeMapping;

class Shader
{
public:
  template <typename... TShaderStageTypes>
  Shader(const AttributeMapping& attribute_mapping, TShaderStageTypes*... stages)
    : m_handle(0)
    , m_stages{stages...}
    , m_attribute_mapping(attribute_mapping)
  {
    init();
  }

  ~Shader();

  std::string getInfoLog() const;

  std::string toString() const
  {
    return "shader program";
  }

  void pre(RenderContext& context);
  void post(RenderContext& context);

  template <typename T>
  friend class Uniform;

private:
  GLuint m_handle;
  std::vector<ShaderStage*> m_stages;
  const AttributeMapping& m_attribute_mapping;

  void init();
};

} // end of ns gl

#endif // VIEW_GL_CORE_SHADER_H