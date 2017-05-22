#include "VertexArrayObject.h"

namespace gl {

VertexArrayObject::VertexArrayObject(const AttributeMapping& attribute_mapping, GLenum render_type)
    : m_handle(0)
    , m_attribute_mapping(attribute_mapping)
    , m_render_type(render_type)
    , m_vertex_num(0)
{
  glGenVertexArrays(1, &m_handle);
  GL_CHECK_ERROR("Failed to create vertex array object");

  LOG(debug, "gl") << "Created vertex array object";
}

VertexArrayObject::~VertexArrayObject()
{
  glDeleteVertexArrays(1, &m_handle);
  m_handle = 0;
  GL_CHECK_ERROR("Failed to delete vertex array object");
  LOG(debug, "gl") << "Destroyed vertex array object";
}

void VertexArrayObject::addAttribute(VertexAttribute attrib)
{
  bind();
  GLuint attrib_location = m_attribute_mapping.getInputLocation(attrib.m_name);
  if (attrib_location == AttributeMapping::NO_ATTRIBUTE)
  {
    throw GlException(std::string("No attribute found for name '") + attrib.m_name + "'");
  }

  attrib.m_data->bind(GL_ARRAY_BUFFER);
  if (attrib.m_el_type == GL_BYTE || attrib.m_el_type == GL_UNSIGNED_BYTE || attrib.m_el_type == GL_SHORT || attrib.m_el_type == GL_UNSIGNED_SHORT || attrib.m_el_type == GL_INT || attrib.m_el_type == GL_UNSIGNED_INT)
  {
    glVertexAttribIPointer(attrib_location, attrib.m_vertex_size, attrib.m_el_type, attrib.m_stride, reinterpret_cast<const GLvoid*>(attrib.m_offset));
  }
  else
  {
    glVertexAttribPointer(attrib_location, attrib.m_vertex_size, attrib.m_el_type, false, attrib.m_stride, reinterpret_cast<const GLvoid*>(attrib.m_offset));
  }
  glEnableVertexAttribArray(attrib_location);
  unbind();
  GL_CHECK_ERROR("Failed to configure vao attributes");

  LOG(debug, "gl") << "Added attribute to vertex array object";
}

void VertexArrayObject::render(RenderContext& context)
{
  if (!context.active_shader)
  {
    throw GlException("No active shader when rendering VAO");
  }

  // Identity: glm::mat4(1.0f);
  /* TODO: Matrix4f mv = modelMatrices.isEmpty() ? (Matrix4f) new Matrix4f().setIdentity() : modelMatrices.peek();
  vao.getShader().set(Shader.MVP_MATRIX_GLSL_NAME, Matrix4f.mul(projections.peek().getProjectionMatrix(), mv, null));
  vao.getShader().set(Shader.MV_MATRIX_GLSL_NAME, mv);
  vao.getShader().set(Shader.PROJECTION_MATRIX_GLSL_NAME, projections.peek().getProjectionMatrix());*/

  bind();
  glDrawArrays(this->m_render_type, 0, this->m_vertex_num); // TODO: move vertex_num to RenderVAO ?
  GL_CHECK_ERROR("Failed to draw vertex array object");
  unbind();
}

} // end of ns gl