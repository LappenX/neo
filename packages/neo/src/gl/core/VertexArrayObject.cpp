#include "VertexArrayObject.h"

namespace gl {

VertexArrayObject::VertexArrayObject(const AttributeMapping& attribute_mapping)
    : m_handle(0)
    , m_attribute_mapping(attribute_mapping)
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

void VertexArrayObject::render(GLenum render_type, size_t first, size_t num)
{
  bind();
  glDrawArrays(render_type, first, num);
  GL_CHECK_ERROR("Failed to draw vertex array object");
  unbind();
}

void VertexArrayObject::render(GLenum render_type, size_t first, size_t num, IndexBufferObject* ibo)
{
  this->bind();
  ibo->bind();
  glDrawElements(render_type, num, ibo->m_index_type, reinterpret_cast<void*>(ibo->m_index_size * first));
  GL_CHECK_ERROR("Failed to draw vertex array object");
  ibo->unbind();
  this->unbind();
}

} // end of ns gl