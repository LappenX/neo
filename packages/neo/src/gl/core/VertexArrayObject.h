#ifndef VIEW_GL_CORE_VERTEXARRAYOBJECT_H
#define VIEW_GL_CORE_VERTEXARRAYOBJECT_H

#include <Common.h>

#include "../RenderStep.h"
#include "../GlError.h"
#include "../core/BufferObject.h"
#include "../GlConsts.h"

#include <vector>
#include <map>

#include <GL/glew.h>
#include <GL/gl.h>

namespace gl {

class VertexArrayObject;

class AttributeMapping
{
public:
  static const uint32_t NO_ATTRIBUTE = static_cast<uint32_t>(-1);

  void registerInput(std::string name)
  {
    if (m_inputs.count(name) == 0)
    {
      m_inputs.emplace(name, m_inputs.size());
    }
  }

  void registerOutput(std::string name)
  {
    if (m_outputs.count(name) == 0)
    {
      m_outputs.emplace(name, m_outputs.size());
    }
  }

  uint32_t getInputLocation(std::string name) const // TODO: Make sure that return values are checked
  {
    try
    {
      return m_inputs.at(name);
    }
    catch (std::out_of_range e)
    {
      return NO_ATTRIBUTE;  
    }
  }

  uint32_t getOutputLocation(std::string name) const // TODO: Make sure that return values are checked
  {
    try
    {
      return m_outputs.at(name);
    }
    catch (std::out_of_range e)
    {
      return NO_ATTRIBUTE;  
    }
  }

  const std::map<std::string, uint32_t>& getInputs() const
  {
    return m_inputs;
  }

  const std::map<std::string, uint32_t>& getOutputs() const
  {
    return m_outputs;
  }

private:
  std::map<std::string, uint32_t> m_inputs;
  std::map<std::string, uint32_t> m_outputs;
};

class VertexAttribute
{
public:
  /**
   * \brief
   * \param name name of the attribute as specified in attribute mapping
   * \param vertex_size Specifies the number of components per generic vertex attribute. Must be 1, 2, 3, 4.
   * \param el_type Specifies the data type of each component in the array. GL_BYTE, GL_UNSIGNED_BYTE, GL_SHORT, GL_UNSIGNED_SHORT, GL_INT, GL_UNSIGNED_INT, GL_HALF_FLOAT, GL_FLOAT, GL_DOUBLE, GL_FIXED, GL_INT_2_10_10_10_REV, GL_UNSIGNED_INT_2_10_10_10_REV, GL_UNSIGNED_INT_10F_11F_11F_REV
   */
  VertexAttribute(std::string name, BufferObject* data, uint32_t stride, uint32_t offset, GLenum el_type, uint32_t vertex_size)
    : m_name(name)
    , m_data(data)
    , m_stride(stride)
    , m_offset(offset)
    , m_el_type(el_type)
    , m_vertex_size(vertex_size)
  {
  }

  friend class VertexArrayObject;

private:
  std::string m_name;
  BufferObject* m_data;
  uint32_t m_stride;
  uint32_t m_offset;
  GLenum m_el_type;
  uint32_t m_vertex_size;
};

class IndexBufferObject
{
public:
  IndexBufferObject(BufferObject* data)
    : m_data(data)
    , m_index_type(-1)
  {
  }

  template <typename T, size_t TNum>
  void write(boost::array<T, TNum> indices)
  {
    m_data->write(GL_ELEMENT_ARRAY_BUFFER, reinterpret_cast<uint8_t*>(&indices[0]), TNum * sizeof(T));
    m_index_type = gl::type<T>::value;
    ASSERT(m_index_type == GL_UNSIGNED_INT || m_index_type == GL_UNSIGNED_SHORT || m_index_type == GL_UNSIGNED_BYTE, "Invalid index type");
    m_index_size = sizeof(T);
  }

  void bind()
  {
    ASSERT(m_index_type != -1, "Index buffer object contains no data");
    m_data->bind(GL_ELEMENT_ARRAY_BUFFER);
  }

  static void unbind()
  {
    BufferObject::unbind(GL_ELEMENT_ARRAY_BUFFER);
  }

  friend class VertexArrayObject;

private:
  BufferObject* m_data;
  GLenum m_index_type;
  size_t m_index_size;
};

class VertexArrayObject
{
public:
  /*!
   * \brief 
   * \param render_type Specifies what kind of primitives to render. Symbolic constants GL_POINTS, GL_LINE_STRIP, GL_LINE_LOOP, GL_LINES, GL_LINE_STRIP_ADJACENCY, GL_LINES_ADJACENCY, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN, GL_TRIANGLES, GL_TRIANGLE_STRIP_ADJACENCY, GL_TRIANGLES_ADJACENCY and GL_PATCHES are accepted. 
   */
  VertexArrayObject(const AttributeMapping& attribute_mapping, GLenum render_type);
  ~VertexArrayObject();

  void addAttribute(VertexAttribute attrib);

  void bind()
  {
    glBindVertexArray(m_handle);
  }

  static void unbind()
  {
    glBindVertexArray(0);
  }
    
  /*!
   * \brief
   * \param first Specifies the starting index in the enabled arrays.
   * \param num Specifies the number of indices to be rendered.
   */
  void render(size_t first, size_t num);
  void render(size_t first, size_t num, IndexBufferObject* ibo);

private:
  GLuint m_handle;
  const AttributeMapping& m_attribute_mapping;
  GLenum m_render_type;
  size_t m_vertex_num;
};

} // gl

#endif // VIEW_GL_CORE_VERTEXARRAYOBJECT_H