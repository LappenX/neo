#ifndef VIEW_GL_CORE_BUFFEROBJECT_H
#define VIEW_GL_CORE_BUFFEROBJECT_H

#include <Common.h>

#include "../GlError.h"
#include <util/Logging.h>

#include <GL/glew.h>
#include <GL/gl.h>

namespace gl {

class BufferObject
{
public:
  /*!
   * \brief 
   * \param usage Specifies the expected usage pattern of the data store. The symbolic constant must be GL_STREAM_DRAW, GL_STREAM_READ, GL_STREAM_COPY, GL_STATIC_DRAW, GL_STATIC_READ, GL_STATIC_COPY, GL_DYNAMIC_DRAW, GL_DYNAMIC_READ, or GL_DYNAMIC_COPY
   */
  BufferObject(GLenum usage)
    : m_usage(usage)
    , m_handle(0)
  {
    glGenBuffers(1, &m_handle);
    GL_CHECK_ERROR("Failed to generate buffer object");
    LOG(debug, "gl") << "Created buffer object";
  }

  ~BufferObject()
  {
    glDeleteBuffers(1, &m_handle);
    m_handle = 0;
    GL_CHECK_ERROR("Failed to delete buffer object");
    LOG(debug, "gl") << "Destroyed buffer object";
  }

  bool isInitialized() const
  {
    return m_handle;
  }

  /*!
   * \brief 
   * \param target Specifies the target to which the buffer object is bound.
   */
  void bind(GLenum target)
  {
    glBindBuffer(target, m_handle);
    GL_CHECK_ERROR("Failed to bind buffer object");
  }

  static void unbind(GLenum target)
  {
    glBindBuffer(target, 0);
  }

  void write(GLenum target, uint8_t* data, uint32_t length)
  {
    // TODO: GL_VERSION_4_5 glBufferDataNamed(m_handle, length, data, m_usage);
    bind(target);
    glBufferData(target, length, data, m_usage);
    unbind(target);
    GL_CHECK_ERROR("Failed to write data from host memory to device buffer object");
  }

private:
  GLenum m_usage;
  GLuint m_handle;
};

} // gl

#endif // VIEW_GL_CORE_BUFFEROBJECT_H