#ifndef VIEW_GL_CONSTS_H
#define VIEW_GL_CONSTS_H

#include <Common.h>

#include <GL/gl.h>

namespace gl {

template <typename T>
struct type;

template <>
struct type<uint8_t>
{
  static const GLenum value = GL_UNSIGNED_BYTE;
};

template <>
struct type<uint16_t>
{
  static const GLenum value = GL_UNSIGNED_SHORT;
};

template <>
struct type<uint32_t>
{
  static const GLenum value = GL_UNSIGNED_INT;
};

template <>
struct type<int8_t>
{
  static const GLenum value = GL_BYTE;
};

template <>
struct type<int16_t>
{
  static const GLenum value = GL_SHORT;
};

template <>
struct type<int32_t>
{
  static const GLenum value = GL_INT;
};

template <>
struct type<float>
{
  static const GLenum value = GL_FLOAT;
};

template <>
struct type<double>
{
  static const GLenum value = GL_DOUBLE;
};

} // end of ns gl

#endif // VIEW_GL_CONSTS_H