#ifndef VIEW_GL_PROJECTION_H
#define VIEW_GL_PROJECTION_H

#include <Common.h>

#include <observer/ObservableProperty.h>

namespace gl {

template <typename TScalar>
tensor::MatrixXXT<TScalar, 4, 4> perspective(TScalar fov, TScalar aspect_ratio, TScalar near, TScalar far)
{
  ASSERT(aspect_ratio != 0, "Aspect ratio cannot be zero");
  ASSERT(near != far, "Near and far values cannot be equal");

  TScalar tan_half_fov = math::tan(fov / 2);

  tensor::MatrixXXT<TScalar, 4, 4> result;
  result = tensor::broadcast<4, 4>(tensor::SingletonT<TScalar>(0));
  result(0, 0) = static_cast<TScalar>(1) / (aspect_ratio * tan_half_fov);
  result(1, 1) = static_cast<TScalar>(1) / (tan_half_fov);
  result(2, 2) = -(far + near) / (far - near);
  result(3, 2) = -1;
  result(2, 3) = -2 * far * near / (far - near);
  return result;
}

template <bool TEager>
class PerspectiveProjection : public property::observable::mapped::AutoContainer<PerspectiveProjection<TEager>, TEager, tensor::Matrix4f, float, float, float, float>
{
public:
  PerspectiveProjection(float fov, float aspect_ratio, float near, float far)
    : property::observable::mapped::AutoContainer<PerspectiveProjection<TEager>, TEager, tensor::Matrix4f, float, float, float, float>::AutoContainer
        (fov, aspect_ratio, near, far)
  {
  }

  tensor::Matrix4f forward(float fov, float aspect_ratio, float near, float far) const
  {
    return perspective<float>(fov, aspect_ratio, near, far);
  }

  AUTO_CONTAINER_PROPERTY(float, Fov, 0)
  AUTO_CONTAINER_PROPERTY(float, AspectRatio, 1)
  AUTO_CONTAINER_PROPERTY(float, Near, 2)
  AUTO_CONTAINER_PROPERTY(float, Far, 3)

private:
  NO_COPYING(PerspectiveProjection, <TEager>)
};





template <typename TScalar>
tensor::MatrixXXT<TScalar, 4, 4> orthographic(TScalar left, TScalar right, TScalar bottom, TScalar top, TScalar near, TScalar far)
{
  ASSERT(left != right, "Left and right values cannot be equal");
  ASSERT(bottom != top, "Bottom and top values cannot be equal");
  ASSERT(near != far, "Near and far values cannot be equal");

  tensor::MatrixXXT<TScalar, 4, 4> result;
  result = tensor::broadcast<4, 4>(tensor::SingletonT<TScalar>(0));
  result(0, 0) = 2.0f / (right - left);
  result(1, 1) = 2.0f / (top - bottom);
  result(2, 2) = 2.0f / (far - near);
  result(0, 3) = -(right + left) / (right - left);
  result(1, 3) = -(top + bottom) / (top - bottom);
  result(2, 3) = -(far + near) / (far - near);
  result(3, 3) = 1.0f;
  return result;
}

template <bool TEager>
class OrthographicProjection : public property::observable::mapped::AutoContainer<OrthographicProjection<TEager>, TEager, tensor::Matrix4f, float, float, float, float, float, float>
{
public:
  OrthographicProjection(float left, float right, float bottom, float top, float near, float far)
    : property::observable::mapped::AutoContainer<OrthographicProjection<TEager>, TEager, tensor::Matrix4f, float, float, float, float, float, float>::AutoContainer
        (left, right, bottom, top, near, far)
  {
  }

  tensor::Matrix4f forward(float left, float right, float bottom, float top, float near, float far) const
  {
    return orthographic<float>(left, right, bottom, top, near, far);
  }

  AUTO_CONTAINER_PROPERTY(float, Left, 0)
  AUTO_CONTAINER_PROPERTY(float, Right, 1)
  AUTO_CONTAINER_PROPERTY(float, Bottom, 2)
  AUTO_CONTAINER_PROPERTY(float, Top, 3)
  AUTO_CONTAINER_PROPERTY(float, Near, 4)
  AUTO_CONTAINER_PROPERTY(float, Far, 5)

private:
  NO_COPYING(OrthographicProjection, <TEager>)
};

} // end of ns gl

#endif // VIEW_GL_PROJECTION_H