#ifndef VIEW_GL_LEVEL1_PROJECTION_H
#define VIEW_GL_LEVEL1_PROJECTION_H

#include <Common.h>

#include <observer/ObservableProperty.h>

namespace gl {
// TODO: for general types, no inline
inline tensor::Matrix4f perspective(float fov, float aspect_ratio, float near, float far)
{
  ASSERT(aspect_ratio != 0, "Aspect ratio cannot be zero");
  ASSERT(near != far, "Near and far values cannot be equal");

  float tan_half_fov = math::tan(fov / 2);

  tensor::Matrix4f result;
  result = tensor::broadcast<4, 4>(tensor::SingletonT<float>(0));
  result(0, 0) = 1.0f / (aspect_ratio * tan_half_fov);
  result(1, 1) = 1.0f / (tan_half_fov);
  result(2, 2) = -(far + near) / (far - near);
  result(3, 2) = -1;
  result(2, 3) = -2 * far * near / (far - near);
  return result;
}

template <typename... TInputProperties>
class PerspectiveProjection : public LazyMappedObservableProperty<PerspectiveProjection<TInputProperties...>, tensor::Matrix4f, TInputProperties...>
{
public:
  using LazyMappedObservableProperty<PerspectiveProjection<TInputProperties...>, tensor::Matrix4f, TInputProperties...>::LazyMappedObservableProperty;

  tensor::Matrix4f forward(float fov, float aspect_ratio, float near, float far) const
  {
    return perspective(fov, aspect_ratio, near, far);
  }

private:
  NO_COPYING(PerspectiveProjection, <TInputProperties...>)
};
/* TODO:
template <typename... TInputProperties>
class OrthographicProjection : public LazyMappedObservableProperty<OrthographicProjection<TInputProperties...>, tensor::Matrix4f, TInputProperties...>
{
public:
  using LazyMappedObservableProperty<OrthographicProjection<TInputProperties...>, tensor::Matrix4f, TInputProperties...>::LazyMappedObservableProperty;

  tensor::Matrix4f forward(float left, float right, float bottom, float top, float near, float far) const
  {
    return glm::ortho(left, right, bottom, top, near, far);
  }

private:
  NO_COPYING(OrthographicProjection, <TInputProperties...>)
};*/

} // end of ns gl

#endif // VIEW_GL_LEVEL1_PROJECTION_H