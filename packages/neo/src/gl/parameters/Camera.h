#ifndef VIEW_GL_LEVEL1_CAMERA_H
#define VIEW_GL_LEVEL1_CAMERA_H

#include <Common.h>

#include <observer/ObservableProperty.h>

namespace gl {
// TODO: for general types, no inline

inline tensor::Matrix4f lookAt(tensor::Vector3f pos, tensor::Vector3f target, tensor::Vector3f up)
{
  tensor::Vector3f f, s, u;
  f = tensor::normalize(target - pos);
  s = tensor::normalize(tensor::cross(f, up));
  u = tensor::normalize(tensor::cross(s, f));

  tensor::Matrix4f result;
  result = tensor::broadcast<4, 4>(tensor::SingletonT<float>(0));
  result(0, 0) = s(0);
  result(0, 1) = s(1);
  result(0, 2) = s(2);
  result(1, 0) = u(0);
  result(1, 1) = u(1);
  result(1, 2) = u(2);
  result(2, 0) = -f(0);
  result(2, 1) = -f(1);
  result(2, 2) = -f(2);
  result(0, 3) = -tensor::dot(s, pos);
  result(1, 3) = -tensor::dot(u, pos);
  result(2, 3) = tensor::dot(f, pos);
  result(3, 3) = 1;
  return result;
}

template <typename... TInputProperties>
class LookAtCamera : public LazyMappedObservableProperty<LookAtCamera<TInputProperties...>, tensor::Matrix4f, TInputProperties...>
{
public:
  using LazyMappedObservableProperty<LookAtCamera<TInputProperties...>, tensor::Matrix4f, TInputProperties...>::LazyMappedObservableProperty;

  tensor::Matrix4f forward(tensor::Vector3f pos, tensor::Vector3f target, tensor::Vector3f up) const
  {
    return lookAt(pos, target, up);
  }

private:
  NO_COPYING(LookAtCamera, <TInputProperties...>)
};
/*
template <typename... TInputProperties>
class LocRotCamera : public LazyMappedObservableProperty<LocRotCamera<TInputProperties...>, glm::mat4, TInputProperties...>
{
public:
  using LazyMappedObservableProperty<LocRotCamera<TInputProperties...>, glm::mat4, TInputProperties...>::LazyMappedObservableProperty;

  glm::mat4 forward(tensor::Vector3f location, tensor::Vector3f rotation) const
  {
    return glm::translate(pos, target, up);
  }

private:
  NO_COPYING(LocRotCamera, <TInputProperties...>)
};
*/
} // end of ns gl

#endif // VIEW_GL_LEVEL1_CAMERA_H