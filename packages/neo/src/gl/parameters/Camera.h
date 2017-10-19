#ifndef VIEW_GL_LEVEL1_CAMERA_H
#define VIEW_GL_LEVEL1_CAMERA_H

#include <Common.h>

#include <observer/ObservableProperty.h>

namespace gl {

template <typename TScalar>
inline tensor::MatrixXXT<TScalar, 4, 4> lookAt(tensor::VectorXT<TScalar, 3> pos, tensor::VectorXT<TScalar, 3> target, tensor::VectorXT<TScalar, 3> up)
{
  tensor::VectorXT<TScalar, 3> f, s, u;
  f = tensor::normalize(target - pos);
  s = tensor::normalize(tensor::cross(f, up)); // TODO: precalculate cross? plus new function?
  u = tensor::normalize(tensor::cross(s, f));

  tensor::MatrixXXT<TScalar, 4, 4> result;
  result = tensor::broadcast<4, 4>(tensor::SingletonT<TScalar>(0));
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

template <bool TEager>
class LookAtCamera : public property::observable::mapped::AutoContainer<LookAtCamera<TEager>, TEager, tensor::Matrix4f, tensor::Vector3f, tensor::Vector3f, tensor::Vector3f>
{
public:
  LookAtCamera(tensor::Vector3f pos, tensor::Vector3f target, tensor::Vector3f up)
    : property::observable::mapped::AutoContainer<LookAtCamera<TEager>, TEager, tensor::Matrix4f, tensor::Vector3f, tensor::Vector3f, tensor::Vector3f>
        (pos, target, up)
  {
  }
  
  tensor::Matrix4f forward(tensor::Vector3f pos, tensor::Vector3f target, tensor::Vector3f up) const
  {
    return lookAt<float>(pos, target, up);
  }

  AUTO_CONTAINER_PROPERTY(tensor::Vector3f, Position, 0)
  AUTO_CONTAINER_PROPERTY(tensor::Vector3f, Target, 1)
  AUTO_CONTAINER_PROPERTY(tensor::Vector3f, Up, 2)

private:
  NO_COPYING(LookAtCamera, <TEager>)
};
/* TODO
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