#ifndef VIEW_GL_LEVEL1_CAMERA_H
#define VIEW_GL_LEVEL1_CAMERA_H

#include <Common.h>

#include <observer/ObservableProperty.h>
#include "../Glm.h"

namespace gl {

template <typename... TInputProperties>
class LookAtCamera : public LazyMappedObservableProperty<LookAtCamera<TInputProperties...>, glm::mat4, TInputProperties...>
{
public:
  using LazyMappedObservableProperty<LookAtCamera<TInputProperties...>, glm::mat4, TInputProperties...>::LazyMappedObservableProperty;

  glm::mat4 forward(glm::vec3 pos, glm::vec3 target, glm::vec3 up) const
  {
    return glm::lookAt(pos, target, up);
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

  glm::mat4 forward(glm::vec3 location, glm::vec3 rotation) const
  {
    return glm::translate(pos, target, up);
  }

private:
  NO_COPYING(LocRotCamera, <TInputProperties...>)
};
*/
} // end of ns gl

#endif // VIEW_GL_LEVEL1_CAMERA_H