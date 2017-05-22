#ifndef VIEW_GL_LEVEL1_PROJECTION_H
#define VIEW_GL_LEVEL1_PROJECTION_H

#include <Common.h>

#include <observer/ObservableProperty.h>
#include "../Glm.h"

namespace gl {

template <typename... TInputProperties>
class PerspectiveProjection : public LazyMappedObservableProperty<PerspectiveProjection<TInputProperties...>, glm::mat4, TInputProperties...>
{
public:
  using LazyMappedObservableProperty<PerspectiveProjection<TInputProperties...>, glm::mat4, TInputProperties...>::LazyMappedObservableProperty;

  glm::mat4 forward(float fov, float aspect_ratio, float near, float far) const
  {
    return glm::perspective(fov, aspect_ratio, near, far);
  }

private:
  NO_COPYING(PerspectiveProjection, <TInputProperties...>)
};

template <typename... TInputProperties>
class OrthographicProjection : public LazyMappedObservableProperty<OrthographicProjection<TInputProperties...>, glm::mat4, TInputProperties...>
{
public:
  using LazyMappedObservableProperty<OrthographicProjection<TInputProperties...>, glm::mat4, TInputProperties...>::LazyMappedObservableProperty;

  glm::mat4 forward(float left, float right, float bottom, float top, float near, float far) const
  {
    return glm::ortho(left, right, bottom, top, near, far);
  }

private:
  NO_COPYING(OrthographicProjection, <TInputProperties...>)
};

} // end of ns gl

#endif // VIEW_GL_LEVEL1_PROJECTION_H