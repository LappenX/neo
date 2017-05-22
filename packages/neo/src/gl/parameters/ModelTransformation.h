#ifndef VIEW_GL_LEVEL1_MODELTRANSFORMATION_H
#define VIEW_GL_LEVEL1_MODELTRANSFORMATION_H

#include <Common.h>

#include "../RenderStep.h"
#include "../Glm.h"
#include <observer/ObservableProperty.h>

namespace gl {

template <typename CRTP>
class ModelTransformation : public PushPopRenderStep<ModelTransformation<CRTP>>
{
public:
  glm::mat4 pushEmpty()
  {
    return static_cast<CRTP&>(*this);
  }

  glm::mat4 push(glm::mat4 top)
  {
    return top * static_cast<CRTP&>(*this);
  }

  auto& stack(RenderContext& context)
  {
    return context.model_matrices;
  }

private:
  NO_COPYING(ModelTransformation, <CRTP>)
};

class Translation : public ModelTransformation<Translation>, protected LazyMappedObservableValue<glm::mat4, glm::vec3>
{
public:
  using LazyMappedObservableValue<glm::mat4, glm::vec3>::LazyMappedObservableValue;

  glm::mat4 forward(glm::vec3 translation) const
  {
    return glm::translate(translation);
  }

private:
  NO_COPYING(Translation)
};

class Rotation : public ModelTransformation<Rotation>, protected LazyMappedObservableValue<glm::mat4, float, glm::vec3>
{
public:
  using LazyMappedObservableValue<glm::mat4, float, glm::vec3>::LazyMappedObservableValue;

  glm::mat4 forward(float angle, glm::vec3 axis) const
  {
    return glm::rotate(angle, axis);
  }

private:
  NO_COPYING(Rotation)
};

using SizeMode = glm::vec3;

const SizeMode TOP_LEFT_2 =     glm::vec3(0.0f, 1.0f, 0.0f);
const SizeMode BOTTOM_LEFT_2 =  glm::vec3(0.0f, 0.0f, 0.0f);
const SizeMode CENTER_2 =       glm::vec3(0.5f, 0.5f, 0.0f);

class Size :  public ModelTransformation<Size>,
              protected LazyMappedObservableValue<glm::mat4, glm::vec3>
{
public:
  using LazyMappedObservableValue<glm::mat4, glm::vec3>::LazyMappedObservableValue;

  glm::mat4 forward(glm::vec3 size) const
  {
    return elwiseMul(m_self_mode - m_target_mode, size);
  }

private:
  NO_COPYING(Size)

  SizeMode m_self_mode;
  SizeMode m_target_mode;
};

} // gl

#endif // VIEW_GL_LEVEL1_MODELTRANSFORMATION_H