#ifndef VIEW_GL_RENDERSTEP_H
#define VIEW_GL_RENDERSTEP_H

#include <Common.h>

#include <util/Exception.h>
#include <util/Util.h>

#include <vector>
#include <memory>

namespace gl {

class RenderContext;

class UnRenderStep
{
public:
  virtual void render(RenderContext& context) = 0; // TODO: make constant?
};

class BinRenderStep : public UnRenderStep
{
public:
  virtual void pre(RenderContext& context) = 0;
  virtual void post(RenderContext& context) = 0;

  void render(RenderContext& context)
  {
    pre(context);
    post(context);
  }
};





template <typename TRenderStep1, typename TRenderStep2>
class SequentialPairRenderStep : public UnRenderStep
{
public:
  SequentialPairRenderStep(TRenderStep1 step1, TRenderStep2 step2)
    : m_step1(step1)
    , m_step2(step2)
  {
  }

  void render(RenderContext& context)
  {
    m_step1->render(context);
    m_step2->render(context);
  }

private:
  TRenderStep1 m_step1;
  TRenderStep2 m_step2;
};

template <typename TRenderStep1>
auto then(TRenderStep1 first)
  RETURN_AUTO(first)

template <typename TRenderStep1, typename TRenderStep2, typename... TRenderStepRest>
auto then(TRenderStep1 first, TRenderStep2 second, TRenderStepRest... rest)
  RETURN_AUTO(then(std::make_shared<SequentialPairRenderStep<TRenderStep1, TRenderStep2>>(first, second), rest...))
/*
template <typename TRenderStep>
class SequentialListRenderStep : public UnRenderStep
{
public:
  static_assert(std::is_base_of<UnRenderStep, TRenderStep>::value, "TRenderStep has to inherit from UnRenderStep");

  SequentialListRenderStep()
    : m_children()
  {
  }

  void render(RenderContext& context)
  {
    for (TRenderStep* step : m_children)
    {
      step->pre(context);
      step->post(context);
    }
  }

  void add(TRenderStep* step)
  {
    m_children.push_back(step);
  }

private:
  std::vector<TRenderStep*> m_children;
};
*/




template <typename TRenderStep1, typename TRenderStep2>
class NestedPairRenderStep : public UnRenderStep
{
public:
  NestedPairRenderStep(TRenderStep1 step1, TRenderStep2 step2)
    : m_step1(step1)
    , m_step2(step2)
  {
  }

  void render(RenderContext& context)
  {
    m_step1->pre(context);
    m_step2->render(context);
    m_step1->post(context);
  }

private:
  TRenderStep1 m_step1;
  TRenderStep2 m_step2;
};

template <typename TRenderStep1, typename TRenderStep2>
auto sub(TRenderStep1 first, TRenderStep2 second)
  RETURN_AUTO(std::make_shared<NestedPairRenderStep<TRenderStep1, TRenderStep2>>(first, second))
/*
template <typename TRenderStep>
class NestedListRenderStep
{
public:
  static_assert(std::is_base_of<BinRenderStep, TRenderStep>::value, "TRenderStep has to inherit from BinRenderStep");

  NestedListRenderStep()
    : m_children()
  {
  }

  void pre(RenderContext& context)
  {
    auto it = m_children.begin();
    while (it != m_children.end())
    {
      (*it)->pre(context);
      ++it;
    }
  }

  void post(RenderContext& context)
  {
    auto it = m_children.rbegin();
    while (it != m_children.rend())
    {
      (*it)->post(context);
      ++it;
    }
  }

  void add(TRenderStep* step)
  {
    m_children.push_back(step);
  }

private:
  std::vector<TRenderStep*> m_children;
};
*/
} // end of ns gl

#include "RenderContext.h"

#endif // VIEW_GL_RENDERSTEP_H