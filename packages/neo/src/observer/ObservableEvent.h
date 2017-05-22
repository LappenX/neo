#ifndef NEO_OBSERVABLEEVENT_H
#define NEO_OBSERVABLEEVENT_H

#include <Common.h>
#include <util/Assert.h>
#include <util/Util.h>

#include <vector>
#include <utility>

template <typename... TArgs>
class Observer;
template <typename... TArgs>
class ObservableEvent;
template <typename... TArgs>
class RaisableObservableEvent;





template <typename... TArgs>
class Observer
{
public:
  Observer()
  {
  }
  
  virtual ~Observer()
  {
    for (ObservableEvent<TArgs...>* observed : m_observed)
    {
      bool found = false;
      auto it = observed->m_observers.begin();
      for (; it != observed->m_observers.end(); ++it)
      {
        if (it->first == this)
        {
          observed->m_observers.erase(it);
          found = true;
          break;
        }
      }
      ASSERT(found, "Observer not found in observed->m_observers");
    }
  }

  void subscribe(ObservableEvent<TArgs...>* observed) const
  {
    m_observed.push_back(observed);
    observed->m_observers.push_back(this);
  }

  virtual void handle(TArgs... args) = 0;

  template <typename... TArgs2>
  friend class ObservableEvent;

private:
  mutable std::vector<ObservableEvent<TArgs...>*> m_observed;

  NO_COPYING(Observer, <TArgs...>)
};

template <typename TFunctor, typename... TArgs>
class FunctorObserver : public Observer<TArgs...>
{
public:
  FunctorObserver(TFunctor functor)
    : m_functor(functor)
  {
  }

  void handle(TArgs... args)
  {
    m_functor(args...);
  }

private:
  TFunctor m_functor;
};

template <typename... TArgs>
class ObservableEvent
{
public:
  ObservableEvent()
  {
  }

  virtual ~ObservableEvent()
  {
    for (auto pair : m_observers)
    {
      auto it = std::find(pair.first->m_observed.begin(), pair.first->m_observed.end(), this);
      ASSERT(it != pair.first->m_observed.end(), "ObservableEvent not found in observer->m_observed");
      pair.first->m_observed.erase(it);
      if (pair.second)
      {
        delete pair.first;
      }
    }
  }

  void subscribe(Observer<TArgs...>* observer)
  {
    this->m_observers.push_back(std::make_pair(observer, false));
    observer->m_observed.push_back(this);
  }

  template <typename TFunctor>
  void operator+=(TFunctor observer)
  {
    this->m_observers.push_back(std::make_pair(new FunctorObserver<TFunctor, TArgs...>(observer), true));
  }

  template <typename... TArgs2>
  friend class Observer;

protected:
  std::vector<std::pair<Observer<TArgs...>*, bool>> m_observers; // TODO: replace with maybe_delete_ptr/ maybe_owns_ptr class

private:
  NO_COPYING(ObservableEvent, <TArgs...>)
};

template <typename... TArgs>
class RaisableObservableEvent : public ObservableEvent<TArgs...>, public Observer<TArgs...>
{
public:
  RaisableObservableEvent()
  {
  }

  void raise(TArgs... args)
  {
    for (auto pair : this->m_observers)
    {
      pair.first->handle(args...);
    }
  }

  void handle(TArgs... args)
  {
    raise(args...);
  }

private:
  NO_COPYING(RaisableObservableEvent, <TArgs...>)
};

#endif // NEO_OBSERVABLEEVENT_H