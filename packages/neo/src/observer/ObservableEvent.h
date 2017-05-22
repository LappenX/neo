#ifndef NEO_OBSERVABLEEVENT_H
#define NEO_OBSERVABLEEVENT_H

#include <Common.h>
#include <util/Assert.h>
#include <util/Util.h>

#include <vector>

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
      auto it = std::find(observed->m_observers.begin(), observed->m_observers.end(), this);
      ASSERT(it != observed->m_observers.end(), "Observer not found in observed->m_observers");
      observed->m_observers.erase(it);
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

template <typename... TArgs>
class ObservableEvent
{
public:
  ObservableEvent()
  {
  }

  virtual ~ObservableEvent()
  {
    for (Observer<TArgs...>* observer : m_observers)
    {
      auto it = std::find(observer->m_observed.begin(), observer->m_observed.end(), this);
      ASSERT(it != observer->m_observed.end(), "ObservableEvent not found in observer->m_observed");
      observer->m_observed.erase(it);
    }
  }

  void subscribe(Observer<TArgs...>* observer)
  {
    this->m_observers.push_back(observer);
    observer->m_observed.push_back(this);
  }

  template <typename... TArgs2>
  friend class Observer;

protected:
  std::vector<Observer<TArgs...>*> m_observers;

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
    for (Observer<TArgs...>* observer : this->m_observers)
    {
      observer->handle(args...);
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