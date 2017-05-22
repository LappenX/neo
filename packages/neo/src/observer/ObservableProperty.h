#ifndef NEO_OBSERVABLE_PROPERTY_H
#define NEO_OBSERVABLE_PROPERTY_H

#include <Common.h>

#include <observer/ObservableEvent.h>
#include <util/Tuple.h>
#include <util/Property.h>
#include <tmp/ValueSequence.h>

class HasEventChanged
{
public:
  HasEventChanged()
  {
  }

  virtual ~HasEventChanged()
  {
  }

  ObservableEvent<>& getEventChanged() const
  {
    return m_event_changed;
  }

protected:
  mutable RaisableObservableEvent<> m_event_changed;

private:
  NO_COPYING(HasEventChanged)
};

template <typename T>
class ObservableProperty : public HasEventChanged, public virtual Property<T>
{
public:
  virtual ~ObservableProperty()
  {
  }
};

template <typename T>
class ObservableMutableProperty : public ObservableProperty<T>, public MutableProperty<T>
{
public:
  virtual ~ObservableMutableProperty()
  {
  }
};





template <typename T>
class SimpleObservableProperty : public ObservableMutableProperty<T>
{
public:
  SimpleObservableProperty(T value)
    : m_value(value)
  {
  }

  T get() const
  {
    return m_value;
  }

  void set(const T& value)
  {
    m_value = value;
    this->m_event_changed.raise(); // TODO: Check if value was actually changed
  }

private:
  T m_value;

  NO_COPYING(SimpleObservableProperty, <T>)
};





template <typename CRTP, typename TO, typename... TInputProperties>
class LazyMappedObservableProperty : public ObservableProperty<TO>, private Observer<>
{
public:
  LazyMappedObservableProperty(TInputProperties*... input) // TODO: initialize validated?
    : m_input(input...)
    , m_cache()
    , m_validated(false)
  {
    tuple::for_each([this](const HasEventChanged* in){
      in->getEventChanged().subscribe(this);
    }, input...);
  }

  void validate() const
  {
    if (!m_validated)
    {
      m_cache = callForward(tmp::value_sequence::ascending_numbers_t<sizeof...(TInputProperties)>());
      m_validated = true;
    }
  }

  void invalidate() const
  {
    m_validated = false;
  }

  TO get() const
  {
    validate();
    return m_cache;
  }

private:
  void handle()
  {
    invalidate();
    this->m_event_changed.raise(); // TODO: Cannot check if value was actually changed? Note in documentation?
  }

  template <size_t... TIndices>
  TO callForward(tmp::value_sequence::IndexSequence<TIndices...>) const
  {
    return static_cast<const CRTP*>(this)->forward((m_input.template get<TIndices>()->get())...);
  }

  const tuple::Tuple<TInputProperties*...> m_input;
  mutable TO m_cache;
  mutable bool m_validated;

  NO_COPYING(LazyMappedObservableProperty, <CRTP, TO, TInputProperties...>)
};

template <typename TFunction, typename... TInputProperties>
class StaticFunctionLazyMappedObservableProperty : public LazyMappedObservableProperty<StaticFunctionLazyMappedObservableProperty<TFunction, TInputProperties...>,
                                                           decltype(std::declval<TFunction>()(std::declval<TInputProperties>().get()...)),
                                                           TInputProperties...>
{
public:
  using LazyMappedObservableProperty<StaticFunctionLazyMappedObservableProperty<TFunction, TInputProperties...>,
                                                           decltype(std::declval<TFunction>()(std::declval<TInputProperties>().get()...)),
                                                           TInputProperties...>::LazyMappedObservableProperty;
  
  template <typename... TI>
  auto forward(TI&&... input) const
  RETURN_AUTO(TFunction()(util::forward<TI>(input)...))
};

#endif // NEO_OBSERVABLE_PROPERTY_H