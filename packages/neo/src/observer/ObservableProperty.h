#ifndef NEO_OBSERVABLE_PROPERTY_H
#define NEO_OBSERVABLE_PROPERTY_H

#include <Common.h>

#include <observer/ObservableEvent.h>
#include <util/Tuple.h>
#include <util/Property.h>
#include <tmp/ValueSequence.h>



namespace property {

namespace observable {

class HasEventChanged
{
public:
  HasEventChanged()
  {
  }

  virtual ~HasEventChanged()
  {
  }

  ObservableEvent<>& EventChanged() const
  {
    return m_event_changed;
  }

protected:
  mutable RaisableObservableEvent<> m_event_changed;

private:
  NO_COPYING(HasEventChanged)
};

template <typename T>
class Property : public HasEventChanged, public virtual property::Property<T>
{
public:
  virtual ~Property()
  {
  }
};

template <typename T>
class MutableProperty : public observable::Property<T>, public property::MutableProperty<T>
{
public:
  virtual ~MutableProperty()
  {
  }
};

template <typename T>
class Simple : public observable::MutableProperty<T>
{
public:
  Simple(T value)
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
    this->m_event_changed.raise(); // TODO: Check if value was actually changed, or note in docs
  }

private:
  T m_value;

  NO_COPYING(Simple, <T>)
};

template <typename T>
class Container : public observable::Property<T>, private Observer<>
{
public:
  template <typename T2>
  Container(Property<T2>* contained)
    : m_contained(lambda_as_unique_ptr<T>([contained](){return contained->get();}))
  {
    contained->EventChanged().subscribe(this);
  }

  Container(const T& value)
    : m_contained(util::make_unique<Simple<T>>(value))
  {
  }

  T get() const
  {
    return m_contained->get();
  }

  template <typename T2>
  void refer_to(Property<T2>* contained)
  {
    m_contained->EventChanged().unsubscribe(this);
    m_contained = lambda_as_unique_ptr<T>([contained](){return contained->get();});
    contained->EventChanged().subscribe(this);
  }

  void refer_to(const T& value)
  {
    m_contained->EventChanged().unsubscribe(this);
    m_contained = util::make_unique<Simple<T>>(value);
  }

private:
  void handle()
  {
    this->m_event_changed.raise();
  }

  std::unique_ptr<Property<T>> m_contained;
};





namespace mapped {

#define MAP_EAGER true
#define MAP_LAZY false

template <typename CRTP, typename TO, typename... TInputProperties>
class Eager : public observable::Property<TO>, private Observer<>
{
public:
  Eager(TInputProperties*... input)
    : m_input(input...)
  {
    tuple::for_each([this](const HasEventChanged* in){
      in->EventChanged().subscribe(this);
    }, input...);
  }

  TO get() const
  {
    return callForward(tmp::value_sequence::ascending_numbers_t<sizeof...(TInputProperties)>());
  }

  // virtual TO forward(T... inputs) = 0

  template <typename CRTP2, bool TEager2, typename TO2, typename... TInputs2>
  friend class AutoContainer;

private:
  template <size_t... TIndices>
  TO callForward(tmp::value_sequence::IndexSequence<TIndices...>) const
  {
    return static_cast<const CRTP*>(this)->forward(m_input.template get<TIndices>()->get()...);
  }

  void handle()
  {
    this->m_event_changed.raise();
  }

  const tuple::Tuple<TInputProperties*...> m_input;

  NO_COPYING(Eager, <CRTP, TO, TInputProperties...>)
};

template <typename CRTP, typename TO, typename... TInputProperties>
class Lazy : public observable::Property<const TO&>, private Observer<>
{
public:
  Lazy(TInputProperties*... input) // TODO: initialize validated?
    : m_input(input...)
    , m_cache()
    , m_validated(false)
  {
    tuple::for_each([this](const HasEventChanged* in){
      in->EventChanged().subscribe(this);
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

  const TO& get() const
  {
    validate();
    return m_cache;
  }

  // virtual TO forward(T... inputs) = 0

  template <typename CRTP2, bool TEager2, typename TO2, typename... TInputs2>
  friend class AutoContainer;

private:
  void handle()
  {
    invalidate();
    this->m_event_changed.raise(); // TODO: shouldnt check if value was actually changed? Note in documentation?
  }

  template <size_t... TIndices>
  TO callForward(tmp::value_sequence::IndexSequence<TIndices...>) const
  {
    return static_cast<const CRTP*>(this)->forward((m_input.template get<TIndices>()->get())...);
  }

  const tuple::Tuple<TInputProperties*...> m_input;
  mutable TO m_cache;
  mutable bool m_validated;

  NO_COPYING(Lazy, <CRTP, TO, TInputProperties...>)
};

template <typename CRTP, bool TEager, typename TO, typename... TInputProperties>
using EagerOrLazy = typename std::conditional<TEager, Eager<CRTP, TO, TInputProperties...>, Lazy<CRTP, TO, TInputProperties...>>::type;





template <bool TEager, typename T>
using AutoContainerProperty = typename std::conditional<TEager, observable::Container<T>, property::observable::Simple<T>>::type;

#define AUTO_CONTAINER_PROPERTY(TYPE, NAME, INDEX) property::observable::mapped::AutoContainerProperty<TEager, TYPE>& NAME() {return this->template getNthProperty<INDEX>();}

template <typename CRTP, bool TEager, typename TO, typename... TInputs>
class AutoContainer : public EagerOrLazy<CRTP, TEager, TO, AutoContainerProperty<TEager, TInputs>...>
{
public:
  template <typename T>
  using AutoProperty = AutoContainerProperty<TEager, T>;

  AutoContainer(TInputs... initial_inputs)
    : EagerOrLazy<CRTP, TEager, TO, AutoProperty<TInputs>...>(new AutoProperty<TInputs>(initial_inputs)...)
  {
  }

  virtual ~AutoContainer()
  {
    this->m_input.for_each(util::deleter());
  }

private:
  template <size_t TIndex>
  AutoProperty<tmp::type_sequence::nth_type_t<TIndex, tmp::type_sequence::Sequence<TInputs...>>>& getNthProperty()
  {
    return *this->m_input.template get<TIndex>();
  }

  NO_COPYING(AutoContainer, <CRTP, TEager, TO, TInputs...>)
};

template <typename TFunctor, bool TEager, typename... TInputProperties>
class Functor : public EagerOrLazy<Functor<TFunctor, TEager, TInputProperties...>, TEager,
                                                           decltype(std::declval<TFunctor>()(std::declval<TInputProperties>().get()...)),
                                                           TInputProperties...>
{
public:
  using EagerOrLazy<Functor<TFunctor, TEager, TInputProperties...>, TEager,
                                                           decltype(std::declval<TFunctor>()(std::declval<TInputProperties>().get()...)),
                                                           TInputProperties...>::EagerOrLazy;

  Functor(TFunctor functor)
    : m_functor(functor)
  {
  }

  Functor()
    : m_functor()
  {
  }

  template <typename... TI>
  auto forward(TI&&... input) const
  RETURN_AUTO(m_functor(util::forward<TI>(input)...))

private:
  TFunctor m_functor;

  NO_COPYING(Functor, <TFunctor, TEager, TInputProperties...>)
};

} // end of ns mapped





namespace link {

template <typename T1, typename T2>
void uni(property::MutableProperty<T1>& dest, observable::Property<T2>& src)
{
  src.EventChanged() += [&](){
    dest.set(src.get());
  };
}

template <typename T1, typename T2>
void bi(observable::MutableProperty<T1>& first, observable::MutableProperty<T2>& second)
{
  uni(first, second);
  uni(second, first);
}

} // end of ns link

} // end of ns observable

} // end of ns property

#endif // NEO_OBSERVABLE_PROPERTY_H