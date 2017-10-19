#ifndef PROPERTY_H
#define PROPERTY_H

#include <util/Util.h>
#include <util/Tuple.h>
#include <tmp/ValueSequence.h>


namespace property {

template <typename T>
class Property
{
public:
  virtual T get() const = 0;
};

template <typename T>
class MutableProperty : public virtual Property<T>
{
public:
  virtual void set(const T&) = 0;
};





template <typename T>
class Simple : public MutableProperty<T>
{
public:
  Simple(const T& value)
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
  }

private:
  T m_value;
};

template <typename T, typename TGetter>
class Lambda : public Property<T>
{
public:
  Lambda(TGetter getter)
    : m_getter(getter)
  {
  }

  T get() const
  {
    return m_getter();
  }

private:
  TGetter m_getter;
};

template <typename T, typename TGetter>
auto lambda(TGetter getter)
  RETURN_AUTO(Lambda<T, TGetter>(getter))

template <typename T, typename TGetter>
auto lambda_as_unique_ptr(TGetter getter)
  RETURN_AUTO(util::make_unique<Lambda<T, TGetter>>(getter))

template <typename T, typename TGetter, typename TSetter>
class MutableLambda : public MutableProperty<T>
{
public:
  MutableLambda(TGetter getter, TSetter setter)
    : m_getter(getter)
    , m_setter(setter)
  {
  }

  T get() const
  {
    return m_getter();
  }

  void set(const T& value)
  {
    m_setter(value);
  }

private:
  TGetter m_getter;
  TSetter m_setter;
};

template <typename T, typename TGetter, typename TSetter>
auto lambda(TGetter getter, TSetter setter)
  RETURN_AUTO(MutableLambda<T, TGetter, TSetter>(getter, setter))

template <typename T, typename TGetter, typename TSetter>
auto lambda_as_unique_ptr(TGetter getter, TSetter setter)
  RETURN_AUTO(util::make_unique<MutableLambda<T, TGetter, TSetter>>(getter, setter))

template <typename T>
class Container : public Property<T>
{
public:
  template <typename T2>
  Container(Property<T2>* contained)
    : m_contained(lambda_as_unique_ptr<T>([contained](){return contained->get();}))
  {
  }

  Container(const T& value)
    : m_contained(util::make_unique<Simple<T>>(value))
  {
  }

  template <typename TGetter, ENABLE_IF(std::is_assignable<T&, decltype(std::declval<TGetter>()())>::value)>
  Container(TGetter getter)
    : m_contained(lambda_as_unique_ptr<T>(getter))
  {
  }

  T get() const
  {
    return m_contained->get();
  }

  template <typename T2>
  void refer_to(Property<T2>* contained)
  {
    m_contained = lambda_as_unique_ptr<T>([contained](){return contained->get();});
  }

  void refer_to(const T& value)
  {
    m_contained = util::make_unique<Simple<T>>(value);
  }

  template <typename TGetter, ENABLE_IF(std::is_assignable<T&, decltype(std::declval<TGetter>()())>::value)>
  void refer_to(TGetter getter)
  {
    m_contained = lambda_as_unique_ptr<T>(getter);
  }

private:
  std::unique_ptr<Property<T>> m_contained;
};

template <typename T>
class MutableContainer : public MutableProperty<T>
{
public:
  template <typename T2>
  MutableContainer(MutableProperty<T2>* contained)
    : m_contained(lambda_as_unique_ptr<T>([contained](){return contained->get();}), [contained](const T& t){contained->set(t);})
  {
  }

  MutableContainer(const T& value)
    : m_contained(util::make_unique<Simple<T>>(value))
  {
  }

  template <typename TGetter, typename TSetter>
  MutableContainer(TGetter getter, TSetter setter)
    : m_contained(util::make_unique<MutableLambda<T, TGetter, TSetter>>(getter, setter))
  {
  }

  T get() const
  {
    return m_contained->get();
  }

  void set(const T& value)
  {
    m_contained->set(value);
  }

  template <typename T2>
  void refer_to(MutableProperty<T2>* contained)
  {
    m_contained = lambda_as_unique_ptr<T>([contained](){return contained->get();}), [contained](const T& t){contained->set(t);};
  }

  void refer_to(const T& value)
  {
    m_contained = util::make_unique<Simple<T>>(value);
  }

  template <typename TGetter, typename TSetter>
  void refer_to(TGetter getter, TSetter setter)
  {
    m_contained = lambda_as_unique_ptr<T>(getter, setter);
  }

private:
  std::unique_ptr<MutableProperty<T>> m_contained;
};




namespace mapped {

template <typename CRTP, typename TO, typename... TInputProperties>
class Mapped : public Property<TO>
{
public:
  Mapped(TInputProperties*... input)
    : m_input(input...)
  {
  }

  TO get() const
  {
    return callForward(tmp::value_sequence::ascending_numbers_t<sizeof...(TInputProperties)>());
  }

  // virtual TO forward(T... inputs) = 0

private:
  const tuple::Tuple<TInputProperties*...> m_input;

  template <size_t... TIndices>
  TO callForward(tmp::value_sequence::IndexSequence<TIndices...>) const
  {
    return static_cast<const CRTP*>(this)->forward(m_input.template get<TIndices>()->get()...);
  }
};

template <typename TFunctor, typename... TInputProperties>
class Functor : public Mapped<Functor<TFunctor, TInputProperties...>,
                                 decltype(std::declval<TFunctor>()(std::declval<TInputProperties>().get()...)),
                                 TInputProperties...>
{
private:
  TFunctor m_functor;

public:
  using SuperType = Mapped<Functor<TFunctor, TInputProperties...>,
                         decltype(std::declval<TFunctor>()(std::declval<TInputProperties>().get()...)),
                         TInputProperties...>;
  
  Functor(TFunctor functor, TInputProperties*... input)
    : m_functor(functor)
    , SuperType(input...)
  {
  }

  Functor(TInputProperties*... input)
    : m_functor()
    , SuperType(input...)
  {
  }

  template <typename... TI>
  auto forward(TI&&... input) const
  RETURN_AUTO(m_functor(util::forward<TI>(input)...))
};

} // end of ns mapped

} // end of ns property

#endif // PROPERTY_H