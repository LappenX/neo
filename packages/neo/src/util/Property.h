#ifndef PROPERTY_H
#define PROPERTY_H

#include <util/Util.h>
#include <util/Tuple.h>
#include <tmp/ValueSequence.h>



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
class SimpleProperty : public MutableProperty<T>
{
public:
  SimpleProperty(T value)
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

template <typename T, typename TSupplier>
class LambdaProperty : public Property<T>
{
public:
  LambdaProperty(TSupplier supplier)
    : m_supplier(supplier)
  {
  }

  T get() const
  {
    return m_supplier();
  }

private:
  TSupplier m_supplier;
};

template <typename T, typename TSupplier>
auto lambdaProperty(TSupplier supplier)
  RETURN_AUTO(LambdaProperty<T, TSupplier>(supplier))





template <typename CRTP, typename TO, typename... TInputProperties>
class MappedProperty : public Property<TO>
{
public:
  MappedProperty(TInputProperties*... input)
    : m_input(input...)
  {
  }

  TO get() const
  {
    return callForward(tmp::value_sequence::ascending_numbers_t<sizeof...(TInputProperties)>());
  }

private:
  const tuple::Tuple<TInputProperties*...> m_input;

  template <size_t... TIndices>
  TO callForward(tmp::value_sequence::IndexSequence<TIndices...>) const
  {
    return static_cast<const CRTP*>(this)->forward((m_input.template get<TIndices>()->get())...);
  }
};

template <typename TFunction, typename... TInputProperties>
class StaticFunctionMappedProperty : public MappedProperty<StaticFunctionMappedProperty<TFunction, TInputProperties...>,
                                                           decltype(std::declval<TFunction>()(std::declval<TInputProperties>().get()...)),
                                                           TInputProperties...>
{
public:
  using MappedProperty<StaticFunctionMappedProperty<TFunction, TInputProperties...>,
                                                           decltype(std::declval<TFunction>()(std::declval<TInputProperties>().get()...)),
                                                           TInputProperties...>::MappedProperty;
  
  template <typename... TI>
  auto forward(TI&&... input) const
  RETURN_AUTO(TFunction()(util::forward<TI>(input)...))
};

#endif // PROPERTY_H