#ifndef NEO_TUPLE_H
#define NEO_TUPLE_H

#include <Common.h>
#include <util/Util.h>

namespace tuple {

template <typename... TTypes>
class Tuple;



namespace detail {

template <size_t I, typename... TTypes>
struct TupleGet;

template <size_t I, typename TFirst, typename... TRest>
struct TupleGet<I, TFirst, TRest...>
{
  using output_t = typename TupleGet<I - 1, TRest...>::output_t;

  static output_t& get(Tuple<TFirst, TRest...>& tuple)
  {
    return TupleGet<I - 1, TRest...>::get(tuple.m_rest);
  }

  static const output_t& get(const Tuple<TFirst, TRest...>& tuple)
  {
    return TupleGet<I - 1, TRest...>::get(tuple.m_rest);
  }
};

template <typename TFirst, typename... TRest>
struct TupleGet<0, TFirst, TRest...>
{
  using output_t = TFirst;

  static output_t& get(Tuple<TFirst, TRest...>& tuple)
  {
    return tuple.m_first;
  }

  static const output_t& get(const Tuple<TFirst, TRest...>& tuple)
  {
    return tuple.m_first;
  }
};

} // end of ns detail






template <typename TFirst, typename... TRest>
class Tuple<TFirst, TRest...>
{
public:
  template <size_t I>
  using nth_type_t = typename detail::TupleGet<I, TFirst, TRest...>::output_t;

  template <typename TFirst2, typename... TRest2>
  Tuple(TFirst2&& first, TRest2&&... rest)
    : m_first(util::forward<TFirst2>(first))
    , m_rest(util::forward<TRest2>(rest)...)
  {
  }


  template <size_t I, typename... TTypes>
  friend struct detail::TupleGet;

  template <size_t I>
  nth_type_t<I>& get()
  {
    return detail::TupleGet<I, TFirst, TRest...>::get(*this);
  }

  template <size_t I>
  const nth_type_t<I>& get() const
  {
    return detail::TupleGet<I, TFirst, TRest...>::get(*this);
  }

private:
  TFirst m_first;
  Tuple<TRest...> m_rest;
};

template <>
class Tuple<>
{
};





template <size_t I>
struct nth_element
{
  template <typename TFirst, typename... TRest>
  static auto get(TFirst&& first, TRest&&... rest)
    RETURN_AUTO(nth_element<I - 1>::get(util::forward<TRest>(rest)...))
};

template <>
struct nth_element<0>
{
  template <typename TFirst, typename... TRest>
  static auto get(TFirst&& first, TRest&&... rest)
    RETURN_AUTO(util::forward<TFirst>(first))
};





template <typename TFunc>
void for_each(TFunc func)
{
}

template <typename TFunc, typename TArg0>
void for_each(TFunc func, TArg0&& arg0)
{
   func(util::forward<TArg0>(arg0));
}

template <typename TFunc, typename TArg0, typename... TArgs>
void for_each(TFunc func, TArg0&& arg0, TArgs&&... args)
{
   func(util::forward<TArg0>(arg0));
   for_each(func, util::forward<TArgs>(args)...);
}





template <typename TFunction, typename... TArgs>
using result_type_t = decltype(TFunction(std::declval<TArgs>()...));

} // end of ns tuple

#endif // NEO_TUPLE_H
