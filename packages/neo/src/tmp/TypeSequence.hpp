namespace detail {

template <size_t N, typename TSequence>
struct NthType;

template <size_t N, typename TFirst, typename... TRest>
struct NthType<N, Sequence<TFirst, TRest...>>
{
  using type = typename NthType<N - 1, Sequence<TRest...>>::type;
};

template <typename TFirst, typename... TRest>
struct NthType<0, Sequence<TFirst, TRest...>>
{
  using type = TFirst;
};

} // end of ns detail