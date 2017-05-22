namespace detail {

template <size_t TCur, size_t... TSequence>
struct AscendingNumberSequenceGenerator : AscendingNumberSequenceGenerator<TCur - 1, TCur - 1, TSequence...>
{
};

template <size_t... TSequence>
struct AscendingNumberSequenceGenerator<0, TSequence...>
{
  using type = Sequence<size_t, TSequence...>;
};

template <size_t TCur, size_t... TSequence>
struct DescendingNumberSequenceGenerator : DescendingNumberSequenceGenerator<TCur - 1, TSequence..., TCur - 1>
{
};

template <size_t... TSequence>
struct DescendingNumberSequenceGenerator<0, TSequence...>
{
  using type = Sequence<size_t, TSequence...>;
};

template <typename T, T TValue, size_t I, T... TSequence>
struct RepeatSequenceGenerator : RepeatSequenceGenerator<T, TValue, I - 1, TValue, TSequence...>
{
};

template <typename T, T TValue, T... TSequence>
struct RepeatSequenceGenerator<T, TValue, 0, TSequence...>
{
  using type = Sequence<T, TSequence...>;
};





template <size_t N, typename TSequence>
struct NthElement;

template <size_t N, typename T, T TFirst, T... TRest>
struct NthElement<N, Sequence<T, TFirst, TRest...>>
{
  static const T value = NthElement<N - 1, Sequence<T, TRest...>>::value;
};

template <typename T, T TFirst, T... TRest>
struct NthElement<0, Sequence<T, TFirst, TRest...>>
{
  static const T value = TFirst;
};

} // end of ns detail