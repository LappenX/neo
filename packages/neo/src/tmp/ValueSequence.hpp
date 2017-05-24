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





template <typename TSequence, typename TReversedSequence>
struct ReverseSequence;

template <typename T, T TFirst, T... TRest, T... TReversed>
struct ReverseSequence<Sequence<T, TFirst, TRest...>, Sequence<T, TReversed...>>
{
  using type = typename ReverseSequence<Sequence<T, TRest...>, Sequence<T, TFirst, TReversed...>>::type;
};

template <typename T, T... TReversed>
struct ReverseSequence<Sequence<T>, Sequence<T, TReversed...>>
{
  using type = Sequence<T, TReversed...>;
};

template <typename TSequence>
using reverse_t = typename ReverseSequence<TSequence, Sequence<typename TSequence::ElementType>>::type;

static_assert(std::is_same<reverse_t<Sequence<size_t, 1, 2, 3, 4>>, Sequence<size_t, 4, 3, 2, 1>>::value, "reverse_t not working");





template <bool TCut, typename TSequence, size_t TLength>
struct SequenceCutToLengthFromStart;

template <typename T, T TFirst, T... TRest, size_t TLength>
struct SequenceCutToLengthFromStart<true, Sequence<T, TFirst, TRest...>, TLength>
{
  static_assert(math::lt(TLength, sizeof...(TRest) + 1), "Invalid length");
  using type = typename SequenceCutToLengthFromStart<math::gt(sizeof...(TRest), TLength), Sequence<T, TRest...>, TLength>::type;
};

template <typename T, T... TValues, size_t TLength>
struct SequenceCutToLengthFromStart<false, Sequence<T, TValues...>, TLength>
{
  static_assert(math::gte(TLength, sizeof...(TValues)), "Invalid length");
  using type = Sequence<T, TValues...>;
};

template <typename TSequence, size_t TLength>
using cut_to_length_from_start_t = typename SequenceCutToLengthFromStart<math::gt(TSequence::SIZE, TLength), TSequence, TLength>::type;

static_assert(std::is_same<
    cut_to_length_from_start_t<Sequence<size_t, 1, 2, 3, 4, 5, 6>, 2>,
    Sequence<size_t, 5, 6>
  >::value, "cut_to_length_from_start_t not working");

template <typename TSequence, size_t TLength>
using cut_to_length_from_end_t = reverse_t<cut_to_length_from_start_t<reverse_t<TSequence>, TLength>>;

static_assert(std::is_same<
    cut_to_length_from_end_t<Sequence<size_t, 1, 2, 3, 4, 5, 6>, 2>,
    Sequence<size_t, 1, 2>
  >::value, "cut_to_length_from_end_t not working");





template <typename T, typename TSequence, T TElement>
struct SequenceContains;

template <typename T, T TFirst, T... TRest, T TElement>
struct SequenceContains<T, Sequence<T, TFirst, TRest...>, TElement>
{
  static const bool value = TFirst == TElement || SequenceContains<T, Sequence<T, TRest...>, TElement>::value;
};

template <typename T, T TElement>
struct SequenceContains<T, Sequence<T>, TElement>
{
  static const bool value = false;
};

template <typename T, typename TSequence, T TElement>
TVALUE(bool, contains_v, SequenceContains<T, TSequence, TElement>::value)

static_assert(contains_v<size_t, Sequence<size_t, 1, 2, 3, 4, 5, 6>, 5>::value,
  "contains_v not working");
static_assert(!contains_v<size_t, Sequence<size_t, 1, 2, 3, 4, 5, 6>, 7>::value,
  "contains_v not working");





template <typename T, typename TSequence, size_t TSetIndex, T TSetValue, typename TIndexSequence>
struct SequenceSetHelper;

template <typename T, typename TSequence, size_t TSetIndex, T TSetValue, size_t... TIndices>
struct SequenceSetHelper<T, TSequence, TSetIndex, TSetValue, Sequence<size_t, TIndices...>>
{
  using type = Sequence<T, (TSetIndex == TIndices ? TSetValue : NthElement<TIndices, TSequence>::value)...>;
};

template <typename T, typename TSequence, size_t TSetIndex, T TSetValue>
using set_t = typename SequenceSetHelper<T, TSequence, TSetIndex, TSetValue, typename AscendingNumberSequenceGenerator<TSequence::SIZE>::type>::type;

static_assert(std::is_same<
    set_t<size_t, Sequence<size_t, 1, 2, 3, 4, 5, 6>, 4, 2>,
    Sequence<size_t, 1, 2, 3, 4, 2, 6>
  >::value, "set_t not working");

} // end of ns detail