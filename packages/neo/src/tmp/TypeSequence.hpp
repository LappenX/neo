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





template <typename TPredicate, typename TSequence>
struct AllApply;

template <typename TPredicate, typename TType0, typename... TTypesRest>
struct AllApply<TPredicate, Sequence<TType0, TTypesRest...>>
{
  static const bool value = TPredicate::template get<TType0>::value && AllApply<TPredicate, Sequence<TTypesRest...>>::value;
};

template <typename TPredicate>
struct AllApply<TPredicate, Sequence<>>
{
  static const bool value = true;
};

static_assert(!pred::is_convertible_to<size_t>::template get<Sequence<>>::value, "pred::is_convertile_to not working");

static_assert(AllApply<pred::is_convertible_to<size_t>, Sequence<int>>::value, "all_apply_v not working");
static_assert(!AllApply<pred::is_convertible_to<size_t>, Sequence<Sequence<>>>::value, "all_apply_v not working");

} // end of ns detail