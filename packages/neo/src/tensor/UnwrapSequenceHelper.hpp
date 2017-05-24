template <UNWRAP_ARGS_DECLARE(typename TDimSeq)>
struct UNWRAP_NAME(FromSequenceHelper);

template <UNWRAP_ARGS_DECLARE(size_t... TDims)>
struct UNWRAP_NAME(FromSequenceHelper) <UNWRAP_ARGS_USE(DimSeq<TDims...>)>
{
  using type = UNWRAP_NAME() <UNWRAP_ARGS_USE(TDims...)>;
};

template <UNWRAP_ARGS_DECLARE(typename TDimSeq)>
using UNWRAP_NAME(FromSequence) = typename UNWRAP_NAME(FromSequenceHelper) <UNWRAP_ARGS_USE(TDimSeq)>::type;

#undef UNWRAP_NAME
#undef UNWRAP_ARGS_DECLARE
#undef UNWRAP_ARGS_USE