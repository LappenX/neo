#include "Tensor.h"

namespace tensor {

namespace detail {

template <size_t I, bool TIsCurrent>
struct ReductionCoordHelper
{
  template <typename... TCoordArgTypes>
  __host__ __device__
  static size_t get(size_t new_coord, TCoordArgTypes&&... old_coords)
  {
    return new_coord;
  }
};

template <size_t I>
struct ReductionCoordHelper<I, false>
{
  template <typename... TCoordArgTypes>
  __host__ __device__
  static size_t get(size_t new_coord, TCoordArgTypes&&... old_coords)
  {
    return getNthCoordinate<I>(util::forward<TCoordArgTypes>(old_coords)...);
  }
};

template <bool TIsReducedDimension, size_t I>
struct ReductionHelper
{
  template <typename TReducedDimsAsSeq, typename TRedOp, typename TTensorTypeIn, typename... TCoordArgTypes, size_t... TIndices>
  __host__ __device__
  static void reduce(tmp::value_sequence::Sequence<size_t, TIndices...> seq, TRedOp& op, TTensorTypeIn&& tensor, TCoordArgTypes&&... coords)
  {
    ASSERT(getNthCoordinate<I - 1>(util::forward<TCoordArgTypes>(coords)...) == 0, "Reduced coordinate has to be zero");
    for (size_t i = 0; i < tensor.template dim<I - 1>(); i++)
    {
      ReductionHelper<tmp::value_sequence::contains_v<size_t, TReducedDimsAsSeq, I - 2>::value, I - 1>
        ::template reduce<TReducedDimsAsSeq>(seq, op, util::forward<TTensorTypeIn>(tensor),
            ReductionCoordHelper<TIndices, TIndices == I - 1>::get(i, util::forward<TCoordArgTypes>(coords)...)...
          );
    }
  }
};

template <size_t I>
struct ReductionHelper<false, I>
{
  template <typename TReducedDimsAsSeq, typename TRedOp, typename TTensorTypeIn, typename... TCoordArgTypes, size_t... TIndices>
  __host__ __device__
  static void reduce(tmp::value_sequence::Sequence<size_t, TIndices...> seq, TRedOp& op, TTensorTypeIn&& tensor, TCoordArgTypes&&... coords)
  {
    ReductionHelper<tmp::value_sequence::contains_v<size_t, TReducedDimsAsSeq, I - 2>::value, I - 1>
        ::template reduce<TReducedDimsAsSeq>(seq, op, util::forward<TTensorTypeIn>(tensor),
            util::forward<TCoordArgTypes>(coords)...
          );
  }
};

template <>
struct ReductionHelper<true, 0>
{
  template <typename TReducedDimsAsSeq, typename TRedOp, typename TTensorTypeIn, typename... TCoordArgTypes, size_t... TIndices>
  __host__ __device__
  static void reduce(tmp::value_sequence::Sequence<size_t, TIndices...> seq, TRedOp& op, TTensorTypeIn&& tensor, TCoordArgTypes&&... coords)
  {
    op.visit(tensor(util::forward<TCoordArgTypes>(coords)...));
  }
};

template <>
struct ReductionHelper<false, 0>
{
  template <typename TReducedDimsAsSeq, typename TRedOp, typename TTensorTypeIn, typename... TCoordArgTypes, size_t... TIndices>
  __host__ __device__
  static void reduce(tmp::value_sequence::Sequence<size_t, TIndices...> seq, TRedOp& op, TTensorTypeIn&& tensor, TCoordArgTypes&&... coords)
  {
    op.visit(tensor(util::forward<TCoordArgTypes>(coords)...));
  }
};





template <size_t I, typename TOriginalDimSeq, typename TReducedDimsAsSeq, bool TIsReducedDim = tmp::value_sequence::contains_v<size_t, TReducedDimsAsSeq, I>::value>
struct StaticReducedDimHelper
{
  static const size_t value = nth_dimension_v<I, TOriginalDimSeq>::value;
};

template <size_t I, typename TOriginalDimSeq, typename TReducedDimsAsSeq>
struct StaticReducedDimHelper<I, TOriginalDimSeq, TReducedDimsAsSeq, true>
{
  static const size_t value = 1;
};



template <typename TOriginalDimSeq, typename TReducedDimsAsSeq, typename TIndexSeq>
struct ReducedDimsHelper;

template <typename TOriginalDimSeq, typename TReducedDimsAsSeq, size_t... TIndices>
struct ReducedDimsHelper<TOriginalDimSeq, TReducedDimsAsSeq, tmp::value_sequence::Sequence<size_t, TIndices...>>
{
  using type = DimSeq<StaticReducedDimHelper<TIndices, TOriginalDimSeq, TReducedDimsAsSeq>::value...>;
};

template <typename TOriginalDimSeq, typename TReducedDimsAsSeq>
using ReducedDimSeq = typename ReducedDimsHelper<TOriginalDimSeq, TReducedDimsAsSeq,
                                  tmp::value_sequence::ascending_numbers_t<non_trivial_dimensions_num_v<TOriginalDimSeq>::value>>::type;

static_assert(std::is_same<ReducedDimSeq<DimSeq<2, 3, 4>, tmp::value_sequence::Sequence<size_t, 1, 5>>, DimSeq<2, 1, 4>>::value, "ReducedDimSeq not working");





template <size_t I, typename TOriginalDimSeq, typename TReducedDimsAsSeq, bool TIsReducedDim = tmp::value_sequence::contains_v<size_t, TReducedDimsAsSeq, I>::value>
struct DynamicReducedDimHelper
{
  template <typename TOtherTensor>
  __host__ __device__
  static size_t get(const TOtherTensor& tensor)
  {
    return tensor.template dim<I>();
  }
};

template <size_t I, typename TOriginalDimSeq, typename TReducedDimsAsSeq>
struct DynamicReducedDimHelper<I, TOriginalDimSeq, TReducedDimsAsSeq, true>
{
  template <typename TOtherTensor>
  __host__ __device__
  static size_t get(const TOtherTensor& tensor)
  {
    return 1;
  }
};





template <bool TIsStatic, size_t I, typename TOriginalDimSeq, typename TReducedDimsAsSeq>
struct StaticOrDynamicReducedDimHelper
{
  template <typename TOtherTensor>
  __host__ __device__
  static size_t get(const TOtherTensor& tensor)
  {
    return DynamicReducedDimHelper<I, TOriginalDimSeq, TReducedDimsAsSeq>::get(tensor);
  }
};

template <size_t I, typename TOriginalDimSeq, typename TReducedDimsAsSeq>
struct StaticOrDynamicReducedDimHelper<true, I, TOriginalDimSeq, TReducedDimsAsSeq>
{
  template <typename TOtherTensor>
  __host__ __device__
  static size_t get(const TOtherTensor& tensor)
  {
    return StaticReducedDimHelper<I, TOriginalDimSeq, TReducedDimsAsSeq>::value;
  }
};

} // end of ns detail





template <typename TRedOp, typename TTensorTypeIn, typename TReducedDimsAsSeq>
class ReductionTensor : public StaticOrDynamicTensorFromSequence<
                                        ReductionTensor<TRedOp, TTensorTypeIn, TReducedDimsAsSeq>,
                                        typename TRedOp::ResultType,
                                        false,
                                        detail::ReducedDimSeq<tensor_dimseq_t<TTensorTypeIn>, TReducedDimsAsSeq>
                              >
{
public:
  using ElementType = typename TRedOp::ResultType;
  using DimSeq = detail::ReducedDimSeq<tensor_dimseq_t<TTensorTypeIn>, TReducedDimsAsSeq>;
  using ThisType = ReductionTensor<TRedOp, TTensorTypeIn, TReducedDimsAsSeq>;
  using SuperType = StaticOrDynamicTensorFromSequence<ThisType, ElementType, false, DimSeq>;

  static const size_t ORIGINAL_NON_TRIVIAL_DIMENSIONS_NUM = non_trivial_dimensions_num_v<tensor_dimseq_t<TTensorTypeIn>>::value;

  __host__ __device__
  ReductionTensor(TTensorTypeIn tensor)
    : SuperType(this->dims())
    , m_tensor(tensor)
  {
  }

  template <typename... TCoordArgTypes>
  __host__ __device__
  ElementType operator()(TCoordArgTypes&&... coords) const
  {
    TRedOp op;
    detail::ReductionHelper<tmp::value_sequence::contains_v<
                                                    size_t,
                                                    TReducedDimsAsSeq,
                                                    ORIGINAL_NON_TRIVIAL_DIMENSIONS_NUM - 1
                                                  >::value,
                    ORIGINAL_NON_TRIVIAL_DIMENSIONS_NUM>
        ::template reduce<TReducedDimsAsSeq>(
            tmp::value_sequence::ascending_numbers_t<ORIGINAL_NON_TRIVIAL_DIMENSIONS_NUM>(),
            op,
            m_tensor,
            util::forward<TCoordArgTypes>(coords)...
          );
    return op.get();
  }

  template <size_t TLength = non_trivial_dimensions_num_v<DimSeq>::value>
  __host__ __device__
  VectorXs<TLength> dyn_dims() const
  {
    return dimsHelper(tmp::value_sequence::ascending_numbers_t<TLength>());
  }

  template <size_t TIndex>
  __host__ __device__
  size_t dyn_dim() const
  {
    return detail::DynamicReducedDimHelper<
                      TIndex,
                      tensor_dimseq_t<TTensorTypeIn>,
                      TReducedDimsAsSeq
                >::get(m_tensor);
  }

  __host__ __device__
  size_t dyn_dim(size_t index) const
  {
    // TODO: more efficient
    return dyn_dims()(index);
  }

private:
  TTensorTypeIn m_tensor;

  template <size_t... TIndices>
  __host__ __device__
  VectorXs<sizeof...(TIndices)> dimsHelper(tmp::value_sequence::Sequence<size_t, TIndices...>) const
  {
    return VectorXs<sizeof...(TIndices)>(detail::DynamicReducedDimHelper<TIndices, tensor_dimseq_t<TTensorTypeIn>, TReducedDimsAsSeq>::get(m_tensor)...);
  }
};

template <typename TRedOp, typename TTensorTypeIn, typename TReducedDimsAsSeq>
struct TensorTraits<ReductionTensor<TRedOp, TTensorTypeIn, TReducedDimsAsSeq>>
{
  static const bool RETURNS_REFERENCE = false;
  static const mem::MemoryType MEMORY_TYPE = TensorTraits<tensor_clean_t<TTensorTypeIn>>::MEMORY_TYPE;
};





namespace acc {

template <typename TResultType, typename TFunctor>
class BinOpAcc
{
public:
  using ResultType = TResultType;

  __host__ __device__
  BinOpAcc()
    : m_is_first(true)
  {
  }

  template <typename TElementType>
  __host__ __device__
  void visit(const TElementType& el)
  {
    if (m_is_first)
    {
      m_is_first = false;
      m_value = el;
    }
    else
    {
      m_value = TFunctor()(m_value, el);
    }
  }

  __host__ __device__
  TResultType get() const
  {
    return m_is_first ? TFunctor()() : m_value;
  }

private:
  TResultType m_value;
  bool m_is_first;
};

} // end of ns acc







template <typename TElementType, typename TFunctor, typename TTensorType, ENABLE_IF(is_tensor_v<TTensorType>::value)>
__host__ __device__
auto reduceAll(TTensorType&& tensor)
RETURN_AUTO(
  ReductionTensor<
              acc::BinOpAcc<TElementType, TFunctor>,
              const_param_tensor_t<TTensorType&&>,
              tmp::value_sequence::ascending_numbers_t<non_trivial_dimensions_num_v<tensor_dimseq_t<TTensorType>>::value>
            >(util::forward<TTensorType>(tensor))
)

template <typename TElementType, typename TFunctor, size_t... TReducedDims, typename TTensorType, ENABLE_IF(is_tensor_v<TTensorType>::value)>
__host__ __device__
auto reduce(TTensorType&& tensor)
RETURN_AUTO(
  ReductionTensor<
              acc::BinOpAcc<TElementType, TFunctor>,
              const_param_tensor_t<TTensorType&&>,
              tmp::value_sequence::Sequence<size_t, TReducedDims...>
            >(util::forward<TTensorType>(tensor))
)

template <typename TElementType = util::EmptyDefaultType, typename TTensorType, ENABLE_IF(is_tensor_v<TTensorType>::value)>
__host__ __device__
auto sum(TTensorType&& tensor)
RETURN_AUTO(reduceAll<WITH_DEFAULT_TYPE(TElementType, tensor_elementtype_t<TTensorType>), math::functor::add>(tensor)())

template <typename TElementType = util::EmptyDefaultType, typename TTensorType, ENABLE_IF(is_tensor_v<TTensorType>::value)>
__host__ __device__
auto prod(TTensorType&& tensor)
RETURN_AUTO(reduceAll<WITH_DEFAULT_TYPE(TElementType, tensor_elementtype_t<TTensorType>), math::functor::multiply>(tensor)())

template <typename TElementType = util::EmptyDefaultType, typename TTensorType, ENABLE_IF(is_tensor_v<TTensorType>::value)>
__host__ __device__
auto all(TTensorType&& tensor)
RETURN_AUTO(reduceAll<WITH_DEFAULT_TYPE(TElementType, tensor_elementtype_t<TTensorType>), math::functor::land>(tensor)())

template <typename TElementType = util::EmptyDefaultType, typename TTensorType, ENABLE_IF(is_tensor_v<TTensorType>::value)>
__host__ __device__
auto any(TTensorType&& tensor)
RETURN_AUTO(reduceAll<WITH_DEFAULT_TYPE(TElementType, tensor_elementtype_t<TTensorType>), math::functor::lor>(tensor)())

} // end of ns tensor