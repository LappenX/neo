#include "Tensor.h"

namespace tensor {

namespace detail {

template <typename TOperation, typename... TTensorTypesIn>
struct ElwiseOperationElementTypeHelper
{
  static_assert(sizeof...(TTensorTypesIn) > 0, "Invalid number of input tensors");

  template <typename... TCoordArgTypes, size_t... TIndices>
  __host__ __device__
  static auto get1(tmp::value_sequence::Sequence<size_t, TIndices...>, TCoordArgTypes&&... coords)
  RETURN_AUTO(TOperation()(
    (
      (std::declval<tmp::type_sequence::nth_type_t<TIndices, tmp::type_sequence::Sequence<TTensorTypesIn...>>>())
      (util::forward<TCoordArgTypes>(coords)...)
    )...
  ))

  template <typename... TCoordArgTypes>
  __host__ __device__
  static auto get2(TCoordArgTypes&&... coords)
  RETURN_AUTO(get1(tmp::value_sequence::ascending_numbers_t<sizeof...(TTensorTypesIn)>(), util::forward<TCoordArgTypes>(coords)...))
};

} // end of ns detail

template <typename TOperation, typename... TTensorTypesIn>
using ElwiseOperationElementType = decltype(detail::ElwiseOperationElementTypeHelper<TOperation, TTensorTypesIn...>::get2(std::declval<VectorXs<sizeof...(TTensorTypesIn)>>()));

template <typename TOperation, typename... TTensorTypesIn>
class ElwiseOperationTensor : public StaticOrDynamicTensorFromSequence<
                                        ElwiseOperationTensor<TOperation, TTensorTypesIn...>,
                                        ElwiseOperationElementType<TOperation, TTensorTypesIn...>,
                                        false,
                                        static_dimseq_from_tensors_t<TTensorTypesIn...>
                              >
{
public:
  static_assert(sizeof...(TTensorTypesIn) > 0, "Invalid number of input tensors");

  using ElementType = ElwiseOperationElementType<TOperation, TTensorTypesIn...>;
  using ThisType = ElwiseOperationTensor<TOperation, TTensorTypesIn...>;
  using DimSeq = static_dimseq_from_tensors_t<TTensorTypesIn...>;
  using SuperType = StaticOrDynamicTensorFromSequence<ThisType, ElementType, false, DimSeq>;
  using FirstTensorType = tmp::type_sequence::nth_type_t<0, tmp::type_sequence::Sequence<TTensorTypesIn...>>;

  __host__ __device__
  ElwiseOperationTensor(TOperation op, const TTensorTypesIn&... tensors)
    : SuperType(tuple::nth_element<0>::get(tensors...).dims())
    , m_op(op)
    , m_tensors(tensors...)
  {
    static_assert(are_compatible_dimseqs_v<tensor_dimseq_t<TTensorTypesIn>...>::value, "Incompatible dimensions");
    ASSERT(areSameDimensions(tensors.dims()...), "Operation arguments must have same dimensions");
  }

  template <typename... TCoordArgTypes>
  __host__ __device__
  ElementType get_element_impl(TCoordArgTypes&&... coords) const
  {
    return this->get(tmp::value_sequence::ascending_numbers_t<sizeof...(TTensorTypesIn)>(), util::forward<TCoordArgTypes>(coords)...);
  }

  template <size_t TLength = non_trivial_dimensions_num_v<DimSeq>::value>
  __host__ __device__
  VectorXs<TLength> dyn_dims_impl() const
  {
    return m_tensors.template get<0>().template dims<TLength>();
  }

  template <size_t TIndex>
  __host__ __device__
  size_t dyn_dim_impl() const
  {
    return m_tensors.template get<0>().template dim<TIndex>();
  }

  __host__ __device__
  size_t dyn_dim_impl(size_t index) const
  {
    return m_tensors.template get<0>().dim(index);
  }

private:
  TOperation m_op;
  tuple::Tuple<TTensorTypesIn...> m_tensors;

  template <typename... TCoordArgTypes, size_t... TSequence>
  __host__ __device__
  ElementType get(tmp::value_sequence::Sequence<size_t, TSequence...>, TCoordArgTypes&&... coords) const
  { // TODO: indices are calculated separately for each tensor; might only have to be calculated once if tensor sizes and indexing strategies are equal
    return m_op(
      (
        (m_tensors.template get<TSequence>())
        (util::forward<TCoordArgTypes>(coords)...)
      )...
    );
  }
};

template <typename TOperation, typename... TTensorTypesIn>
struct TensorTraits<ElwiseOperationTensor<TOperation, TTensorTypesIn...>>
{
  static const bool RETURNS_REFERENCE = false;
  static const mem::MemoryType MEMORY_TYPE = mem::combine<TensorTraits<tensor_clean_t<TTensorTypesIn>>::MEMORY_TYPE...>();
};

template <typename TOperation, typename... TTensorTypesIn>
__host__ __device__
ElwiseOperationTensor<TOperation, const_param_tensor_store_t<TTensorTypesIn&&>...> elwise(TOperation op, TTensorTypesIn&&... tensors)
{
  static_assert(sizeof...(TTensorTypesIn) > 0, "Invalid number of input tensors");
  return ElwiseOperationTensor<TOperation, const_param_tensor_store_t<TTensorTypesIn&&>...>(op, static_cast<param_tensor_forward_t<TTensorTypesIn&&>>(tensors)...);
}

#define ELWISE_OP_T(NAME, OPERATION) \
  template <typename TTensorType, ENABLE_IF(is_tensor_v<TTensorType>::value)> \
  __host__ __device__ \
  auto NAME(TTensorType&& t) \
  RETURN_AUTO(elwise<OPERATION>(OPERATION(), \
    util::forward<TTensorType>(t) \
  ))

#define ELWISE_OP_TT(NAME, OPERATION) \
  template <typename TTensorType1, typename TTensorType2, ENABLE_IF(is_tensor_v<TTensorType1>::value && is_tensor_v<TTensorType2>::value)> \
  __host__ __device__ \
  auto NAME(TTensorType1&& t1, TTensorType2&& t2) \
  RETURN_AUTO(elwise<OPERATION>(OPERATION(), \
    util::forward<TTensorType1>(t1), \
    util::forward<TTensorType2>(t2) \
  ))

#define ELWISE_OP_TS(NAME, OPERATION) \
  template <typename TTensorType, typename TElementType, ENABLE_IF(is_tensor_v<TTensorType>::value && !is_tensor_v<TElementType>::value)> \
  __host__ __device__ \
  auto NAME(TTensorType&& t, const TElementType& s) \
  RETURN_AUTO(elwise<OPERATION>(OPERATION(), \
    util::forward<TTensorType>(t), \
    broadcast<tensor_dimseq_t<TTensorType>>(SingletonT<TElementType>(s), t.dims()) \
  ))

#define ELWISE_OP_ST(NAME, OPERATION) \
  template <typename TTensorType, typename TElementType, ENABLE_IF(is_tensor_v<TTensorType>::value && !is_tensor_v<TElementType>::value)> \
  __host__ __device__ \
  auto NAME(const TElementType& s, TTensorType&& t) \
  RETURN_AUTO(elwise<OPERATION>(OPERATION(), \
    broadcast<tensor_dimseq_t<TTensorType>>(SingletonT<TElementType>(s), t.dims()), \
    util::forward<TTensorType>(t) \
  ))


template <typename TNewElementType, typename TTensorType, ENABLE_IF(is_tensor_v<TTensorType>::value)>
__host__ __device__
auto cast_to(TTensorType&& t)
RETURN_AUTO(elwise<math::functor::cast_to<TNewElementType>>(math::functor::cast_to<TNewElementType>(),
  util::forward<TTensorType>(t)
))
// TODO: inplace operators, like operator += etc

ELWISE_OP_TT(operator+, math::functor::add);
ELWISE_OP_TS(operator+, math::functor::add);
ELWISE_OP_ST(operator+, math::functor::add);

ELWISE_OP_T (operator-, math::functor::negate);
ELWISE_OP_TT(operator-, math::functor::subtract);
ELWISE_OP_ST(operator-, math::functor::subtract);
ELWISE_OP_TS(operator-, math::functor::subtract);

ELWISE_OP_TT(elwiseMul, math::functor::multiply);
ELWISE_OP_TS(operator*, math::functor::multiply);
ELWISE_OP_ST(operator*, math::functor::multiply);

ELWISE_OP_TT(operator/, math::functor::divide);
ELWISE_OP_ST(operator/, math::functor::divide);
ELWISE_OP_TS(operator/, math::functor::divide);

ELWISE_OP_TT(operator%, math::functor::mod);
ELWISE_OP_ST(operator%, math::functor::mod);
ELWISE_OP_TS(operator%, math::functor::mod);

ELWISE_OP_TT(fmod, math::functor::fmod);
ELWISE_OP_ST(fmod, math::functor::fmod);
ELWISE_OP_TS(fmod, math::functor::fmod);

ELWISE_OP_T(abs, math::functor::abs);
ELWISE_OP_T(floor, math::functor::floor);
ELWISE_OP_T(ceil, math::functor::ceil);
ELWISE_OP_T(round, math::functor::round);



ELWISE_OP_TT(operator==, math::functor::eq);
ELWISE_OP_ST(operator==, math::functor::eq);
ELWISE_OP_TS(operator==, math::functor::eq);

ELWISE_OP_TT(operator!=, math::functor::neq);
ELWISE_OP_ST(operator!=, math::functor::neq);
ELWISE_OP_TS(operator!=, math::functor::neq);



ELWISE_OP_TT(operator<, math::functor::lt);
ELWISE_OP_ST(operator<, math::functor::lt);
ELWISE_OP_TS(operator<, math::functor::lt);

ELWISE_OP_TT(operator<=, math::functor::lte);
ELWISE_OP_ST(operator<=, math::functor::lte);
ELWISE_OP_TS(operator<=, math::functor::lte);

ELWISE_OP_TT(operator>, math::functor::gt);
ELWISE_OP_ST(operator>, math::functor::gt);
ELWISE_OP_TS(operator>, math::functor::gt);

ELWISE_OP_TT(operator>=, math::functor::gte);
ELWISE_OP_ST(operator>=, math::functor::gte);
ELWISE_OP_TS(operator>=, math::functor::gte);



#undef ELWISE_OP_T
#undef ELWISE_OP_TT
#undef ELWISE_OP_ST
#undef ELWISE_OP_TS

} // end of ns tensor
