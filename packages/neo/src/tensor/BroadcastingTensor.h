#include "Tensor.h"

namespace tensor {

template <typename TTensorTypeIn, typename TDimSeq>
class BroadcastingTensor : public StaticOrDynamicTensorFromSequence<
                                        BroadcastingTensor<TTensorTypeIn, TDimSeq>,
                                        tensor_elementtype_t<TTensorTypeIn>,
                                        true,
                                        TDimSeq
                              >
{
public:
  using ElementType = tensor_elementtype_t<TTensorTypeIn>;
  using ThisType = BroadcastingTensor<TTensorTypeIn, TDimSeq>;
  using SuperType = StaticOrDynamicTensorFromSequence<ThisType, ElementType, true, TDimSeq>;

  template <typename... TDimArgTypes>
  __host__ __device__
  BroadcastingTensor(TTensorTypeIn tensor, TDimArgTypes&&... dim_args)
    : SuperType(util::forward<TDimArgTypes>(dim_args)...)
    , m_tensor(tensor)
  {
  }

  template <typename... TCoordArgTypes>
  __host__ __device__
  ElementType& operator()(TCoordArgTypes&&... coords)
  {
    return getHelper(tmp::value_sequence::ascending_numbers_t<non_trivial_dimensions_num_v<TDimSeq>::value>(), util::forward<TCoordArgTypes>(coords)...);
  }

  template <typename... TCoordArgTypes>
  __host__ __device__
  const ElementType& operator()(TCoordArgTypes&&... coords) const
  {
    return getHelper(tmp::value_sequence::ascending_numbers_t<non_trivial_dimensions_num_v<TDimSeq>::value>(), util::forward<TCoordArgTypes>(coords)...);
  }

private:
  TTensorTypeIn m_tensor;

  template <size_t... TIndices, typename... TCoordArgTypes>
  __host__ __device__
  const ElementType& getHelper(tmp::value_sequence::Sequence<size_t, TIndices...>, TCoordArgTypes&&... coords) const
  {
    return m_tensor((getNthCoordinate<TIndices>(util::forward<TCoordArgTypes>(coords)...) % m_tensor.template dim<TIndices>())...);
  }

  template <size_t... TIndices, typename... TCoordArgTypes>
  __host__ __device__
  ElementType& getHelper(tmp::value_sequence::Sequence<size_t, TIndices...>, TCoordArgTypes&&... coords)
  {
    return m_tensor((getNthCoordinate<TIndices>(util::forward<TCoordArgTypes>(coords)...) % m_tensor.template dim<TIndices>())...);
  }
};

template <typename TTensorTypeIn, typename TDimSeq>
struct TensorTraits<BroadcastingTensor<TTensorTypeIn, TDimSeq>>
{
  static const bool RETURNS_REFERENCE = true;
  static const mem::MemoryType MEMORY_TYPE = TensorTraits<tensor_clean_t<TTensorTypeIn>>::MEMORY_TYPE;
};




template <size_t... TBroadcastedDims, typename... TDimArgTypes, typename TOtherTensorType, ENABLE_IF(is_tensor_v<TOtherTensorType>::value)>
__host__ __device__
auto broadcast(TOtherTensorType&& tensor, TDimArgTypes&&... dim_args)
RETURN_AUTO(BroadcastingTensor<non_const_param_tensor_t<TOtherTensorType&&>, DimSeq<TBroadcastedDims...>>
  (util::forward<TOtherTensorType>(tensor), util::forward<TDimArgTypes>(dim_args)...)
)

template <typename TBroadcastedDimSeq, typename... TDimArgTypes, typename TOtherTensorType, ENABLE_IF(is_tensor_v<TOtherTensorType>::value)>
__host__ __device__
auto broadcast(TOtherTensorType&& tensor, TDimArgTypes&&... dim_args)
RETURN_AUTO(BroadcastingTensor<non_const_param_tensor_t<TOtherTensorType&&>, TBroadcastedDimSeq>
  (util::forward<TOtherTensorType>(tensor), util::forward<TDimArgTypes>(dim_args)...)
)

} // end of ns tensor