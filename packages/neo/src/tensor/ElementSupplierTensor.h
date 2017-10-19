#include "Tensor.h"

namespace tensor {

template <typename TOperation, typename TElementType, size_t TSupplierDims, typename TDimSeq>
class ElementSupplierTensor : public StaticOrDynamicTensorFromSequence<
                                        ElementSupplierTensor<TOperation, TElementType, TSupplierDims, TDimSeq>,
                                        TElementType,
                                        true,
                                        TDimSeq
                              >
{
public:
  static_assert((non_trivial_dimensions_num_v<TDimSeq>::value <= TSupplierDims), "Too few supplier dimensions");

  using ThisType = ElementSupplierTensor<TOperation, TElementType, TSupplierDims, TDimSeq>;
  using SuperType = StaticOrDynamicTensorFromSequence<ThisType, TElementType, true, TDimSeq>;

  template <typename... TDimArgTypes>
  __host__ __device__
  ElementSupplierTensor(TOperation supplier, TDimArgTypes&&... dim_args)
    : SuperType(util::forward<TDimArgTypes>(dim_args)...)
    , m_supplier(supplier)
  {
  }

  template <typename... TCoordArgTypes>
  __host__ __device__
  TElementType get_element_impl(TCoordArgTypes&&... coords) const
  {
    return this->get(tmp::value_sequence::ascending_numbers_t<TSupplierDims>(), util::forward<TCoordArgTypes>(coords)...);
  }

private:
  TOperation m_supplier;

  template <typename... TCoordArgTypes, size_t... TIndices>
  __host__ __device__
  TElementType get(tmp::value_sequence::Sequence<size_t, TIndices...>, TCoordArgTypes&&... coords) const
  {
    return m_supplier(getNthCoordinate<TIndices>(util::forward<TCoordArgTypes>(coords)...)...);
  }
};

template <typename TOperation, typename TElementType, size_t TSupplierDims, typename TDimSeq>
struct TensorTraits<ElementSupplierTensor<TOperation, TElementType, TSupplierDims, TDimSeq>>
{
  static const bool RETURNS_REFERENCE = false;
  static const mem::MemoryType MEMORY_TYPE = mem::LOCAL;
};

template <typename TElementType, size_t... TDims, typename TOperation, typename... TDimArgTypes>
__host__ __device__
auto fromSupplier(TOperation supplier, TDimArgTypes&&... dim_args)
RETURN_AUTO(ElementSupplierTensor<TOperation, TElementType, sizeof...(TDims), DimSeq<TDims...>>(supplier, util::forward<TDimArgTypes>(dim_args)...))

template <typename TElementType, typename TDimSeq, size_t TSupplierDims = non_trivial_dimensions_num_v<TDimSeq>::value, typename TOperation, typename... TDimArgTypes>
__host__ __device__
auto fromSupplier(TOperation supplier, TDimArgTypes&&... dim_args)
RETURN_AUTO(ElementSupplierTensor<TOperation, TElementType, TSupplierDims, TDimSeq>(supplier, util::forward<TDimArgTypes>(dim_args)...))

} // end of ns tensor
