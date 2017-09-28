#include "Tensor.h"

namespace tensor {

namespace detail {

template <size_t TIndex>
struct MatrixProductDimHelper
{
  template <typename TMatrixTypeLeft, typename TMatrixTypeRight>
  static size_t get(TMatrixTypeLeft&& left, TMatrixTypeRight&& right)
  {
    return 1;
  }
};

template <>
struct MatrixProductDimHelper<0>
{
  template <typename TMatrixTypeLeft, typename TMatrixTypeRight>
  static size_t get(TMatrixTypeLeft&& left, TMatrixTypeRight&& right)
  {
    return left.template dim<0>();
  }
};

template <>
struct MatrixProductDimHelper<1>
{
  template <typename TMatrixTypeLeft, typename TMatrixTypeRight>
  static size_t get(TMatrixTypeLeft&& left, TMatrixTypeRight&& right)
  {
    return right.template dim<1>();
  }
};

} // end of ns detail

template <typename TMatrixTypeLeft, typename TMatrixTypeRight>
class MatrixProduct : public StaticOrDynamicTensorFromSequence<
                                        MatrixProduct<TMatrixTypeLeft, TMatrixTypeRight>,
                                        decltype(std::declval<tensor_elementtype_t<TMatrixTypeLeft>>() * std::declval<tensor_elementtype_t<TMatrixTypeRight>>()),
                                        false,
                                        DimSeq<nth_dimension_v<0, tensor_dimseq_t<TMatrixTypeLeft>>::value, nth_dimension_v<1, tensor_dimseq_t<TMatrixTypeRight>>::value>
                              >
{
public:
  static_assert(are_compatible_dimseqs_v<tensor_dimseq_t<TMatrixTypeLeft>, tensor::DimSeq<DYN, DYN>>::value, "Incompatible dimensions");
  static_assert(are_compatible_dimseqs_v<tensor_dimseq_t<TMatrixTypeRight>, tensor::DimSeq<DYN, DYN>>::value, "Incompatible dimensions");
  static_assert(nth_dimension_v<1, tensor_dimseq_t<TMatrixTypeLeft>>::value == nth_dimension_v<0, tensor_dimseq_t<TMatrixTypeRight>>::value, "Incompatible dimensions");

  using ElementType = decltype(std::declval<tensor_elementtype_t<TMatrixTypeLeft>>() * std::declval<tensor_elementtype_t<TMatrixTypeRight>>());
  using ThisType = MatrixProduct<TMatrixTypeLeft, TMatrixTypeRight>;
  using DimSeq = tensor::DimSeq<nth_dimension_v<0, tensor_dimseq_t<TMatrixTypeLeft>>::value, nth_dimension_v<1, tensor_dimseq_t<TMatrixTypeRight>>::value>;
  using SuperType = StaticOrDynamicTensorFromSequence<ThisType, ElementType, false, DimSeq>;

  __host__ __device__
  MatrixProduct(TMatrixTypeLeft left, TMatrixTypeRight right)
    : SuperType(left.template dim<0>(), right.template dim<1>())
    , m_left(left)
    , m_right(right)
  {
  }

  template <typename... TCoordArgTypes>
  __host__ __device__
  ElementType get_element_impl(TCoordArgTypes&&... coords) const
  {
    ElementType sum = 0;
    for (size_t k = 0; k < m_left.template dim<1>(); k++)
    {
      sum += m_left(getNthCoordinate<0>(util::forward<TCoordArgTypes>(coords)...), k) * m_right(k, getNthCoordinate<1>(util::forward<TCoordArgTypes>(coords)...));
    }
    return sum;
  }

  template <size_t TIndex>
  __host__ __device__
  size_t dyn_dim_impl() const
  {
    return detail::MatrixProductDimHelper<TIndex>::get(m_left, m_right);
  }

  __host__ __device__
  size_t dyn_dim_impl(size_t index) const
  {
    switch (index)
    {
      case 0: return m_left.template dim<0>();
      case 1: return m_right.template dim<1>();
      default: return 1;
    }
  }

  TENSOR_DIMS_IMPL_FROM_IND(dyn_dims_impl)

private:
  TMatrixTypeLeft m_left;
  TMatrixTypeRight m_right;
};

template <typename TMatrixTypeLeft, typename TMatrixTypeRight>
struct TensorTraits<MatrixProduct<TMatrixTypeLeft, TMatrixTypeRight>>
{
  static const bool RETURNS_REFERENCE = false;
  static const mem::MemoryType MEMORY_TYPE = mem::combine<TensorTraits<tensor_clean_t<TMatrixTypeLeft>>::MEMORY_TYPE, TensorTraits<tensor_clean_t<TMatrixTypeRight>>::MEMORY_TYPE>();
};




template <typename TMatrixTypeLeft, typename TMatrixTypeRight, ENABLE_IF(is_tensor_v<TMatrixTypeLeft>::value && is_tensor_v<TMatrixTypeRight>::value)>
__host__ __device__
auto operator*(TMatrixTypeLeft&& left, TMatrixTypeRight&& right)
RETURN_AUTO(MatrixProduct<const_param_tensor_store_t<TMatrixTypeLeft&&>, const_param_tensor_store_t<TMatrixTypeRight&&>>
  (static_cast<param_tensor_forward_t<TMatrixTypeLeft&&>>(left), static_cast<param_tensor_forward_t<TMatrixTypeRight&&>>(right))
)

} // end of ns tensor