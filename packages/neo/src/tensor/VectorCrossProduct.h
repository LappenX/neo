#include "Tensor.h"

namespace tensor {



template <typename TVectorTypeLeft, typename TVectorTypeRight>
class VectorCrossProduct : public StaticOrDynamicTensor<
                                        VectorCrossProduct<TVectorTypeLeft, TVectorTypeRight>,
                                        decltype(std::declval<tensor_elementtype_t<TVectorTypeLeft>>() * std::declval<tensor_elementtype_t<TVectorTypeRight>>()),
                                        false,
                                        3
                              >
{
public:
  static_assert(are_compatible_dimseqs_v<tensor_dimseq_t<TVectorTypeLeft>, tensor_dimseq_t<TVectorTypeRight>, DimSeq<3>>::value, "Incompatible dimensions");

  using ElementType = decltype(std::declval<tensor_elementtype_t<TVectorTypeLeft>>() * std::declval<tensor_elementtype_t<TVectorTypeRight>>());
  using ThisType = VectorCrossProduct<TVectorTypeLeft, TVectorTypeRight>;
  using SuperType = StaticOrDynamicTensor<ThisType, ElementType, false, 3>;

  __host__ __device__
  VectorCrossProduct(TVectorTypeLeft left, TVectorTypeRight right)
    : SuperType(3)
    , m_left(left)
    , m_right(right)
  {
  }

  template <typename... TCoordArgTypes>
  __host__ __device__
  ElementType get_element_impl(TCoordArgTypes&&... coords) const
  {
    // TODO: assert non_trivial_coordinates_num <= 1
    switch (getNthCoordinate<0>(util::forward<TCoordArgTypes>(coords)...))
    {
      case 0: return m_left(1) * m_right(2) - m_left(2) * m_right(1);
      case 1: return m_left(2) * m_right(0) - m_left(0) * m_right(2);
      case 2: return m_left(0) * m_right(1) - m_left(1) * m_right(0);
      default: return 0;
    }
  }

private:
  TVectorTypeLeft m_left;
  TVectorTypeRight m_right;
};

template <typename TVectorTypeLeft, typename TVectorTypeRight>
struct TensorTraits<VectorCrossProduct<TVectorTypeLeft, TVectorTypeRight>>
{
  static const bool RETURNS_REFERENCE = false;
  static const mem::MemoryType MEMORY_TYPE = mem::combine<TensorTraits<tensor_clean_t<TVectorTypeLeft>>::MEMORY_TYPE, TensorTraits<tensor_clean_t<TVectorTypeRight>>::MEMORY_TYPE>();
};





template <typename TVectorTypeLeft, typename TVectorTypeRight, ENABLE_IF(is_tensor_v<TVectorTypeLeft>::value && is_tensor_v<TVectorTypeRight>::value)>
__host__ __device__
auto cross(TVectorTypeLeft&& left, TVectorTypeRight&& right)
RETURN_AUTO(VectorCrossProduct<const_param_tensor_store_t<TVectorTypeLeft&&>, const_param_tensor_store_t<TVectorTypeRight&&>>
  (static_cast<param_tensor_forward_t<TVectorTypeLeft&&>>(left), static_cast<param_tensor_forward_t<TVectorTypeRight&&>>(right))
)

} // end of ns tensor
