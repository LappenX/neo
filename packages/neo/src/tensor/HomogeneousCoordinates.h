#include "Tensor.h"

namespace tensor {

namespace detail {

template <size_t TRows>
struct HomogenizedRowsHelper
{
  static const size_t value = TRows + 1;
};

template <>
struct HomogenizedRowsHelper<DYN>
{
  static const size_t value = DYN;
};

template <size_t TRows>
struct DehomogenizedRowsHelper
{
  static const size_t value = TRows - 1;
};

template <>
struct DehomogenizedRowsHelper<DYN>
{
  static const size_t value = DYN;
};

} // end of ns detail

template <typename TVectorType>
class HomogenizedVector : public StaticOrDynamicTensor<
                                        HomogenizedVector<TVectorType>,
                                        tensor_elementtype_t<TVectorType>,
                                        false,
                                        detail::HomogenizedRowsHelper<tmp::value_sequence::nth_element_v<0, tensor_dimseq_t<TVectorType>>::value>::value
                              >
{
public:
  using ElementType = tensor_elementtype_t<TVectorType>;
  using ThisType = HomogenizedVector<TVectorType>;
  using SuperType = StaticOrDynamicTensor<
                                        HomogenizedVector<TVectorType>,
                                        tensor_elementtype_t<TVectorType>,
                                        false,
                                        detail::HomogenizedRowsHelper<tmp::value_sequence::nth_element_v<0, tensor_dimseq_t<TVectorType>>::value>::value
                              >;
  __host__ __device__
  HomogenizedVector(TVectorType input)
    : SuperType(input.template dim<0>() + 1)
    , m_input(input)
  {
  }

  template <typename... TCoordArgTypes>
  __host__ __device__
  ElementType get_element_impl(TCoordArgTypes&&... coords) const
  {
    if (getNthCoordinate<0>(util::forward<TCoordArgTypes>(coords)...) < this->template dim<0>() - 1)
    {
      return m_input(getNthCoordinate<0>(util::forward<TCoordArgTypes>(coords)...));
    }
    else
    {
      return static_cast<ElementType>(1); // TODO: this, or define ones in math::?
    }
  }

private:
  TVectorType m_input;
};

template <typename TVectorType>
struct TensorTraits<HomogenizedVector<TVectorType>>
{
  static const bool RETURNS_REFERENCE = false;
  static const mem::MemoryType MEMORY_TYPE = TensorTraits<tensor_clean_t<TVectorType>>::MEMORY_TYPE;
};





template <typename TVectorType>
class DehomogenizedVector : public StaticOrDynamicTensor<
                                        DehomogenizedVector<TVectorType>,
                                        tensor_elementtype_t<TVectorType>,
                                        false,
                                        detail::DehomogenizedRowsHelper<tmp::value_sequence::nth_element_v<0, tensor_dimseq_t<TVectorType>>::value>::value
                              >
{
public:
  static_assert(tmp::value_sequence::nth_element_v<0, tensor_dimseq_t<TVectorType>>::value >= 1, "Cannot dehomogenize a vector with 0 rows!");

  using ElementType = tensor_elementtype_t<TVectorType>;
  using ThisType = DehomogenizedVector<TVectorType>;
  using SuperType = StaticOrDynamicTensor<
                                        DehomogenizedVector<TVectorType>,
                                        tensor_elementtype_t<TVectorType>,
                                        false,
                                        detail::DehomogenizedRowsHelper<tmp::value_sequence::nth_element_v<0, tensor_dimseq_t<TVectorType>>::value>::value
                              >;
  __host__ __device__
  DehomogenizedVector(TVectorType input)
    : SuperType(input.template dim<0>() - 1)
    , m_input(input)
  {
    ASSERT(input.template dim<0>() >= 1, "Cannot dehomogenize a vector with 0 rows!");
  }

  template <typename... TCoordArgTypes>
  __host__ __device__
  ElementType get_element_impl(TCoordArgTypes&&... coords) const
  {
    return m_input(getNthCoordinate<0>(util::forward<TCoordArgTypes>(coords)...)) / m_input(this->template dim<0>());
  }

  // TODO: implement size methods, because it shouldnt be stored, but be returned here? Also check VectorCrossProduct.h

private:
  TVectorType m_input;
};

template <typename TVectorType>
struct TensorTraits<DehomogenizedVector<TVectorType>>
{
  static const bool RETURNS_REFERENCE = false;
  static const mem::MemoryType MEMORY_TYPE = TensorTraits<tensor_clean_t<TVectorType>>::MEMORY_TYPE;
};





template <typename TVectorType, ENABLE_IF(is_tensor_v<TVectorType>::value)>
__host__ __device__
auto homogenize(TVectorType&& vector)
RETURN_AUTO(
  HomogenizedVector<const_param_tensor_store_t<TVectorType&&>>(static_cast<param_tensor_forward_t<TVectorType&&>>(vector))
)

template <typename TVectorType, ENABLE_IF(is_tensor_v<TVectorType>::value)>
__host__ __device__
auto dehomogenize(TVectorType&& vector)
RETURN_AUTO(
  DehomogenizedVector<const_param_tensor_store_t<TVectorType&&>>(static_cast<param_tensor_forward_t<TVectorType&&>>(vector))
)

} // end of ns tensor
