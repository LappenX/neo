#include "Tensor.h"

namespace tensor {

template <typename TTensor, size_t TRank>
class DimensionVector : public StaticTensor<DimensionVector<TTensor, TRank>, size_t, TRank>
{
public:
  static_assert(TRank >= non_trivial_dimensions_num_v<tensor_dimseq_t<TTensor>>::value, "Non-trivial dimensions are cut off");

  using ElementType = size_t;
  using ThisType = DimensionVector<TTensor, TRank>;
  using SuperType = StaticTensor<ThisType, ElementType, TRank>;

  __host__ __device__
  DimensionVector(const TTensor& tensor)
    : SuperType(TRank)
    , m_tensor(tensor)
  {
  }

  template <typename... TCoordArgTypes>
  __host__ __device__
  size_t get_element_impl(TCoordArgTypes&&... coords) const
  {
    return m_tensor.dim(getNthCoordinate<0>(util::forward<TCoordArgTypes>(coords)...));
  }

private:
  const TTensor& m_tensor;
};

template <typename TTensor, size_t TRank>
struct TensorTraits<DimensionVector<TTensor, TRank>>
{
  static const bool RETURNS_REFERENCE = false;
  static const mem::MemoryType MEMORY_TYPE = TensorTraits<tensor_clean_t<TTensor>>::MEMORY_TYPE;
};

} // end of ns tensor
