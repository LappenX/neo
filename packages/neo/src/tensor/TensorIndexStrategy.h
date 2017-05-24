#include "Tensor.h"

namespace tensor {

struct ColMajorIndexStrategy
{
  template <size_t... TDims, typename... TCoords, ENABLE_IF_ARE_SIZE_T(TCoords...)>
  __host__ __device__
  static size_t toIndex(TCoords&&... coords);

  template <typename TVectorType, typename TElementType, size_t TRank, typename... TCoords>
  __host__ __device__
  static size_t toIndex(const Vector<TVectorType, TElementType, TRank>& dims, TCoords&&... coords);
};

#include "TensorIndexStrategy.hpp"

} // end of ns tensor