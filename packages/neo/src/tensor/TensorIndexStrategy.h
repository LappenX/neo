#include "Tensor.h"

namespace tensor {
// TODO: rename to something along the lines of "LowDimMajor" to avoid confusion between cases where first coordinate
//       refers to the row index (matrices) and cases where it refers to the horizontal x-coordinate (images)
struct ColMajorIndexStrategy
{
  template <size_t... TDims, typename... TCoords, ENABLE_IF_ARE_SIZE_T(TCoords...)>
  __host__ __device__
  static size_t toIndex(TCoords&&... coords);

  template <typename TVectorType, typename TElementType, size_t TRank, typename... TCoords>
  __host__ __device__
  static size_t toIndex(const Vector<TVectorType, TElementType, TRank>& dims, TCoords&&... coords);
};

struct RowMajorIndexStrategy
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