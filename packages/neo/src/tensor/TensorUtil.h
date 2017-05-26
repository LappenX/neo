#include "Tensor.h"

namespace tensor {

template <typename TTensorType, typename TElementType, size_t TRows, size_t TCols>
__host__ __device__
bool isSymmetric(const Matrix<TTensorType, TElementType, TRows, TCols>& m)
{
  if (m.rows () != m.cols())
  {
    return false;
  }

  for (size_t r = 0; r < m.rows(); r++)
  {
    for (size_t c = r + 1; c < m.cols(); c++)
    {
      if (m(r, c) != m(c, r))
      {
        return false;
      }
    }
  }
  return true;
}

template <typename TTensorType, typename TElementType, size_t TRows, size_t TCols>
__host__ __device__
bool isQuadratic(const Matrix<TTensorType, TElementType, TRows, TCols>& m)
{
  return m.rows() == m.cols();
}

template <typename TTensorType1, typename TTensorType2, typename TElementType1, typename TElementType2, size_t... TDims1, size_t... TDims2>
__host__ __device__
auto dot(const Tensor<TTensorType1, TElementType1, TDims1...>& t1, const Tensor<TTensorType2, TElementType2, TDims2...>& t2)
RETURN_AUTO(sum(elwiseMul(t1, t2)))

template <typename TTensorType, typename TElementType, size_t... TDims>
__host__ __device__
auto normalize(const Tensor<TTensorType, TElementType, TDims...>& t)
RETURN_AUTO(t / math::sqrt(dot(t, t)))

} // end of ns tensor