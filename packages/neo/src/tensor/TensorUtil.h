#include "Tensor.h"

namespace tensor {

template <typename TTensorType, typename TElementType, size_t TRows, size_t TCols>
__host__ __device__
bool isSymmetric(const Matrix<TTensorType, TElementType, TRows, TCols>& m)
{
  if (m.template dim<0> () != m.template dim<1>())
  {
    return false;
  }

  for (size_t r = 0; r < m.template dim<0>(); r++)
  {
    for (size_t c = r + 1; c < m.template dim<1>(); c++)
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
  return m.template dim<0>() == m.template dim<1>();
}




#define OPERATION_T(NAME, OPERATION) \
  template <typename TTensorType, ENABLE_IF(is_tensor_v<TTensorType>::value)> \
  __host__ __device__ \
  auto NAME(TTensorType&& t) \
  RETURN_AUTO(OPERATION) \
  namespace functor { \
    struct NAME \
    { \
      template <typename TTensorType, ENABLE_IF(is_tensor_v<TTensorType>::value)> \
      __host__ __device__ \
      auto operator()(TTensorType&& t) const \
      RETURN_AUTO(OPERATION) \
    }; \
  }

#define OPERATION_TT(NAME, OPERATION) \
  template <typename TTensorType1, typename TTensorType2, ENABLE_IF(is_tensor_v<TTensorType1>::value && is_tensor_v<TTensorType2>::value)> \
  __host__ __device__ \
  auto NAME(TTensorType1&& t1, TTensorType2&& t2) \
  RETURN_AUTO(OPERATION) \
  namespace functor { \
    struct NAME \
    { \
      template <typename TTensorType1, typename TTensorType2, ENABLE_IF(is_tensor_v<TTensorType1>::value && is_tensor_v<TTensorType2>::value)> \
      __host__ __device__ \
      auto operator()(TTensorType1&& t1, TTensorType2&& t2) const \
      RETURN_AUTO(OPERATION) \
    }; \
  }


OPERATION_TT(dot, sum(elwiseMul(t1, t2)))
OPERATION_T(length, math::sqrt(tensor::dot(t, t)))
OPERATION_TT(distance, tensor::length(t2 - t1))
OPERATION_T(normalize, t * (static_cast<tensor_elementtype_t<TTensorType>>(1) / math::sqrt(tensor::dot(t, t))))

#undef OPERATION_T
#undef OPERATION_TT

} // end of ns tensor