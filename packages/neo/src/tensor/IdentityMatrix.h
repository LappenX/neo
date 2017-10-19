#include "Tensor.h"

namespace tensor {

template <typename TElementType, size_t TRowsCols>
class IdentityMatrix : public StaticOrDynamicTensorFromSequence<
                                        IdentityMatrix<TElementType, TRowsCols>,
                                        TElementType,
                                        false,
                                        tensor::DimSeq<TRowsCols, TRowsCols>
                              >
{
public:
  using ElementType = TElementType;
  using ThisType = IdentityMatrix<TElementType, TRowsCols>;
  using DimSeq = tensor::DimSeq<TRowsCols, TRowsCols>;
  using SuperType = StaticOrDynamicTensorFromSequence<ThisType, ElementType, false, DimSeq>;

  __host__ __device__
  IdentityMatrix()
    : SuperType(TRowsCols, TRowsCols)
  {
  }

  __host__ __device__
  IdentityMatrix(size_t rows_cols)
    : SuperType(rows_cols, rows_cols)
  {
  }

  template <typename... TCoordArgTypes>
  __host__ __device__
  ElementType get_element_impl(TCoordArgTypes&&... coords) const
  {
    return getNthCoordinate<0>(util::forward<TCoordArgTypes>(coords)...) == getNthCoordinate<1>(util::forward<TCoordArgTypes>(coords)...)
            ? 1 : 0;
  }
};

template <typename TElementType>
class IdentityMatrix<TElementType, DYN> : public StaticOrDynamicTensorFromSequence<
                                        IdentityMatrix<TElementType, DYN>,
                                        TElementType,
                                        false,
                                        tensor::DimSeq<DYN, DYN>
                              >
{
public:
  using ElementType = TElementType;
  using ThisType = IdentityMatrix<TElementType, DYN>;
  using DimSeq = tensor::DimSeq<DYN, DYN>;
  using SuperType = StaticOrDynamicTensorFromSequence<ThisType, ElementType, false, DimSeq>;

  __host__ __device__
  IdentityMatrix(size_t rows_cols)
    : SuperType(rows_cols, rows_cols)
    , m_rows_cols(rows_cols)
  {
  }

  template <typename... TCoordArgTypes>
  __host__ __device__
  ElementType get_element_impl(TCoordArgTypes&&... coords) const
  {
    return getNthCoordinate<0>(util::forward<TCoordArgTypes>(coords)...) == getNthCoordinate<1>(util::forward<TCoordArgTypes>(coords)...)
            ? 1 : 0;
  }

  template <size_t TIndex>
  __host__ __device__
  size_t dyn_dim_impl() const
  {
    return tuple::conditional<math::lt(TIndex, 2)>::get(m_rows_cols, 1);
  }

  __host__ __device__
  size_t dyn_dim_impl(size_t index) const
  {
    return index < 2 ? m_rows_cols : 1;
  }

private:
  size_t m_rows_cols;
};

template <typename TElementType, size_t TRowsCols>
struct TensorTraits<IdentityMatrix<TElementType, TRowsCols>>
{
  static const bool RETURNS_REFERENCE = false;
  static const mem::MemoryType MEMORY_TYPE = mem::LOCAL;
};



template <typename TElementType, size_t TRowsCols>
struct identity_matrix
{
  template <typename... TDimArgTypes>
  __host__ __device__
  static auto make(TDimArgTypes&&... dim_args)
  RETURN_AUTO(IdentityMatrix<TElementType, TRowsCols>(util::forward<TDimArgTypes>(dim_args)...))
};

} // end of ns tensor

namespace math {

namespace consts {

template <typename TStorageType, typename TElementType, typename TIndexStrategy, size_t TRowsCols>
struct one<tensor::DenseStaticStorageTensor<TStorageType, TElementType, TIndexStrategy, TRowsCols, TRowsCols>>
{
  __host__ __device__
  static auto get()
  RETURN_AUTO(tensor::identity_matrix<TElementType, TRowsCols>::make())
};

} // end of ns consts

} // end of ns math
