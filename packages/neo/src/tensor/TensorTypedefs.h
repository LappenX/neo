#include "Tensor.h"
// TODO: combine Typedefs and CoordsAndDims files
namespace tensor {

class ColMajorIndexStrategy;
#ifdef DEFAULT_TENSOR_INDEX_STRATEGY
using DefaultIndexStrategy = DEFAULT_TENSOR_INDEX_STRATEGY;
#else
using DefaultIndexStrategy = ColMajorIndexStrategy;
#endif

#define ENABLE_IF_SUPERTENSOR_CONSTRUCTIBLE(...) ENABLE_IF(std::is_constructible<SuperType, __VA_ARGS__>::value)

static const size_t DYN = static_cast<size_t>(-1);

template <size_t... TDims>
using DimSeq = tmp::value_sequence::Sequence<size_t, TDims...>;

template <typename TTensorType>
struct TensorTraits;





#define TENSOR_DIMS_DECLARE_DEFAULT size_t TDim0 = 1, size_t TDim1 = 1, size_t TDim2 = 1, size_t TDim3 = 1, size_t TDim4 = 1
#define TENSOR_DIMS_DECLARE_NO_DEFAULT size_t TDim0, size_t TDim1, size_t TDim2, size_t TDim3, size_t TDim4
#define TENSOR_DIMS_USE TDim0, TDim1, TDim2, TDim3, TDim4
// TODO: choose copier based on tensor types
#define TENSOR_ASSIGN \
  template <typename TTensorType2, typename TElementType2, size_t... TDims2> \
  __host__ __device__ \
  ThisType& operator=(const Tensor<TTensorType2, TElementType2, TDims2...>& other) \
  { \
    copier::Default::copy(*this, other); \
    return *this; \
  }
#define TENSOR_DIMS_IMPL_FROM_IND(NAME) \
  template <size_t TLength> \
  __host__ __device__ \
  VectorXs<TLength> NAME() const \
  { \
    return dims_impl_helper(tmp::value_sequence::ascending_numbers_t<TLength>()); \
  } \
  private: \
  template <size_t... TIndices> \
  __host__ __device__ \
  VectorXs<sizeof...(TIndices)> dims_impl_helper(tmp::value_sequence::Sequence<size_t, TIndices...>) const \
  { \
    return VectorXs<sizeof...(TIndices)>(this->template dim_impl<TIndices>()...); \
  } \
  public:
// TODO: implement TENSOR_DYN_DIMS_IMPL_FROM_IND in base class, not separately in every special class; bool flag




namespace detail {

template <typename TThisType, typename TElementType, TENSOR_DIMS_DECLARE_DEFAULT>
class Tensor;

} // end of ns detail

template <typename TThisType, typename TElementType, size_t... TDims>
using Tensor = detail::Tensor<TThisType, TElementType, TDims...>;

#include "TensorTypedefs.hpp"

template <typename TTensorRefType>
using tensor_clean_t = typename std::remove_const<typename std::remove_reference<TTensorRefType>::type>::type::ThisType;
template <typename TTensorRefType>
using tensor_dimseq_t = decltype(detail::TensorDimSeqHelper::deduce(std::declval<TTensorRefType>()));
template <typename TTensorRefType>
using tensor_elementtype_t = decltype(detail::TensorElementTypeHelper::deduce(std::declval<TTensorRefType>()));
template <typename TType>
TVALUE(bool, is_tensor_v, decltype(detail::IsTensorHelper::deduce(std::declval<TType>()))::value)

template <typename TTensorType>
using const_param_tensor_store_t = decltype(detail::ConstParamTensorStoreHelper::deduce(std::declval<TTensorType>()));
template <typename TTensorType>
using non_const_param_tensor_store_t = decltype(detail::NonConstParamTensorStoreHelper::deduce(std::declval<TTensorType>()));
template <typename TTensorType>
using param_tensor_forward_t = decltype(detail::ParamTensorForwardHelper::deduce(std::declval<TTensorType>()));




template <typename TThisType, typename TElementType, size_t... TDims>
class StaticTensor;
template <typename TThisType, typename TElementType, size_t... TDims>
class DynamicTensor;
template <typename TThisType, typename TElementType, size_t... TDims>
class DynamicTensorStoreDimensions;

template <typename TThisType, typename TElementType>
using Singleton = Tensor<TThisType, TElementType>;
template <typename TThisType, typename TElementType, size_t TDim0>
using Vector = Tensor<TThisType, TElementType, TDim0>;
template <typename TThisType, typename TElementType, size_t TRows, size_t TCols>
using Matrix = Tensor<TThisType, TElementType, TRows, TCols>;





template <typename TStorageType, typename TElementType, typename TIndexStrategy, size_t... TDims>
class DenseStaticStorageTensor;
template <typename TElementType, typename TIndexStrategy, size_t... TDims>
using DenseLocalStorageTensor = DenseStaticStorageTensor<mem::LocalStorage<TElementType, math::multiply(TDims...)>, TElementType, TIndexStrategy, TDims...>;

#define UNWRAP_NAME(POSTFIX) DenseLocalStorageTensor##POSTFIX
#define UNWRAP_ARGS_DECLARE(...) typename TElementType, typename TIndexStrategy, __VA_ARGS__
#define UNWRAP_ARGS_USE(...) TElementType, TIndexStrategy, __VA_ARGS__
#include "UnwrapSequenceHelper.hpp"

template <typename TElementType, size_t TRows, size_t TCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXXT = DenseLocalStorageTensor<TElementType, TIndexStrategy, TRows, TCols>;

template <size_t TRows, size_t TCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXXf = MatrixXXT<float, TRows, TCols, TIndexStrategy>;
template <size_t TRows, size_t TCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXXd = MatrixXXT<double, TRows, TCols, TIndexStrategy>;
template <size_t TRows, size_t TCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXXi = MatrixXXT<int32_t, TRows, TCols, TIndexStrategy>;
template <size_t TRows, size_t TCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXXui = MatrixXXT<uint32_t, TRows, TCols, TIndexStrategy>;
template <size_t TRows, size_t TCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXXs = MatrixXXT<size_t, TRows, TCols, TIndexStrategy>;
template <size_t TRows, size_t TCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXXb = MatrixXXT<bool, TRows, TCols, TIndexStrategy>;

template <size_t TRowsCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXf = MatrixXXf<TRowsCols, TRowsCols, TIndexStrategy>;
template <size_t TRowsCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXd = MatrixXXd<TRowsCols, TRowsCols, TIndexStrategy>;
template <size_t TRowsCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXi = MatrixXXi<TRowsCols, TRowsCols, TIndexStrategy>;
template <size_t TRowsCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXui = MatrixXXui<TRowsCols, TRowsCols, TIndexStrategy>;
template <size_t TRowsCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXs = MatrixXXs<TRowsCols, TRowsCols, TIndexStrategy>;
template <size_t TRowsCols, typename TIndexStrategy = DefaultIndexStrategy>
using MatrixXb = MatrixXXb<TRowsCols, TRowsCols, TIndexStrategy>;

template <typename TElementType, size_t TRows, typename TIndexStrategy = DefaultIndexStrategy>
using VectorXT = DenseLocalStorageTensor<TElementType, TIndexStrategy, TRows>;

template <size_t TRows, typename TIndexStrategy = DefaultIndexStrategy>
using VectorXf = VectorXT<float, TRows, TIndexStrategy>;
template <size_t TRows, typename TIndexStrategy = DefaultIndexStrategy>
using VectorXd = VectorXT<double, TRows, TIndexStrategy>;
template <size_t TRows, typename TIndexStrategy = DefaultIndexStrategy>
using VectorXi = VectorXT<int32_t, TRows, TIndexStrategy>;
template <size_t TRows, typename TIndexStrategy = DefaultIndexStrategy>
using VectorXui = VectorXT<uint32_t, TRows, TIndexStrategy>;
template <size_t TRows, typename TIndexStrategy = DefaultIndexStrategy>
using VectorXs = VectorXT<size_t, TRows, TIndexStrategy>;
template <size_t TRows, typename TIndexStrategy = DefaultIndexStrategy>
using VectorXb = VectorXT<bool, TRows, TIndexStrategy>;

using Matrix13f = MatrixXXf<1, 3>;
using Matrix14f = MatrixXXf<1, 4>;
using Matrix34f = MatrixXXf<3, 4>;

using Matrix1f = MatrixXf<1>;
using Matrix2f = MatrixXf<2>;
using Matrix3f = MatrixXf<3>;
using Matrix4f = MatrixXf<4>;

using Matrix13d = MatrixXXd<1, 3>;
using Matrix23d = MatrixXXd<2, 3>;
using Matrix32d = MatrixXXd<3, 2>;
using Matrix34d = MatrixXXd<3, 4>;

using Matrix13ui = MatrixXXui<1, 3>;
using Matrix23ui = MatrixXXui<2, 3>;
using Matrix32ui = MatrixXXui<3, 2>;
using Matrix34ui = MatrixXXui<3, 4>;

using Matrix2d = MatrixXd<2>;
using Matrix3d = MatrixXd<3>;
using Matrix4d = MatrixXd<4>;

using Matrix2i = MatrixXi<2>;
using Matrix3i = MatrixXi<3>;
using Matrix4i = MatrixXi<4>;

using Vector1f = VectorXf<1>;
using Vector2f = VectorXf<2>;
using Vector3f = VectorXf<3>;
using Vector4f = VectorXf<4>;

using Vector1d = VectorXd<1>;
using Vector2d = VectorXd<2>;
using Vector3d = VectorXd<3>;
using Vector4d = VectorXd<4>;

using Vector1i = VectorXi<1>;
using Vector2i = VectorXi<2>;
using Vector3i = VectorXi<3>;
using Vector4i = VectorXi<4>;

using Vector1ui = VectorXui<1>;
using Vector2ui = VectorXui<2>;
using Vector3ui = VectorXui<3>;
using Vector4ui = VectorXui<4>;

using Vector1s = VectorXs<1>;
using Vector2s = VectorXs<2>;
using Vector3s = VectorXs<3>;
using Vector4s = VectorXs<4>;

template <typename TElementType, typename TIndexStrategy = DefaultIndexStrategy>
using SingletonT = DenseLocalStorageTensor<TElementType, TIndexStrategy>;





template <typename TStorageType, typename TElementType, typename TIndexStrategy, size_t... TDims>
class DenseDynamicStorageTensor;
template <typename TElementType, typename TAllocator, typename TIndexStrategy, size_t... TDims>
using DenseAllocStorageTensor = DenseDynamicStorageTensor<mem::AllocatedStorage<TElementType, TAllocator>, TElementType, TIndexStrategy, TDims...>;

#define UNWRAP_NAME(POSTFIX) DenseAllocStorageTensor##POSTFIX
#define UNWRAP_ARGS_DECLARE(...) typename TElementType, typename TAllocator, typename TIndexStrategy, __VA_ARGS__
#define UNWRAP_ARGS_USE(...) TElementType, TAllocator, TIndexStrategy, __VA_ARGS__
#include "UnwrapSequenceHelper.hpp"

template <typename TElementType, typename TAllocator, typename TIndexStrategy, size_t TRank>
using AllocTensorT = DenseAllocStorageTensorFromSequence<TElementType, TAllocator, TIndexStrategy, tmp::value_sequence::repeat_value_t<size_t, DYN, TRank>>;

template <typename TElementType, typename TAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocMatrixT = AllocTensorT<TElementType, TAllocator, TIndexStrategy, 2>;
template <typename TAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocMatrixf = AllocMatrixT<float, TAllocator, TIndexStrategy>;
template <typename TAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocMatrixd = AllocMatrixT<double, TAllocator, TIndexStrategy>;
template <typename TAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocMatrixi = AllocMatrixT<int32_t, TAllocator, TIndexStrategy>;
template <typename TAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocMatrixui = AllocMatrixT<uint32_t, TAllocator, TIndexStrategy>;
template <typename TAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocMatrixs = AllocMatrixT<size_t, TAllocator, TIndexStrategy>;

template <typename TElementType, typename TAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocVectorT = AllocTensorT<float, TAllocator, TIndexStrategy, 1>;
template <typename TAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocVectorf = AllocVectorT<float, TAllocator, TIndexStrategy>;
template <typename TAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocVectord = AllocVectorT<double, TAllocator, TIndexStrategy>;
template <typename TAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocVectori = AllocVectorT<int32_t, TAllocator, TIndexStrategy>;
template <typename TAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocVectorui = AllocVectorT<uint32_t, TAllocator, TIndexStrategy>;
template <typename TAllocator, typename TIndexStrategy = DefaultIndexStrategy>
using AllocVectors = AllocVectorT<size_t, TAllocator, TIndexStrategy>;

} // end of ns tensor
