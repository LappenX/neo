#include "Tensor.h"

namespace tensor {

template <size_t TRank, typename... TDimArgTypes> 
__host__ __device__
size_t getNthDimension(TDimArgTypes&&... dims);

template <size_t TRank, typename... TCoordArgTypes> 
__host__ __device__
size_t getNthCoordinate(TCoordArgTypes&&... coords);

template <typename... TCoordArgTypes> 
__host__ __device__
constexpr size_t getCoordinateNum();

#include "TensorCoordsAndDims.hpp"

template <typename TDimSeq>
TVALUE(size_t, non_trivial_dimensions_num_v, detail::DimsHelper<TDimSeq>::non_trivial_dimensions_num());

template <typename TDimSeq>
TVALUE(size_t, is_static_dimseq_v, detail::DimsHelper<TDimSeq>::are_static_dims());

template <size_t N, typename TDimSeq>
TVALUE(size_t, nth_dimension_v, detail::DimsHelper<TDimSeq>::nth_dimension(N))

template <size_t N, typename TCoordSeq>
TVALUE(size_t, nth_coordinate_v, detail::DimsHelper<TCoordSeq>::nth_coordinate(N))

template <typename... TDimSeqs>
TVALUE(size_t, are_compatible_dimseqs_v, detail::AreCompatibleDimseqs<TDimSeqs...>::value);

template <typename TDimSeq>
using non_trivial_dim_seq_t = tmp::value_sequence::cut_to_length_from_end_t<TDimSeq, non_trivial_dimensions_num_v<TDimSeq>::value>;

template <typename TDimSeq1, typename TDimSeq2>
TVALUE(size_t, is_same_dimseq_v, std::is_same<non_trivial_dim_seq_t<TDimSeq1>, non_trivial_dim_seq_t<TDimSeq2>>::value);

template <typename TTensorType>
TVALUE(bool, is_vector_v, non_trivial_dimensions_num_v<tensor_dimseq_t<TTensorType>>::value <= 1);

template <typename TTensorType>
TVALUE(bool, is_matrix_v, non_trivial_dimensions_num_v<tensor_dimseq_t<TTensorType>>::value <= 2);

template <typename T>
TVALUE(bool, is_dimseq_v, detail::IsDimSeq<T>::value)

template <typename... TTensorTypes>
using static_dimseq_from_tensors_t = typename detail::GetStaticDimSeqFromTensors<TTensorTypes...>::type;

template <typename... TVectorTypes>
__host__ __device__
bool areSameDimensions(TVectorTypes&&... dims);

template <typename TVectorType, size_t TVectorLength>
__host__ __device__
size_t getNonTrivialDimensionsNum(const Vector<TVectorType, size_t, TVectorLength>& dims);

template <typename TVectorType1, typename... TSrcDimArgs>
__host__ __device__
void copyDims(TVectorType1&& dest, TSrcDimArgs&&... src);





template <typename TThisType, typename TElementType, bool TStoreDimensionsIfDynamic, size_t... TDims>
using StaticOrDynamicTensor = typename std::conditional<is_static_dimseq_v<DimSeq<TDims...>>::value, StaticTensor<TThisType, TElementType, TDims...>,
  typename std::conditional<TStoreDimensionsIfDynamic, DynamicTensorStoreDimensions<TThisType, TElementType, TDims...>, DynamicTensor<TThisType, TElementType, TDims...>>::type
>::type;

#define UNWRAP_NAME(POSTFIX) StaticOrDynamicTensor##POSTFIX
#define UNWRAP_ARGS_DECLARE(...) typename TThisType, typename TElementType, bool TStoreDimensionsIfDynamic, __VA_ARGS__
#define UNWRAP_ARGS_USE(...) TThisType, TElementType, TStoreDimensionsIfDynamic, __VA_ARGS__
#include "UnwrapSequenceHelper.hpp"

} // end of ns tensor