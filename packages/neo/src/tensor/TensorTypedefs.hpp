
namespace detail {

struct IsTensorHelper
{
  template <typename TThisType, typename TElementType, size_t... TDims>
  static std::true_type deduce(Tensor<TThisType, TElementType, TDims...>)
  {
    return std::true_type();
  }

  static std::false_type deduce(...)
  {
    return std::false_type();
  }
};

struct TensorDimSeqHelper
{
  template <typename TThisType, typename TElementType, size_t... TDims>
  static DimSeq<TDims...> deduce(Tensor<TThisType, TElementType, TDims...>)
  {
    return DimSeq<TDims...>();
  }
};

struct TensorElementTypeHelper
{
  template <typename TThisType, typename TElementType, size_t... TDims>
  static TElementType deduce(Tensor<TThisType, TElementType, TDims...>)
  {
    return std::declval<TElementType>();
  }
};





struct ConstParamTensorHelper
{
  template <typename TTensorType, typename TElementType, size_t... TDims>
  static const TTensorType& deduce(Tensor<TTensorType, TElementType, TDims...>&)
  {
    return std::declval<const TTensorType&>();
  }

  template <typename TTensorType, typename TElementType, size_t... TDims>
  static TTensorType deduce(Tensor<TTensorType, TElementType, TDims...>&&)
  {
    return std::declval<TTensorType>();
  }

  template <typename TTensorType, typename TElementType, size_t... TDims>
  static const TTensorType& deduce(const Tensor<TTensorType, TElementType, TDims...>&)
  {
    return std::declval<const TTensorType&>();
  }

  template <typename TTensorType, typename TElementType, size_t... TDims>
  static TTensorType deduce(const Tensor<TTensorType, TElementType, TDims...>&&)
  {
    return std::declval<TTensorType>();
  }
};

struct NonConstParamTensorHelper
{
  template <typename TTensorType, typename TElementType, size_t... TDims>
  static TTensorType& deduce(Tensor<TTensorType, TElementType, TDims...>&)
  {
    return std::declval<TTensorType&>();
  }

  template <typename TTensorType, typename TElementType, size_t... TDims>
  static TTensorType deduce(Tensor<TTensorType, TElementType, TDims...>&&)
  {
    return std::declval<TTensorType>();
  }

  template <typename TTensorType, typename TElementType, size_t... TDims>
  static const TTensorType& deduce(const Tensor<TTensorType, TElementType, TDims...>&)
  {
    return std::declval<const TTensorType&>();
  }

  template <typename TTensorType, typename TElementType, size_t... TDims>
  static TTensorType deduce(const Tensor<TTensorType, TElementType, TDims...>&&)
  {
    return std::declval<TTensorType>();
  }
};

} // end of ns detail
