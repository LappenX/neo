#ifndef STORAGE_H
#define STORAGE_H

#include <Common.h>

#include <type_traits>
#include <util/Util.h>
#include <util/Assert.h>

namespace mem {

enum MemoryType
{
  LOCAL,
  DEVICE,
  HOST
};

template <MemoryType T1>
__host__ __device__
constexpr MemoryType combine()
{
  return T1;
}

template <MemoryType T1, MemoryType T2>
__host__ __device__
constexpr MemoryType combine()
{
  static_assert(T1 == LOCAL || T2 == LOCAL || T1 == T2, "Invalid combination of memory types");
  return T1 == LOCAL ? T2 : T1;
}

template <MemoryType T1, MemoryType T2, MemoryType T3, MemoryType... TRest>
__host__ __device__
constexpr MemoryType combine()
{
  return combine<combine<T1, T2>, T3, TRest...>();
}

template <MemoryType T>
__host__ __device__
constexpr bool is_on_host()
{
  return T == HOST || (T == LOCAL && IS_ON_HOST);
}

template <MemoryType T>
__host__ __device__
constexpr bool is_on_device()
{
  return T == DEVICE || (T == LOCAL && IS_ON_DEVICE);
}

template <MemoryType T>
__host__ __device__
constexpr bool is_on_current()
{
  return (T == LOCAL) || (T == HOST && IS_ON_HOST) || (T == DEVICE && IS_ON_DEVICE);
}



namespace alloc {

namespace detail {

template <bool THostHeap>
struct heap
{
  static const MemoryType MEMORY_TYPE = THostHeap ? HOST : DEVICE;

  template <typename TType>
  __host__ __device__
  static TType* allocate(size_t size)
  {
    return new TType[size];
  }

  template <typename TType>
  __host__ __device__
  static void free(TType* data)
  {
    delete[] data;
  }
};

} // end of ns detail

using heap = detail::heap<IS_ON_HOST>;

struct device
{
  static const MemoryType MEMORY_TYPE = DEVICE;
  
  template <typename TType>
  __host__ __device__
  static TType* allocate(size_t size)
  {
    TType* data;
    cudaMalloc(&data, size * sizeof(TType));
    return data;
  }

  template <typename TType>
  __host__ __device__
  static void free(TType* data)
  {
    cudaFree(data);
  }
};

} // end of ns alloc



template <typename TElementType, size_t TSize>
class LocalStorage final
{
public:
  static const bool HAS_STATIC_SIZE = true;
  static const bool HAS_DYN_SIZE_CONSTRUCTOR = false;
  static const size_t SIZE = TSize;
  static const MemoryType MEMORY_TYPE = LOCAL;
  using ElementType = TElementType;

  __host__ __device__
  static LocalStorage<TElementType, TSize> makeFromSize(size_t size)
  {
    ASSERT(size == TSize, "Dynamic size must equal static size");
    return LocalStorage<TElementType, TSize>();
  }

  __host__ __device__
  constexpr LocalStorage()
    : m_data{}
#ifdef __CUDA_ARCH__
    // TODO: see below
    , m_dummy(true)
#endif
  {
  }

  template <typename TValue0, typename... TValues, ENABLE_IF(sizeof...(TValues) + 1 == TSize
    && std::is_convertible<TValue0, TElementType>::value)>
  __host__ __device__
  constexpr LocalStorage(TValue0 arg0, TValues... args)
    : m_data{static_cast<TElementType>(arg0), static_cast<TElementType>(args)...}
#ifdef __CUDA_ARCH__
    // TODO: see below
    , m_dummy(true)
#endif
  {
  }

  __host__ __device__
  TElementType& operator[](size_t index)
  {
    ASSERT(TSize != 0 && index < TSize, "Index " << index << " out of bounds");
    return m_data[index];
  }

  __host__ __device__
  const TElementType& operator[](size_t index) const
  {
    ASSERT(TSize != 0 && index < TSize, "Index " << index << " out of bounds");
    return m_data[index];
  }

  __host__ __device__
  TElementType* ptr()
  {
    return m_data;
  }

  __host__ __device__
  constexpr const TElementType* ptr() const
  {
    return m_data;
  }

private:
  TElementType m_data[TSize];
#ifdef __CUDA_ARCH__
  // TODO: Ensure object is not zero-sized, since "zero-sized variables are not allowed in device code" ?
  bool m_dummy;
#endif
};

template <typename TElementType, typename TAllocator>
class AllocatedStorage final
{
public:
  static const bool HAS_STATIC_SIZE = false;
  static const bool HAS_DYN_SIZE_CONSTRUCTOR = true;
  static const MemoryType MEMORY_TYPE = TAllocator::MEMORY_TYPE;
  using ElementType = TElementType;

  __host__ __device__
  static AllocatedStorage<TElementType, TAllocator> makeFromSize(size_t size)
  {
    return AllocatedStorage<TElementType, TAllocator>(size);
  }

  __host__ __device__
  AllocatedStorage()
    : m_data(NULL)
    , m_owner(false)
  {
  }

  __host__ __device__
  AllocatedStorage(size_t size)
    : m_data(TAllocator::template allocate<TElementType>(size))
    , m_owner(true)
  {
  }

  __host__ __device__
  AllocatedStorage(TElementType* data) // TODO: allocator can/ should be void for this constructor
    : m_data(data)
    , m_owner(false)
  {
  }

  __host__ __device__
  AllocatedStorage(const AllocatedStorage& other)
    : m_data(other.m_data)
    , m_owner(false)
  {
  }

  __host__ __device__
  AllocatedStorage& operator=(const AllocatedStorage& other)
  {
    m_data = other.m_data;
    m_owner = false;

    return *this;
  }

  __host__ __device__
  AllocatedStorage(AllocatedStorage&& other)
    : m_data(other.m_data)
    , m_owner(other.m_owner)
  {
    other.m_data = NULL;
    other.m_owner = false;
  }

  __host__ __device__
  AllocatedStorage& operator=(AllocatedStorage&& other)
  {
    m_data = other.m_data;
    m_owner = other.m_owner;

    other.m_data = NULL;
    other.m_owner = false;

    return *this;
  }

  __host__ __device__
  ~AllocatedStorage()
  {
    if (m_owner)
    {
      TAllocator::free(m_data);
    }
  }

  __host__ __device__
  TElementType& operator[](size_t index)
  {
    ASSERT(m_data != NULL, "Storage data pointer is null");
    static_assert(!(MEMORY_TYPE == HOST && IS_ON_DEVICE), "Cannot access host elements from device");
    static_assert(!(MEMORY_TYPE == DEVICE && IS_ON_HOST), "Cannot access device elements from host");
    return m_data[index];
  }

  __host__ __device__
  const TElementType& operator[](size_t index) const
  {
    ASSERT(m_data != NULL, "Storage data pointer is null");
    static_assert(!(MEMORY_TYPE == HOST && IS_ON_DEVICE), "Cannot access host elements from device");
    static_assert(!(MEMORY_TYPE == DEVICE && IS_ON_HOST), "Cannot access device elements from host");
    return m_data[index];
  }

  __host__ __device__
  TElementType* ptr()
  {
    return m_data;
  }

  __host__ __device__
  const TElementType* ptr() const
  {
    return m_data;
  }

private:
  TElementType* m_data;
  bool m_owner;
};

} // end of ns mem

#endif // STORAGE_H