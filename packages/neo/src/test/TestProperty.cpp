#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Tuple
#include <boost/test/unit_test.hpp>

#include <observer/ObservableProperty.h>
#include <util/Property.h>
#include <util/Math.h>

BOOST_AUTO_TEST_CASE(mapped_property)
{
  SimpleProperty<uint32_t> p1(11);
  const SimpleProperty<uint32_t> p2(12);
  StaticFunctionMappedProperty<math::functor::add, SimpleProperty<uint32_t>, const SimpleProperty<uint32_t>> sum(&p1, &p2);

  BOOST_CHECK(sum.get() == 23);

  p1.set(13);
  BOOST_CHECK(sum.get() == 25);
}

BOOST_AUTO_TEST_CASE(lazy_mapped_observable_property)
{
  SimpleObservableProperty<uint32_t> p1(11);
  const SimpleObservableProperty<uint32_t> p2(12);
  StaticFunctionLazyMappedObservableProperty<math::functor::add, SimpleObservableProperty<uint32_t>, const SimpleObservableProperty<uint32_t>> sum(&p1, &p2);

  BOOST_CHECK(sum.get() == 23);

  p1.set(13);
  BOOST_CHECK(sum.get() == 25);
}