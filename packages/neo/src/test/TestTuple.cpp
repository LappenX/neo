#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Tuple
#include <boost/test/unit_test.hpp>

#include <util/Tuple.h>

BOOST_AUTO_TEST_CASE(tuple_elements)
{
  tuple::Tuple<int, float, char> tuple(1, 2.0f, 'c');
  BOOST_CHECK(tuple.get<0>() == 1);
  BOOST_CHECK(tuple.get<1>() == 2.0f);
  BOOST_CHECK(tuple.get<2>() == 'c');

  tuple.get<1>() = 3.0f;
  BOOST_CHECK(tuple.get<1>() == 3.0f);
}

BOOST_AUTO_TEST_CASE(nth_element)
{
  BOOST_CHECK(tuple::nth_element<0>::get(1, 2.0f, 'c') == 1);
  BOOST_CHECK(tuple::nth_element<1>::get(1, 2.0f, 'c') == 2.0f);
  BOOST_CHECK(tuple::nth_element<2>::get(1, 2.0f, 'c') == 'c');
}