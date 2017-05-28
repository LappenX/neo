#define DEFAULT_TENSOR_INDEX_STRATEGY ColMajorIndexStrategy
#include <tensor/Tensor.h>

using namespace tensor;



TEST_CASE(storage_tensor_values)
{
  Vector3ui vui(1, 2, 3);
  CHECK(vui(0) == 1);
  CHECK(vui(1) == 2);
  CHECK(vui(2) == 3);

  MatrixXXd<3, 4, ColMajorIndexStrategy> md(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  CHECK(md(0, 0) == 1.0);
  CHECK(md(2, 0) == 3.0);
  CHECK(md(2, 2) == 9.0);

  AllocVectorui<mem::alloc::heap> vui2(3);
  vui2 = vui;
  CHECK(vui2(0) == 1);
  CHECK(vui2(1) == 2);
  CHECK(vui2(2) == 3);

  AllocMatrixd<mem::alloc::heap> md2(3, 4);
  md2 = md;

  CHECK(md2(0, 0) == 1.0);
  CHECK(md2(2, 0) == 3.0);
  CHECK(md2(2, 2) == 9.0);
}

TEST_CASE(strided_tensor)
{
  Matrix34ui m(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

  mem::AllocatedStorage<uint32_t, mem::alloc::heap> storage(12);
  for (size_t i = 0; i < 12; i++)
  {
    storage[i] = i + 1;
  }

  StridedStorageTensor<mem::AllocatedStorage<uint32_t, mem::alloc::heap>, uint32_t, 3, 4>
    strided_matrix(Vector2s(1, 3), storage);

  CHECK(all(m == strided_matrix));
}

TEST_CASE(tensor_reduction_and_broadcasting)
{
  Vector3ui vui(1, 2, 3);
  CHECK(sum(vui) == 6);
  CHECK(prod(vui) == 6);
  // TODO: take values independent of storage order of matrix in constructor
  Matrix34d md(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  CHECK(sum(md) == 1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0 + 9.0 + 10.0 + 11.0 + 12.0);
  CHECK(prod(md) == 1.0 * 2.0 * 3.0 * 4.0 * 5.0 * 6.0 * 7.0 * 8.0 * 9.0 * 10.0 * 11.0 * 12.0);
  CHECK(all(reduce<double, math::functor::add, 1>(md) == Vector3d(22, 26, 30)));
  CHECK(all(reduce<double, math::functor::add, 0>(md) == MatrixXXd<1, 4>(6, 15, 24, 33)));

  CHECK(all(broadcast<3, 3>(vui) == MatrixXXui<3, 3>(1, 2, 3, 1, 2, 3, 1, 2, 3)));
  CHECK(all(broadcast<3, DYN>(vui, 3, 3) == MatrixXXui<3, 3>(1, 2, 3, 1, 2, 3, 1, 2, 3)));
}

TEST_CASE(tensor_equality)
{
  Vector3ui vui1(1, 2, 3);
  Vector3ui vui2(1, 2, 4);
  CHECK(all(vui1 == vui1));
  CHECK(any(vui1 != vui2));

  Matrix34d md1(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  Matrix34d md2(1.1, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  CHECK(all(md1 == md1));
  CHECK(any(md1 != md2));
}

TEST_CASE(elwise_operations)
{
  CHECK(all(Vector3ui(1, 2, 3) + Vector3ui(1, 2, 3) == Vector3ui(2, 4, 6)));
  CHECK(all(Vector3ui(1, 2, 3) + 1 == Vector3ui(2, 3, 4)));
  CHECK(all(2 + Vector3ui(1, 2, 3) == Vector3ui(3, 4, 5)));
  CHECK(all(cast_to<uint32_t>(Vector3f(1.2, 2.2, 3.2)) == Vector3ui(1, 2, 3)));
  CHECK(distance(fmod(Vector3f(2, 4, 6), Vector3f(3, 3, 3)), Vector3ui(2, 1, 0)) <= 1e-5f);
}

TEST_CASE(tensor_util)
{
  CHECK(isSymmetric(MatrixXXd<3, 3, ColMajorIndexStrategy>(1.0, 2.0, 1.0, 2.0, 5.0, 4.0, 1.0, 4.0, 3.0)));
  CHECK(!isSymmetric(MatrixXXd<3, 3, ColMajorIndexStrategy>(1.0, 2.0, 1.0, 2.0, 5.0, 4.0, 1.0, 5.0, 3.0)));
  CHECK(all(normalize(13 * Vector2d(0.6, 0.8)) == Vector2d(0.6, 0.8)));
}

TEST_CASE(tensor_cross)
{
  CHECK(all(cross(Vector3ui(1, 0, 0), Vector3ui(0, 1, 0)) == Vector3ui(0, 0, 1)));
}

TEST_CASE(matrix_product)
{
  Matrix23d md1(1, 4, 2, 5, 3, 6);
  Matrix32d md2(7, 9, 11, 8, 10, 12);
  Matrix2d md3(58, 139, 64, 154);
  CHECK(all(md1 * md2 == md3));
}

TEST_CASE(special_tensor_constants)
{
  CHECK(all(identity_matrix<double, 3>::make() == MatrixXXd<3, 3>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)));
  CHECK(all(identity_matrix<double, DYN>::make(3) == MatrixXXd<3, 3>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)));
  CHECK(all(math::consts::one<MatrixXXd<3, 3>>::get() == MatrixXXd<3, 3>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)));

  CHECK(all(identity_matrix<double, 3>::make() == fromSupplier<double, 3, 3>([](size_t r, size_t c){return r == c ? 1 : 0;})));

  //CHECK((unit_vector::type<double>::length<4>::static_direction<2>::make() == Vector4d(0.0, 0.0, 1.0, 0.0)).all());
  //CHECK((unit_vector::type<double>::length<DYN>::static_direction<2>::make(4) == Vector4d(0.0, 0.0, 1.0, 0.0)).all());
  //CHECK((unit_vector::type<double>::length<4>::dynamic_direction::make(2) == Vector4d(0.0, 0.0, 1.0, 0.0)).all());
  //CHECK((unit_vector::type<double>::length<DYN>::dynamic_direction::make(2, 4) == Vector4d(0.0, 0.0, 1.0, 0.0)).all());
}

/*
TEST_CASE(tensor_index_strategy)
{
  CHECK((ColMajorIndexStrategy::fromIndex<11, 4, 8>(ColMajorIndexStrategy::toIndex<11, 4, 8>(5, 3, 7)) == Vector3s(5, 3, 7)).all());
  CHECK((ColMajorIndexStrategy::fromIndex(Vector3s(11, 4, 8), ColMajorIndexStrategy::toIndex<11, 4, 8>(5, 3, 7)) == Vector3s(5, 3, 7)).all());
  CHECK(ColMajorIndexStrategy::toIndex<11, 4, 8>(5, 3, 7) == ColMajorIndexStrategy::toIndex(Vector3s(11, 4, 8), Vector3s(5, 3, 7)));
  CHECK(ColMajorIndexStrategy::toIndex<11, 4, 8>(5, 3, 7) == ColMajorIndexStrategy::toIndex<11, 4, 8>(Vector3s(5, 3, 7)));

  CHECK(ColMajorIndexStrategy::toIndex<11, 4, 8>(Vector3s(5, 0, 0)) == ColMajorIndexStrategy::toIndex(Vector3s(11, 4, 8), Vector1s(5)));
  CHECK(ColMajorIndexStrategy::toIndex<11, 4, 8>(Vector3s(5, 0, 0)) == ColMajorIndexStrategy::toIndex(Vector3s(11, 4, 8), 5));
  CHECK(ColMajorIndexStrategy::toIndex<11, 4, 8>(Vector3s(5, 0, 0)) == ColMajorIndexStrategy::toIndex<11, 4, 8>(5));

  CHECK((RowMajorIndexStrategy::fromIndex<11, 4, 8>(RowMajorIndexStrategy::toIndex<11, 4, 8>(5, 3, 7)) == Vector3s(5, 3, 7)).all());
  CHECK((RowMajorIndexStrategy::fromIndex(Vector3s(11, 4, 8), RowMajorIndexStrategy::toIndex<11, 4, 8>(5, 3, 7)) == Vector3s(5, 3, 7)).all());
  CHECK(RowMajorIndexStrategy::toIndex<11, 4, 8>(5, 3, 7) == RowMajorIndexStrategy::toIndex(Vector3s(11, 4, 8), Vector3s(5, 3, 7)));
  CHECK(RowMajorIndexStrategy::toIndex<11, 4, 8>(5, 3, 7) == RowMajorIndexStrategy::toIndex<11, 4, 8>(Vector3s(5, 3, 7)));

  CHECK(RowMajorIndexStrategy::toIndex<11, 4, 8>(Vector3s(5, 0, 0)) == RowMajorIndexStrategy::toIndex(Vector3s(11, 4, 8), Vector1s(5)));
  CHECK(RowMajorIndexStrategy::toIndex<11, 4, 8>(Vector3s(5, 0, 0)) == RowMajorIndexStrategy::toIndex(Vector3s(11, 4, 8), 5));
  CHECK(RowMajorIndexStrategy::toIndex<11, 4, 8>(Vector3s(5, 0, 0)) == RowMajorIndexStrategy::toIndex<11, 4, 8>(5));
}

TEST_CASE(tensor_reference)
{
  MatrixXXd<3, 4, ColMajorIndexStrategy> m(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  MatrixXXd<2, 3, ColMajorIndexStrategy> sub(5.0, 6.0, 8.0, 9.0, 11.0, 12.0);

  CHECK((ref::dims<2, 3>::dynamic_offset::make(m, Vector2s(1, 1)) == sub).all());
  CHECK((ref::dims<2, 1>::dynamic_offset::make(m, Vector2s(1, 2)) == Vector2d(8.0, 9.0)).all());
  CHECK((ref::dims<2, 3>::static_offset<1, 1>::make(m) == sub).all());
  CHECK((ref::dims<2, 1>::static_offset<1, 2>::make(m) == Vector2d(8.0, 9.0)).all());

  CHECK((ref::dims<DYN, DYN>::dynamic_offset::make(m, Vector2s(1, 1), 2, 3) == sub).all());
  CHECK((ref::dims<DYN, DYN>::dynamic_offset::make(m, Vector2s(1, 2), 2, 1) == Vector2d(8.0, 9.0)).all());
  CHECK((ref::dims<DYN, DYN>::static_offset<1, 1>::make(m, 2, 3) == sub).all());
  CHECK((ref::dims<DYN, DYN>::static_offset<1, 2>::make(m, 2, 1) == Vector2d(8.0, 9.0)).all());
}

TEST_CASE(tensor_transformations)
{
  MatrixXXd<3, 4, ColMajorIndexStrategy> m(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  CHECK((reverse(m) == MatrixXXd<3, 4, ColMajorIndexStrategy>(12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)).all());
  CHECK((reverse(reverse(m)) == m).all());

  CHECK((transpose<2>(m) == MatrixXXd<4, 3, RowMajorIndexStrategy>(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)).all());
  CHECK((transpose<2>(transpose<2>(m)) == m).all());
  CHECK((transpose<2>(Vector3d(1.0, 2.0, 3.0)) == Matrix13d(1.0, 2.0, 3.0)).all());
}



#ifdef __CUDACC__
TEST_CASE_ONLY_HOST(tensor_copying_host_device)
{
  MatrixXXd<3, 4, ColMajorIndexStrategy> m(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);

  MatrixXXd<3, 4, ColMajorIndexStrategy> h;
  copyh<copier::LocalElwise>(h, m);
  AllocMatrixd<mem::alloc::device> d(3, 4);
  copyh<copier::TransferStorage>(d, h);
  copyh<copier::LocalElwise>(h, SingleValueTensor<double, 3, 4>(-1.2));

  copyh<copier::TransferStorage>(h, d);
  CHECK_ONLY_HOST((m == h).all());
  
  copyh<copier::DeviceElwise<>>(d, d * 2);
  copyh<copier::TransferStorage>(h, d);
  CHECK_ONLY_HOST((m * 2 == h).all());
}
#endif

TEST_CASE(matrix_mul)
{
  MatrixXXd<3, 3, ColMajorIndexStrategy> m(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);

  CHECK((m * identity_matrix::type<double>::rowscols<3>::make() == m).all());
  CHECK((identity_matrix::type<double>::rowscols<3>::make() * m == m).all());

  MatrixXXd<3, 2, ColMajorIndexStrategy> m1(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
  MatrixXXd<2, 3, ColMajorIndexStrategy> m2(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
  MatrixXXd<3, 3, ColMajorIndexStrategy> m3(9, 12, 15, 19, 26, 33, 29, 40, 51);

  CHECK((m1 * m2 == m3).all());
  CHECK(isSymmetric(transpose<2>(m) * m));
}

#ifdef __CUDACC__
TEST_CASE_ONLY_HOST(gemm_device_shared_single)
{
  MatrixXXd<3, 2, ColMajorIndexStrategy> m1(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
  MatrixXXd<2, 3, ColMajorIndexStrategy> m2(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
  MatrixXXd<3, 3, ColMajorIndexStrategy> m3(9, 12, 15, 19, 26, 33, 29, 40, 51);

  AllocMatrixd<mem::alloc::heap> h(3, 3);
  AllocMatrixd<mem::alloc::device> d(3, 3);
  copyh<copier::DeviceMatrixMulSharedSingle<>>(d, m1 * m2);
  copyh<copier::TransferStorage>(h, d);

  CHECK_ONLY_HOST((h == m3).all());
}
#endif

TEST_CASE(tensor_concat)
{
  MatrixXXd<3, 2, ColMajorIndexStrategy>(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
  CHECK((MatrixXXd<3, 4, ColMajorIndexStrategy>(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)
    == concat<1>(MatrixXXd<3, 2, ColMajorIndexStrategy>(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
                 MatrixXXd<3, 2, ColMajorIndexStrategy>(7.0, 8.0, 9.0, 10.0, 11.0, 12.0))).all());

  CHECK((concat<1>(concat<1>(Vector3d(1, 2, 3), Vector3d(4, 5, 6)), Vector3d(7, 8, 9))
    == concat<1>(Vector3d(1, 2, 3), concat<1>(Vector3d(4, 5, 6), Vector3d(7, 8, 9)))).all());
  CHECK((concat<1>(Vector3d(1, 2, 3), Vector3d(4, 5, 6), Vector3d(7, 8, 9))
    == concat<1>(Vector3d(1, 2, 3), concat<1>(Vector3d(4, 5, 6), Vector3d(7, 8, 9)))).all());

  CHECK((concat<0>(Vector3d(1, 2, 3), 4)
    == concat<0>(1, Vector3d(2, 3, 4))).all());
}
*/