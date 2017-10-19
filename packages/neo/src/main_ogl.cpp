#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gl/gl.h>

#include <resource/Resources.h>
#include <resource/TextResource.h>
#include <resource/ImageResource.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_on_sphere.hpp>



namespace detail {

template <size_t I>
struct InterpolateHelper
{
  template <size_t TRadius, typename TInterpolator, typename TElementType, typename TTensorTypeSrc, typename TTensorTypeDest, typename... TCoords>
  __host__ __device__
  static void interpolate(TElementType&& interpolate_coord,
    TTensorTypeSrc&& src, TTensorTypeDest&& dest, TCoords... coords)
  {
    for (int32_t r = 0; r < 2 * TRadius; r++)
    {
      InterpolateHelper<I - 1>::template interpolate<TRadius, TInterpolator>(util::forward<TElementType>(interpolate_coord),
        util::forward<TTensorTypeSrc>(src), util::forward<TTensorTypeDest>(dest), r, coords...);
    }
  }
};

template <>
struct InterpolateHelper<0>
{
  template <size_t TRadius, typename TInterpolator, typename TElementType, typename TTensorTypeSrc, typename TTensorTypeDest, typename... TCoords>
  __host__ __device__
  static void interpolate(TElementType&& interpolate_coord,
    TTensorTypeSrc&& src, TTensorTypeDest&& dest, TCoords... coords)
  {
    dest(coords...) = interpolateHelper<TRadius, TInterpolator>(
      tmp::value_sequence::ascending_numbers_t<2 * TRadius>(), util::forward<TTensorTypeSrc>(src),
      util::forward<TElementType>(interpolate_coord), coords...);
  }

  template <size_t TRadius, typename TInterpolator, typename TTensorTypeSrc, typename TElementType, size_t... TRadii, typename... TCoords>
  static TElementType interpolateHelper(tmp::value_sequence::Sequence<size_t, TRadii...>, TTensorTypeSrc&& src, TElementType interpolate_coord, TCoords... coords)
  {
    return TInterpolator()(interpolate_coord, src(TRadii, coords...)...);
  }
};






template <typename TValueSequence>
struct PerlinHelper;

template <size_t TFirstRadius, size_t... TRestRadii>
struct PerlinHelper<tmp::value_sequence::Sequence<size_t, TFirstRadius, TRestRadii...>>
{
  template <typename TInterpolator, typename TTensorTypeLocal, typename TTensorTypeCornerWeights>
  __host__ __device__
  static tensor::tensor_elementtype_t<TTensorTypeLocal>
    interpolate(TTensorTypeLocal&& local, TTensorTypeCornerWeights&& corner_weights)
  {
    using T = tensor::tensor_elementtype_t<TTensorTypeLocal>;
    static const size_t DIMS = tensor::nth_dimension_v<0, tensor::tensor_dimseq_t<TTensorTypeLocal>>::value;

    tensor::DenseLocalStorageTensor<T, tensor::ColMajorIndexStrategy, (2 * TRestRadii)...>
      interpolated_corner_weights;
    
    InterpolateHelper<sizeof...(TRestRadii)>::template interpolate<TFirstRadius, TInterpolator>(
      local(0), util::forward<TTensorTypeCornerWeights>(corner_weights), interpolated_corner_weights);


    return PerlinHelper<tmp::value_sequence::Sequence<size_t, TRestRadii...>>::template interpolate<TInterpolator>(
        *reinterpret_cast<const tensor::VectorXT<T, DIMS - 1>*>(reinterpret_cast<const T*>(&local) + 1),
        interpolated_corner_weights
      );
  }
};

template <>
struct PerlinHelper<tmp::value_sequence::Sequence<size_t>>
{
  template <typename TInterpolator, typename TTensorTypeLocal, typename TTensorTypeCornerWeights>
  __host__ __device__
  static tensor::tensor_elementtype_t<TTensorTypeLocal>
    interpolate(TTensorTypeLocal&& local, TTensorTypeCornerWeights&& corner_weights)
  {
    return corner_weights();
  }
};

} // end of ns detail



struct Lerp
{
  static const size_t RADIUS = 1;

  template <typename T>
  T operator()(T s, T x0, T x1) const
  {
    return x0 * s + x1 * (1 - s);
  }
};




template <typename T, typename TIntegral, size_t TDims>
class CornerGenerator
{
public:
  CornerGenerator(tensor::VectorXT<TIntegral, TDims> map_location)
    : m_map_location(map_location)
    , m_gen()
    , m_distr(TDims)
  {
  }

  template <typename... TCoordArgTypes>
  __host__ __device__
  const tensor::VectorXT<T, TDims>& operator()(TCoordArgTypes&&... coords) const
  {
    //m_gen.seed(); // using coords and m_map_location

    const std::vector<T>& result = m_distr(m_gen);
    return *reinterpret_cast<const tensor::VectorXT<T, TDims>*>(&result[0]);
  }

private:
  tensor::VectorXT<TIntegral, TDims> m_map_location;
  mutable boost::random::mt19937 m_gen;
  mutable boost::uniform_on_sphere<T> m_distr;
};

template <typename T, size_t TDims>
class CenterDiff
{
public:
  CenterDiff(tensor::VectorXT<T, TDims> local)
    : m_local(local)
  {
  }

  template <typename... TCoordArgTypes>
  __host__ __device__
  tensor::VectorXT<T, TDims> operator()(TCoordArgTypes&&... coords) const
  {
    return m_local - tensor::VectorXs<sizeof...(coords)>(coords...);
  }

private:
  tensor::VectorXT<T, TDims> m_local;
};

template <typename TInterpolator, typename T, typename TIntegral, size_t TDims>
T perlin(const tensor::VectorXT<T, TDims>& location)
{
  static_assert(TInterpolator::RADIUS == 1, "Invalid interpolation");
  tensor::VectorXT<T, TDims> local;
  local = tensor::fmod(location, 1.0f);
  // TODO: modify local with smoother function


  // Get corner weights
  tensor::DenseLocalStorageTensorFromSequence<T, tensor::ColMajorIndexStrategy, tmp::value_sequence::repeat_value_t<size_t, 2 * TInterpolator::RADIUS, TDims>>
    corner_weights;

  CornerGenerator<T, TIntegral, TDims> corner_gen(tensor::cast_to<TIntegral>(location));
  auto gradient = tensor::fromSupplier<const tensor::VectorXT<T, TDims>&, tensor::tensor_dimseq_t<decltype(corner_weights)>>(corner_gen);
  CenterDiff<T, TDims> center_diff(local);
  auto diff = tensor::fromSupplier<tensor::VectorXT<T, TDims>, tensor::tensor_dimseq_t<decltype(corner_weights)>>(center_diff);

  corner_weights = tensor::elwise(tensor::functor::dot(), gradient, diff);


  // Interpolate corner weights
  return detail::PerlinHelper<tmp::value_sequence::repeat_value_t<size_t, TInterpolator::RADIUS, TDims>>::
    template interpolate<TInterpolator>(local, corner_weights);
}







/*
void smooth(std::vector<tensor::Vector2d>& out, const std::vector<tensor::Vector2d>& in, int32_t radius)
{
  out.resize(in.size());
  for (int32_t i = 0; i < in.size(); i++)
  {
    size_t num = 0;
    tensor::Vector2d sum(0, 0);
    for (int32_t r = -radius; r <= radius; r++)
    {
      if (i >= -r && i + r < in.size())
      {
        sum = sum + in[i + r];
        num++;
      }
    }
    out[i] = sum / num;
  }
}

void calcDeriv(std::vector<tensor::Vector2d>& out, const std::vector<tensor::Vector2d>& in)
{
  out.resize(in.size() - 1);
  for (size_t i = 0; i < in.size() - 1; i++)
  {
    out[i] = in[i + 1] - in[i];
  }
}

void calcInverseDeriv(std::vector<tensor::Vector2d>& out, const std::vector<tensor::Vector2d>& in, tensor::Vector2d initial)
{
  out.resize(in.size() + 1);
  out[0] = initial;
  for (size_t i = 0; i < in.size(); i++)
  {
    out[i + 1] = out[i] + in[i];
  }
}*/





int main (int argc, char* argv[])
{
  LOG(info, "main") << "Entry point";



  res::freeimage::init(); // TODO: call all init calls automatically?

  //perlin<Lerp, float, size_t>(tensor::Vector2f(3, 3));






  res::Manager manager; 

  gl::glfw::init();
  gl::GlfwWindow window("neo_title", 500, 400, 60, true);
  gl::glew::init();


  auto image = manager.get<res::ImageFile>("/home/lappen", "a.bmp")->getImage();

  std::cout << tensor::cast_to<uint32_t>(*image->getData()) << std::endl;



  gl::Viewport viewport(0, 0, 500, 400);
  gl::ClearColorBuffer clear_buffer(tensor::Vector4f(0, 0, 1, 1));

  gl::AttributeMapping attribute_mapping;
  attribute_mapping.registerInput("in_location");
  attribute_mapping.registerOutput("out_fragment");

  gl::ShaderStage rectangle_vert_stage(manager.get<res::TextFile>(boost::filesystem::current_path() / "res", "rectangle.vert")->getContent(), GL_VERTEX_SHADER);
  gl::ShaderStage rectangle_frag_stage(manager.get<res::TextFile>(boost::filesystem::current_path() / "res", "rectangle.frag")->getContent(), GL_FRAGMENT_SHADER);
  gl::Shader rectangle_shader(attribute_mapping, &rectangle_vert_stage, &rectangle_frag_stage);





  gl::BufferObject rectangle_data(GL_STATIC_DRAW);
  boost::array<tensor::Vector3f, 4> data;
  float size = 0.5f;
  data[0] = tensor::Vector3f(-size, size, 0);
  data[1] = tensor::Vector3f(-size, -size, 0);
  data[2] = tensor::Vector3f(size, size, 0);
  data[3] = tensor::Vector3f(size, -size, 0);
  rectangle_data.write(GL_ARRAY_BUFFER, data);

  gl::BufferObject rectangle_ibo_data(GL_STATIC_DRAW);
  gl::IndexBufferObject rectangle_ibo(&rectangle_ibo_data);
  boost::array<uint32_t, 4> indices;
  indices[0] = 0;
  indices[1] = 1;
  indices[2] = 2;
  indices[3] = 3;
  rectangle_ibo.write(indices);

  gl::VertexArrayObject rectangle_vao(attribute_mapping);
  rectangle_vao.addAttribute(gl::VertexAttribute("in_location", &rectangle_data, 3 * sizeof(float), 0, GL_FLOAT, 3));








  









  gl::MultiplicationRenderStack<tensor::Matrix4f, 64> model_matrix;
  gl::LookAtCamera<MAP_LAZY> view_matrix(tensor::Vector3f(0, 0, 1), tensor::Vector3f(0, 0, 0), tensor::Vector3f(0, 1, 0));
  
  property::mapped::Functor<math::functor::multiply, decltype(view_matrix), decltype(model_matrix)>
    mv_matrix(&view_matrix, &model_matrix);

  //gl::PerspectiveProjection projection_matrix(45.0f / 180.0f * 3.14159265f, ((float) window.getWidth()) / window.getHeight(), 0.01f, 100.0f);
  gl::OrthographicProjection<MAP_LAZY> projection_matrix(0.0f, 1.0f, 1.0f, 0.0f, -1.0f, 1.0f);

  property::mapped::Functor<math::functor::multiply, decltype(projection_matrix), decltype(mv_matrix)>
    mvp_matrix(&projection_matrix, &mv_matrix);

  gl::Uniform<tensor::Matrix4f> rectangle_mvp_uniform(&rectangle_shader, "uModelViewProjectionMatrix", math::consts::one<tensor::Matrix4f>::get());
  rectangle_mvp_uniform.Value().refer_to(&mvp_matrix);







  /*auto render = gl::then(
                          &clear_buffer,
                          &viewport,
                          gl::sub(&rectangle_shader, gl::then(&mvp_setter, &window, &rectangle_vao))
                        );*/


/*
  const size_t MAX_POINT_NUM = 1024;
  gl::BufferObject line_data(GL_DYNAMIC_DRAW);
  line_data.write(GL_ARRAY_BUFFER, NULL, MAX_POINT_NUM * sizeof(tensor::Vector3f));

  gl::VertexArrayObject lines_vao(attribute_mapping);
  lines_vao.addAttribute(gl::VertexAttribute("in_location", &line_data, 3 * sizeof(float), 0, GL_FLOAT, 3));

  bool clicked = false;
  std::vector<tensor::Vector2d> points;

  int deriv_num = 0;

  auto do_calculation = [&](){
    for (size_t i = 0; i < 3; i++)
    {
      std::cout << points[i] << "   ";
    }
    std::cout << std::endl;

    std::vector<std::vector<tensor::Vector2d>> derivs;
    derivs.resize(deriv_num + 1);
    derivs[0] = points;
    size_t i = 0;
    for (; i < deriv_num; i++)
    {
      calcDeriv(derivs[i + 1], derivs[i]);
    }

    std::vector<tensor::Vector2d> smoothed;
    smooth(smoothed, derivs[derivs.size() - 1], 2);
    derivs[derivs.size() - 1] = smoothed;

    for (; i > 0; i--)
    {
      calcInverseDeriv(derivs[i - 1], derivs[i], derivs[i - 1][0]);
    }
    points = derivs[0];






    std::vector<tensor::Vector3f> data;
    data.resize(points.size());
    for (size_t i = 0; i < data.size(); i++)
    {
      data[i] = tensor::Vector3f(points[i](0), points[i](1), 0);
    }

    line_data.write(GL_ARRAY_BUFFER, 0, reinterpret_cast<uint8_t*>(&data[0]), data.size() * sizeof(tensor::Vector3f));
  };


  KeyboardLayout keyboard_layout = makeLayoutUS();
  window.getKeyPressEvent() += [&](gl::GlfwKey key, bool down){
    if (key.getKeyCaption(keyboard_layout) == MOUSE_LEFT)
    {
      if (down)
      {
        clicked = true;
        points.clear();
        std::cout << "Clicked" << std::endl;
      }
      else
      {
        clicked = false;
        std::cout << "Released" << std::endl;
        do_calculation();
      }
    }
    else if (key.getKeyCaption(keyboard_layout) == KEYBOARD_ARROW_UP && down)
    {
      deriv_num++;
      std::cout << "derivs = " << deriv_num << std::endl;
    }
    else if (key.getKeyCaption(keyboard_layout) == KEYBOARD_ARROW_DOWN && down)
    {
      deriv_num--;
      std::cout << "derivs = " << deriv_num << std::endl;
    }
  };
  window.getCursorMoveEvent() += [&](tensor::Vector2d pos){
    if (clicked)
    {
      pos = pos / window.getSize();
      boost::array<tensor::Vector3f, 1> data;
      data[0] = tensor::Vector3f(pos(0), pos(1), 0);
      line_data.write(GL_ARRAY_BUFFER, points.size() * sizeof(tensor::Vector3f), data);
      points.push_back(pos);
      if (points.size() >= MAX_POINT_NUM)
      {
        clicked = false;
        std::cout << "Released" << std::endl;
        do_calculation();
      }
    }
  };*/

  while (true)
  {
    clear_buffer.clear();
    viewport.set();
    rectangle_shader.activate();
    rectangle_mvp_uniform.set();
    window.bind();

    rectangle_vao.render(GL_TRIANGLE_STRIP, 0, 4, &rectangle_ibo);

    //lines_vao.render(GL_LINE_STRIP, 0, points.size());

    rectangle_shader.deactivate();

    window.poll();
    window.swap();
  }

  gl::glfw::deinit();
  gl::glew::deinit();
  res::freeimage::deinit();
  
  // TODO: log flush

  return 0;
}