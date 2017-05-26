#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gl/gl.h>

#include <resource/Resources.h>
#include <resource/TextResource.h>



int main (int argc, char* argv[])
{
  LOG(info, "main") << "Entry point";

  res::Manager manager;

  gl::glfw::init();
  gl::GlfwWindow window("neo_title", 500, 400, 60, true);
  gl::glew::init();





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

  gl::VertexArrayObject rectangle_vao(attribute_mapping, GL_TRIANGLE_STRIP);
  rectangle_vao.addAttribute(gl::VertexAttribute("in_location", &rectangle_data, 3 * sizeof(float), 0, GL_FLOAT, 3));





  gl::MultiplicationRenderStack<tensor::Matrix4f, 64>
    model_matrix;

  SimpleObservableProperty<tensor::Vector3f> camera_pos(tensor::Vector3f(0, 0, -1));
  SimpleObservableProperty<tensor::Vector3f> camera_view_target(tensor::Vector3f(0, 0, 0));
  SimpleObservableProperty<tensor::Vector3f> camera_up(tensor::Vector3f(0, 1, 0));
  gl::LookAtCamera<SimpleObservableProperty<tensor::Vector3f>, SimpleObservableProperty<tensor::Vector3f>, SimpleObservableProperty<tensor::Vector3f>>
    view_matrix(&camera_pos, &camera_view_target, &camera_up);
  
  StaticFunctionMappedProperty<math::functor::multiply, decltype(view_matrix), decltype(model_matrix)>
    mv_matrix(&view_matrix, &model_matrix);

  SimpleObservableProperty<float> fov(45.0f / 180.0f * 3.14159265f);
  SimpleObservableProperty<float> aspect_ratio(((float) window.getWidth()) / window.getHeight());
  SimpleObservableProperty<float> near(0.01f);
  SimpleObservableProperty<float> far(100);
  gl::PerspectiveProjection<SimpleObservableProperty<float>, SimpleObservableProperty<float>, SimpleObservableProperty<float>,
                             SimpleObservableProperty<float>>
    projection_matrix(&fov, &aspect_ratio, &near, &far);

  StaticFunctionMappedProperty<math::functor::multiply, decltype(projection_matrix), decltype(mv_matrix)>
    mvp_matrix(&projection_matrix, &mv_matrix);

  gl::Uniform<tensor::Matrix4f> rectangle_mvp_uniform(&rectangle_shader, "uModelViewProjectionMatrix");

  auto mvp_setter = rectangle_mvp_uniform.makeSetStep(&mvp_matrix);





  /*auto render = gl::then(
                          &clear_buffer,
                          &viewport,
                          gl::sub(&rectangle_shader, gl::then(&mvp_setter, &window, &rectangle_vao))
                        );*/

  window.getKeyPressEvent() += [&](gl::GlfwKey key, bool down){
    
  };

  gl::RenderContext context;
  while (true)
  {
    clear_buffer.render(context);
    viewport.render(context);
    rectangle_shader.pre(context);
    mvp_setter.render(context);
    window.render(context);
    rectangle_vao.render(1, 3, &rectangle_ibo);
    rectangle_shader.post(context);

    window.poll();
    window.swap();
  }

  gl::glfw::deinit();
  gl::glew::deinit();

  // TODO: log flush

  return 0;
}