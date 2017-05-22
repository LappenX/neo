#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <resource/Resources.h>
#include <resource/TextResource.h>

#include <gl/gl.h>

int main (int argc, char* argv[])
{
  LOG(info, "main") << "Entry point";

  res::Manager manager;

  gl::glfw::init();
  gl::GlfwWindow window("neo_title", 500, 400, 60, true);
  gl::glew::init();


  gl::Viewport viewport(0, 0, 500, 400);
  gl::ClearColorBuffer clear_buffer(glm::vec4(0, 0, 1, 1));

  gl::AttributeMapping attribute_mapping;
  attribute_mapping.registerInput("in_location");
  attribute_mapping.registerOutput("out_fragment");

  gl::ShaderStage rectangle_vert_stage(manager.get<res::TextFile>(boost::filesystem::current_path() / "res", "rectangle.vert")->getContent(), GL_VERTEX_SHADER);
  gl::ShaderStage rectangle_frag_stage(manager.get<res::TextFile>(boost::filesystem::current_path() / "res", "rectangle.frag")->getContent(), GL_FRAGMENT_SHADER);
  gl::Shader rectangle_shader(attribute_mapping, &rectangle_vert_stage, &rectangle_frag_stage);

  gl::BufferObject rectangle_data(GL_STATIC_DRAW);
  glm::vec3 data[4];
  data[0] = glm::vec3(-100, 100, 0);
  data[1] = glm::vec3(-100, -100, 0);
  data[2] = glm::vec3(100, 100, 0);
  data[3] = glm::vec3(100, -100, 0);
  rectangle_data.write(GL_ARRAY_BUFFER, reinterpret_cast<uint8_t*>(data), 4 * 3 * sizeof(float));

  gl::VertexArrayObject rectangle_vao(attribute_mapping, GL_TRIANGLE_STRIP);
  rectangle_vao.addAttribute(gl::VertexAttribute("in_location", &rectangle_data, 3 * sizeof(float), 0, GL_FLOAT, 3));
  rectangle_vao.setDrawnVertexNum(4);




  gl::MultiplicationRenderStack<glm::mat4, 64>
    model_matrix;
  
  SimpleObservableProperty<float> left(-200);
  SimpleObservableProperty<float> right(200);
  SimpleObservableProperty<float> bottom(-200);
  SimpleObservableProperty<float> top(200);
  SimpleObservableProperty<float> near(-1);
  SimpleObservableProperty<float> far(1);
  gl::OrthographicProjection<SimpleObservableProperty<float>, SimpleObservableProperty<float>, SimpleObservableProperty<float>,
                             SimpleObservableProperty<float>, SimpleObservableProperty<float>, SimpleObservableProperty<float>>
    projection_matrix(&left, &right, &bottom, &top, &near, &far);

  StaticFunctionMappedProperty<math::functor::multiply, decltype(projection_matrix), decltype(model_matrix)>
    mvp_matrix(&projection_matrix, &model_matrix);

  gl::Uniform<glm::mat4> rectangle_mvp_uniform(&rectangle_shader, "uModelViewProjectionMatrix");

  auto mvp_setter = rectangle_mvp_uniform.makeSetStep(&mvp_matrix);






  
  auto render = gl::then(
                          &clear_buffer,
                          &viewport,
                          gl::sub(&rectangle_shader, gl::then(&mvp_setter, &window, &rectangle_vao))
                        );



  gl::RenderContext context;
  while (true)
  {
    clear_buffer.render(context);
    viewport.render(context);
    rectangle_shader.pre(context);
    mvp_setter.render(context);
    window.render(context);
    rectangle_vao.render(context);
    rectangle_shader.post(context);
    //render->render(context);
    window.afterDraw();
  }

  gl::glfw::deinit();
  gl::glew::deinit();

  // TODO: log flush

  return 0;
}