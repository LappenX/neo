#include "Glm.h"

namespace gl {

glm::vec3 elwiseMul(glm::vec3 first, glm::vec3 second)
{
  return glm::vec3(first.x * second.x, first.y * second.y, first.z * second.z);
}

}