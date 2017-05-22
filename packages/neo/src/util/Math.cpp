#include "Math.h"

#include <glm/glm.hpp>

namespace math {

template <> float Consts<float>::one(1.0f);
template <> glm::mat2 Consts<glm::mat2>::one(1.0f);
template <> glm::mat3 Consts<glm::mat3>::one(1.0f);
template <> glm::mat4 Consts<glm::mat4>::one(1.0f);

} // end of ns math