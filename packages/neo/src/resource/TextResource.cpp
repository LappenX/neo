#include "TextResource.h"

namespace res {

TextFile::TextFile(const TextFileRID& rid)
  : RIDResource<TextFileRID>(rid)
{
  std::ifstream ifs(rid.getPath().c_str());
  if (!ifs)
  {
    throw ResourceLoadException(rid);
  }
  ifs.imbue(boost::locale::generator()(rid.getLocale()));
  m_content = std::string((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
  ifs.close();
}

TextFile::~TextFile()
{
  m_content = "";
}

} // end of ns res