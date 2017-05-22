#ifndef UTIL_RESOURCE_FILE_RESOURCE_H
#define UTIL_RESOURCE_FILE_RESOURCE_H

#include <Common.h>

#include "Resources.h"

#include <boost/filesystem.hpp>
#include <boost/functional/hash.hpp>

#include <string>

namespace res {

template <typename T>
class FileRID : public TypedIdentifier<T>
{
public:
  FileRID(const boost::filesystem::path& location, std::string name)
    : m_path(location / name)
  {
  }

  const boost::filesystem::path& getPath() const
  {
    return m_path;
  }

  bool exists() const
  {
    return boost::filesystem::exists(m_path);
  }

  virtual size_t hash() const
  {
    return boost::filesystem::hash_value(m_path);
  }

  virtual bool operator==(const Identifier& other) const
  {
    const FileRID* o = dynamic_cast<const FileRID*>(&other);
    return o && o->m_path == this->m_path;
  }

private:
  boost::filesystem::path m_path;
};

} // end of ns res

#endif // UTIL_RESOURCE_FILE_RESOURCE_H