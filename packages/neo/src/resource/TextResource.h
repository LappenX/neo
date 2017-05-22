#ifndef UTIL_RESOURCE_TEXTRESOURCE_H
#define UTIL_RESOURCE_TEXTRESOURCE_H

#include <Common.h>
#include <util/Util.h>
#include <util/Property.h>

#include <boost/locale.hpp>
#include <fstream>

#include "FileResource.h"

namespace res {

class TextFile;

class TextFileRID : public FileRID<TextFile>
{
public:
  TextFileRID(const boost::filesystem::path& location, std::string name, std::string locale = "")
    : FileRID<TextFile>(location, name)
    , m_locale(locale)
  {
  }

  TextFileRID(const TextFileRID& other)
    : FileRID<TextFile>(other)
    , m_locale(other.m_locale)
  {
  }

  std::string getLocale() const
  {
    return m_locale;
  }

  size_t hash() const
  {
    return 8623834 + this->FileRID<TextFile>::hash() * 123 + std::hash<std::string>()(m_locale);
  }

  bool operator==(const Identifier& other) const
  {
    const TextFileRID* o = dynamic_cast<const TextFileRID*>(&other);
    return o && o->FileRID<TextFile>::operator==(*this) && o->m_locale == this->m_locale;
  }

  std::string toResourceString() const
  {
    return "text file '" + std::string(this->getPath().c_str()) + "'";
  }

  virtual TextFileRID* copy() const
  {
    return new TextFileRID(*this);
  }

private:
  std::string m_locale;
};

class TextFile : public RIDResource<TextFileRID>
{
public:
  using rid_t = TextFileRID;

  TextFile(const TextFileRID& rid);
  virtual ~TextFile();

  const std::string& getContent() const
  {
    return m_content;
  }

private:
  std::string m_content;

  NO_COPYING(TextFile)
};

} // res

#endif // UTIL_RESOURCE_TEXTRESOURCE_H