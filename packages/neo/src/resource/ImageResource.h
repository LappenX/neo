#ifndef UTIL_RESOURCE_IMAGERESOURCE_H
#define UTIL_RESOURCE_IMAGERESOURCE_H

#include <Common.h>

#include <FreeImagePlus.h>

#include "FileResource.h"

namespace res {

class FreeImageLibrary : public Initializable
{
public:
  static FreeImageLibrary INSTANCE; // TODO: make thread-safe, ResourceManager can use multiple threads?

  void init()
  {
    FreeImage_Initialise(true);
    m_initialized = true;
  }

  void deinit()
  {
    FreeImage_DeInitialise();
    m_initialized = false;
  }

  bool isInitialized() const
  {
    return m_initialized;
  }

private:
  FreeImageLibrary()
    : m_initialized(false)
  {
  }

  FreeImageLibrary(const FreeImageLibrary& other)
    : m_initialized(other.m_initialized)
  {
  }

  bool m_initialized;
};





class ImageFile;

class ImageFileRID : public FileRID<ImageFile>
{
public:
  ImageFileRID(const boost::filesystem::path& location, std::string name, FREE_IMAGE_FORMAT format = FIF_UNKNOWN)
    : FileRID<ImageFile>(location, name)
    , m_format(format)
  {
  }

  FREE_IMAGE_FORMAT getFormat() const
  {
    return m_format;
  }

  size_t hash() const
  {
    return 71625763 + this->FileRID<ImageFile>::hash() * 123 + m_format;
  }

  bool operator==(const Identifier& other) const
  {
    const ImageFileRID* o = dynamic_cast<const ImageFileRID*>(&other);
    return o && o->FileRID<ImageFile>::operator==(*this) && o->m_format == this->m_format;
  }

  std::string toResourceString() const
  {
    return "image file '" + std::string(this->getPath().c_str()) + "'";
  }

  virtual ImageFileRID* copy() const
  {
    return new ImageFileRID(*this);
  }

private:
  FREE_IMAGE_FORMAT m_format;
};

class ImageFile : public RIDResource<ImageFileRID>
{
public:
  using rid_t = ImageFileRID;

  using ResourceType = const uint8_t*; // TODO: make image class

  ImageFile(const ImageFileRID& rid);
  virtual ~ImageFile();

  const ResourceType& getImageData() const // TODO: make image class
  { // TODO: fix weird type
    return m_data;
  }

private:
  FIBITMAP* m_handle;
  uint32_t m_bits_per_pixel;
  uint32_t m_width;
  uint32_t m_height;
  const uint8_t* m_data;

  NO_COPYING(ImageFile)
};

} // res

#endif // UTIL_RESOURCE_IMAGERESOURCE_H