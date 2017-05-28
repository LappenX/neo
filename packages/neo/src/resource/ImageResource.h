#ifndef UTIL_RESOURCE_IMAGERESOURCE_H
#define UTIL_RESOURCE_IMAGERESOURCE_H

#include <Common.h>

#include <util/Logging.h>
#include "FileResource.h"
#include <tensor/Tensor.h>

#include <FreeImage.h>



namespace res {

namespace freeimage {

void init();
void deinit();

} // end of ns freeimage

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

  using ImageType = tensor::StridedStorageTensor<mem::AllocatedStorage<uint8_t, mem::alloc::heap>,
    uint8_t, tensor::DYN, tensor::DYN, tensor::DYN>;

  ImageFile(const ImageFileRID& rid);
  virtual ~ImageFile();

  const ImageType& getImage() const
  {
    return m_image;
  }

private:
  FIBITMAP* m_handle;
  ImageType m_image;

  NO_COPYING(ImageFile)
};

} // res

#endif // UTIL_RESOURCE_IMAGERESOURCE_H