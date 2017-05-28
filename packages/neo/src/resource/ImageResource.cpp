#include "ImageResource.h"

#include <util/Assert.h>



namespace res {

namespace freeimage {

void init()
{
  FreeImage_Initialise(true);
  const char* version = FreeImage_GetVersion();
  LOG(info, "res") << "Initialized FreeImage library version: " << version;
}

void deinit()
{
  FreeImage_DeInitialise();
  LOG(info, "res") << "Deinitialized FreeImage library";
}

} // end of ns freeimage

ImageFile::ImageFile(const ImageFileRID& rid)
  : RIDResource<ImageFileRID>(rid)
  , m_handle(0)
{
  FREE_IMAGE_FORMAT format = rid.getFormat();
  if (format == FIF_UNKNOWN)
  {
    format = FreeImage_GetFileType(rid.getPath().c_str(), 0);
    if (format == FIF_UNKNOWN)
    {
      LOG(warning, "res") << "Could not determine image file format for file '" << rid.getPath().c_str()
        << "', attempting to get format from file extension";
      
      format = FreeImage_GetFIFFromFilename(rid.getPath().c_str());
      if (!FreeImage_FIFSupportsReading(format))
      {
        throw ResourceLoadException(rid, "Format of " + rid.toResourceString() + " is not supported");
      }
    }
  }

  // TODO: bitmap32 = FreeImage_ConvertTo32Bits(bitmap);

  m_handle = FreeImage_Load(format, rid.getPath().c_str());
  if (!m_handle)
  {
    throw ResourceLoadException(rid);
  }
  
  size_t bits_per_pixel = FreeImage_GetBPP(m_handle);
  ASSERT(bits_per_pixel / 8 * 8 == bits_per_pixel, "Image bits-per-pixel must be a multiple of 8");
  m_image = ImageType(
          tensor::Vector3s(1, bits_per_pixel / 8, FreeImage_GetPitch(m_handle)),
          mem::AllocatedStorage<uint8_t, mem::alloc::heap>(FreeImage_GetBits(m_handle)),
          FreeImage_GetHeight(m_handle), FreeImage_GetWidth(m_handle), bits_per_pixel/ 8
        );
}

ImageFile::~ImageFile()
{
  FreeImage_Unload(m_handle);
}

} // res