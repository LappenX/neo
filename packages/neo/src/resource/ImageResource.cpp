#include "ImageResource.h"

namespace res {

FreeImageLibrary FreeImageLibrary::INSTANCE = FreeImageLibrary();

ImageFile::ImageFile(const ImageFileRID& rid)
  : RIDResource<ImageFileRID>(rid)
  , m_handle(0)
  , m_bits_per_pixel(-1)
  , m_width(-1)
  , m_height(-1)
  , m_data(0)
{
  init(FreeImageLibrary::INSTANCE);
  FREE_IMAGE_FORMAT format = rid.getFormat();
  if (format == FIF_UNKNOWN)
  {
    FreeImage_GetFileType(rid.getPath().c_str(), 0);
    if (format == FIF_UNKNOWN)
    {
      LOG(warning, "ImageFile") << "Could not determine image file format for file '" << rid.getPath().c_str()
        << "', attempting to get format from file extension";
      
      format = FreeImage_GetFIFFromFilename(rid.getPath().c_str());
      if (!FreeImage_FIFSupportsReading(format))
      {
        throw InitializationException("Format of " + rid.toResourceString() + " is not supported");
      }
    }
  }

  // bitmap32 = FreeImage_ConvertTo32Bits(bitmap);

  m_handle = FreeImage_Load(format, rid.getPath().c_str());
  if (!m_handle)
  {
    throw InitializationException("Failed to load " + rid.toResourceString());
  }
  m_bits_per_pixel = FreeImage_GetBPP(m_handle);
  m_width = FreeImage_GetWidth(m_handle);
  m_height = FreeImage_GetHeight(m_handle);
  m_data = FreeImage_GetBits(m_handle);
}

ImageFile::~ImageFile()
{
  FreeImage_Unload(m_handle);
  deinit(FreeImageLibrary::INSTANCE);
}

} // res