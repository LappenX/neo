#include "ImageResource.h"

#include <util/Assert.h>
#include <util/Util.h>


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

void Image::save(const boost::filesystem::path& file, FREE_IMAGE_FORMAT format)
{
  FreeImage_ConvertFromRawBitsEx(false, reinterpret_cast<uint8_t*>(m_data->storage().ptr()), FIT_BITMAP,
    m_data->template dim<0>(), m_data->template dim<1>(), m_data->template dim<0>(), m_color_type.getBitsPerPixel(),
    m_color_type.getMask(RED), m_color_type.getMask(GREEN), m_color_type.getMask(BLUE), true);
}





ImageFile::ImageFile(const ImageFileRID& rid)
  : RIDResource<ImageFileRID>(rid)
  //, m_image(mem::AllocatedStorage<uint8_t, mem::alloc::heap>(static_cast<uint8_t*>(NULL)))
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
        throw ResourceLoadException(rid, "Format is not supported");
      }
    }
  }

  FIBITMAP* handle = FreeImage_Load(format, rid.getPath().c_str());
  if (!handle)
  {
    throw ResourceLoadException(rid);
  }
  
  int width = FreeImage_GetWidth(handle);
  int height = FreeImage_GetHeight(handle);
  size_t bits_per_pixel = FreeImage_GetBPP(handle);
  ASSERT(bits_per_pixel / 8 * 8 == bits_per_pixel, "Image bits-per-pixel must be a multiple of 8");

  using StridedImageType = tensor::StridedStorageTensor<mem::AllocatedStorage<uint8_t, mem::alloc::heap>,
    uint8_t, tensor::DYN, tensor::DYN, tensor::DYN>;
  auto strided_data = StridedImageType(
          tensor::Vector3s(1, bits_per_pixel / 8, FreeImage_GetPitch(handle)),
          height, width, bits_per_pixel / 8
        );
  strided_data.storage() = mem::AllocatedStorage<uint8_t, mem::alloc::heap>(FreeImage_GetBits(handle));

  std::shared_ptr<Image::ImageData> image_data = std::make_shared<Image::ImageData>(strided_data.dims());
  *image_data = strided_data;

  ColorComponentMask color_component_masks[4];
  size_t color_component_num = 0;
  switch (FreeImage_GetColorType(handle))
  {
    case FIC_RGB:
    {
      color_component_masks[0] = ColorComponentMask(RED, FreeImage_GetRedMask(handle));
      color_component_masks[1] = ColorComponentMask(GREEN, FreeImage_GetGreenMask(handle));
      color_component_masks[2] = ColorComponentMask(BLUE, FreeImage_GetBlueMask(handle));
      color_component_num = 3;
      break;
    }
    case FIC_RGBALPHA:
    {
      color_component_masks[0] = ColorComponentMask(RED, FreeImage_GetRedMask(handle));
      color_component_masks[1] = ColorComponentMask(GREEN, FreeImage_GetGreenMask(handle));
      color_component_masks[2] = ColorComponentMask(BLUE, FreeImage_GetBlueMask(handle));
      static const union // TODO: replace with helper function
      {
        uint8_t bytes[4];
        uint32_t mask;
      } alpha_mask = {{0, 0, 0, 0xFF}};
      color_component_masks[3] = ColorComponentMask(ALPHA, alpha_mask.mask);
      color_component_num = 3;
      break;
    }
    default:
    {
      throw ResourceLoadException(rid, "Invalid color type");
    }
  }

  m_image = std::make_shared<Image>(image_data, ColorType(bits_per_pixel, color_component_masks, color_component_num));

  FreeImage_Unload(handle);
}

ImageFile::~ImageFile()
{
}

} // end of ns res