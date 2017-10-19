#ifndef UTIL_RESOURCE_IMAGERESOURCE_H
#define UTIL_RESOURCE_IMAGERESOURCE_H

#include <Common.h>

#include <util/Logging.h>
#include "FileResource.h"
#include <tensor/Tensor.h>

#include <FreeImage.h>
#include <memory>



namespace res {

namespace freeimage {

void init();
void deinit();

} // end of ns freeimage





enum ColorComponent
{
  RED,
  BLUE,
  GREEN,
  ALPHA
};

class ColorComponentMask
{
public:
  ColorComponentMask(ColorComponent color_component, size_t mask)
    : m_color_component(color_component)
    , m_mask(mask)
  {
  }

  ColorComponentMask()
    : m_color_component(RED)
    , m_mask(0)
  {
  }

  ColorComponent getColorComponent() const
  {
    return m_color_component;
  }

  size_t getMask() const
  {
    return m_mask;
  }

private:
  ColorComponent m_color_component;
  size_t m_mask;
};

class ColorType
{
public:
  template <typename... TColorComponentMasks>
  ColorType(size_t bits_per_pixel, TColorComponentMasks&&... color_component_masks)
    : m_bits_per_pixel(bits_per_pixel)
    , m_component_num(sizeof...(color_component_masks))
    , m_component_masks(new ColorComponentMask[sizeof...(color_component_masks)]{util::forward<TColorComponentMasks>(color_component_masks)...})
  {
  }

  ColorType(size_t bits_per_pixel, ColorComponentMask* component_masks, size_t component_num)
    : m_bits_per_pixel(bits_per_pixel)
    , m_component_num(component_num)
    , m_component_masks(new ColorComponentMask[component_num])
  {
    for (size_t i = 0; i < m_component_num; i++)
    {
      m_component_masks[i] = component_masks[i];
    }
  }

  ColorType(const ColorType& other)
    : m_bits_per_pixel(other.m_bits_per_pixel)
    , m_component_num(other.m_component_num)
    , m_component_masks(new ColorComponentMask[m_component_num])
  {
    for (size_t i = 0; i < m_component_num; i++)
    {
      m_component_masks[i] = other.m_component_masks[i];
    }
  }

  ColorType& operator=(const ColorType& other)
  {
    delete[] m_component_masks;
    m_bits_per_pixel = other.m_bits_per_pixel;
    m_component_num = other.m_component_num;
    m_component_masks = new ColorComponentMask[m_component_num];
    for (size_t i = 0; i < m_component_num; i++)
    {
      m_component_masks[i] = other.m_component_masks[i];
    }
  }

  ~ColorType()
  {
    delete[] m_component_masks;
  }

  size_t getBitsPerPixel() const
  {
    return m_bits_per_pixel;
  }

  size_t getMask(ColorComponent component) const
  {
    for (size_t i = 0; i < m_component_num; i++)
    {
      if (m_component_masks[i].getColorComponent() == component)
      {
        return m_component_masks[i].getMask();
      }
    }
    return 0;
  }

private:
  size_t m_bits_per_pixel;
  size_t m_component_num;
  ColorComponentMask* m_component_masks;
};
/*
// TODO: check endianess
const ColorType RGB (24, ColorComponent Mask(RED,   0x00FF0000),
                         ColorComponentM ask(GREEN, 0x0000FF00),
                         ColorComponent Mask(BLUE,  0x000000FF));
const ColorType RGBA(32, ColorComponentMask(RED,  0x00FF0000),
                         ColorComponentMask(GREEN, 0x0000FF00),
                         ColorComponentMask(BLUE,  0x000000FF),
                         ColorComponentMask(ALPHA, 0xFF000000));
*/



class Image
{
public:
  using ImageData = tensor::DenseAllocStorageTensor<uint8_t, mem::alloc::heap, tensor::RowMajorIndexStrategy, tensor::DYN, tensor::DYN, tensor::DYN>;

  Image(std::shared_ptr<ImageData> data, ColorType color_type)
    : m_data(data)
    , m_color_type(color_type)
  {
  }

  void save(const boost::filesystem::path& file, FREE_IMAGE_FORMAT format);

  std::shared_ptr<ImageData> getData() const
  {
    return m_data;
  }

  ColorType getColorType() const
  {
    return m_color_type;
  }

private:
  std::shared_ptr<ImageData> m_data;
  ColorType m_color_type;
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
  
  ImageFile(const ImageFileRID& rid);
  virtual ~ImageFile();

  std::shared_ptr<Image> getImage() const
  {
    return m_image;
  }

private:
  std::shared_ptr<Image> m_image;

  NO_COPYING(ImageFile)
};

} // res

#endif // UTIL_RESOURCE_IMAGERESOURCE_H