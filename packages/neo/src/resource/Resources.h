#ifndef UTIL_RESOURCE_RESOURCES_H
#define UTIL_RESOURCE_RESOURCES_H

#include <Common.h>

#include <util/Property.h>
#include <util/Exception.h>
#include <util/Logging.h>

#include <unordered_map>
#include <string>
#include <memory>

namespace res {

class Manager;
class Resource;

class Identifier
{
public:
  virtual ~Identifier() {}

  virtual bool exists() const = 0;

  virtual size_t hash() const = 0;

  virtual bool operator==(const Identifier& other) const = 0;

  virtual Identifier* copy() const = 0;

  std::string toString() const
  {
    return "rid to " + toResourceString();
  }

  virtual std::string toResourceString() const = 0;
};

template <typename T>
class TypedIdentifier : public Identifier
{
public:
  virtual ~TypedIdentifier() {}
};





class Resource
{
public:
  Resource(const Identifier& rid)
    : m_rid(rid.copy())
  {
  }

  virtual ~Resource()
  {
    delete m_rid;
  }

  const Identifier& getRID() const
  {
    return *m_rid;
  }

  virtual std::string toString() const
  {
    return m_rid->toResourceString();
  }

private:
  Identifier* m_rid;
};

template <typename RID>
class RIDResource : public Resource
{
public:
  RIDResource(const RID& rid)
    : Resource(rid)
  {
  }

  virtual ~RIDResource() {}

  const RID& getRID() const
  {
    return static_cast<const RID&>(this->Resource::getRID());
  }
};





class ResourceNotFoundException : public Exception
{
public:
  ResourceNotFoundException(const Identifier& rid)
    : Exception("Failed to find " + rid.toResourceString())
  {
  }
};

class ResourceLoadException : public Exception
{
public:
  ResourceLoadException(const Identifier& rid)
    : Exception("Failed to load " + rid.toResourceString())
  {
  }

  ResourceLoadException(const Identifier& rid, std::string message)
    : Exception("Failed to load " + rid.toResourceString() + ": " + message)
  {
  }
};

class Manager
{
public:
  Manager()
  {
    LOG(info, "resource") << "Created resource manager";
  }

  virtual ~Manager()
  {
    LOG(info, "resource") << "Destroyed resource manager";
    // TODO: assert all resources are no longer in use, force deinit
  }

  template <typename TResourceType, typename... TRidConstructorArgs>
  TResourceType* get(TRidConstructorArgs&&... args)
  {
    using rid_t = typename TResourceType::rid_t;
    auto rid = std::make_shared<rid_t>(util::forward<TRidConstructorArgs>(args)...);
    auto resource_it = m_resources.find(rid);
    if (resource_it == m_resources.end())
    {
      if (!rid->exists())
      {
        throw ResourceNotFoundException(*rid);
      }
      else
      {
        LOG(debug, "res") << "Creating resource for " << rid->toResourceString();
        auto new_resource = std::make_shared<TResourceType>(*rid);
        m_resources[rid] = new_resource;
        return new_resource.get();
      }
    }
    else
    {
      return static_cast<TResourceType*>(resource_it->second.get());
    }
  }

private:
  struct Hasher
  {
    size_t operator()(const std::shared_ptr<Identifier>& rid) const
    {
      return rid->hash();
    }
  };

  struct EqualTest
  {
    size_t operator()(const std::shared_ptr<Identifier>& rid1, const std::shared_ptr<Identifier>& rid2) const
    {
      return rid1->operator==(*rid2);
    }
  };

  std::unordered_map<std::shared_ptr<Identifier>, std::shared_ptr<Resource>, Hasher, EqualTest> m_resources;

  NO_COPYING(Manager)
};

} // end of ns res

#endif // UTIL_RESOURCE_RESOURCES_H