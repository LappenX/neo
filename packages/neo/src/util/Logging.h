#ifndef NEO_LOGGING_H
#define NEO_LOGGING_H

#include <Common.h>

#include <boost/log/trivial.hpp>
#include <boost/log/sources/severity_channel_logger.hpp>
#include <boost/log/sources/global_logger_storage.hpp>
#include <boost/iostreams/device/null.hpp>

#define LOGFILE "logfile.log"

BOOST_LOG_GLOBAL_LOGGER(logger, boost::log::sources::severity_channel_logger_mt<boost::log::trivial::severity_level>);

namespace logging {

using severity_level = boost::log::trivial::severity_level;
using channel = std::string;

namespace detail {

class LoggerMacro
{
public:
  LoggerMacro(severity_level severity_level, channel channel)
    : m_record(logger::get().open_record((boost::log::keywords::severity = severity_level, boost::log::keywords::channel = channel)))
    , m_stream(m_record)
  {
  }

  ~LoggerMacro()
  {
    m_stream.flush();
    logger::get().push_record(boost::move(m_record)); 
  }

  template <typename T>
  LoggerMacro& operator<<(const T& value)
  {
    if (m_record)
    {
      m_stream << value;
    }
    return *this;
  }

private:
  boost::log::record m_record;
  boost::log::record_ostream m_stream;
};

}

} // end of ns logging

#define LOG(sev, chan) logging::detail::LoggerMacro((boost::log::trivial::sev), (chan))

#endif // NEO_LOGGING_H