#include "Logging.h"

#include <boost/log/core/core.hpp>
#include <boost/log/expressions/formatters/date_time.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/sources/severity_channel_logger.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <fstream>
#include <ostream>



BOOST_LOG_ATTRIBUTE_KEYWORD(timestamp, "TimeStamp", boost::posix_time::ptime)



struct null_deleter
{
  typedef void result_type;

  template <typename T>
  void operator() (T*) const BOOST_NOEXCEPT {}
};
 
BOOST_LOG_GLOBAL_LOGGER_INIT(logger, boost::log::sources::severity_channel_logger_mt)
{
  boost::log::sources::severity_channel_logger_mt<boost::log::trivial::severity_level> logger;
  logger.add_attribute("TimeStamp", boost::log::attributes::local_clock());

  using text_sink = boost::log::sinks::synchronous_sink<boost::log::sinks::text_ostream_backend>;
  boost::shared_ptr<text_sink> sink = boost::make_shared<text_sink>();
  sink->locked_backend()->add_stream(boost::make_shared<std::ofstream>(LOGFILE));
  sink->locked_backend()->add_stream(boost::shared_ptr<std::ostream>(&std::clog, null_deleter()));
  sink->set_formatter(boost::log::expressions::stream
    << boost::log::expressions::format_date_time(timestamp, "%Y-%m-%d, %H:%M:%S.%f") << " "
    << "[" << boost::log::trivial::severity << "]"
    << " " << boost::log::expressions::smessage);
  //sink->set_filter(severity >= SEVERITY_THRESHOLD);

  boost::log::core::get()->add_sink(sink);

  return logger;
}