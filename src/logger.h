#ifndef LOGGER_H
#define LOGGER_H
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>

namespace rwkvmobile {

enum {
    RWKV_LOG_LEVEL_DEBUG = 0,
    RWKV_LOG_LEVEL_INFO,
    RWKV_LOG_LEVEL_WARN,
    RWKV_LOG_LEVEL_ERROR,
};

void LOGI(const char *fmt, ...);
void LOGD(const char *fmt, ...);
void LOGW(const char *fmt, ...);
void LOGE(const char *fmt, ...);

std::string logger_get_log();

#define LOG_BUFFER_SIZE 1024

class Logger {
public:
    Logger() {
        _buffer.resize(LOG_BUFFER_SIZE);
    }
    ~Logger() = default;
    void log(const std::string &msg, const int level = RWKV_LOG_LEVEL_INFO);

    std::string get_log() {
        std::lock_guard<std::mutex> lock(_mutex);
        std::string log;
        for (int i = _buffer_start; i != _buffer_end; i = (i + 1) % LOG_BUFFER_SIZE) {
            log += _buffer[i];
        }
        return log;
    }

private:
    int _level = RWKV_LOG_LEVEL_INFO;

    // ring buffer
    std::vector<std::string> _buffer;
    int _buffer_start = 0;
    int _buffer_end = 0;
    std::mutex _mutex;
    std::condition_variable _condition;

    // log into ring buffer
    void _log(const std::string &msg);
};

}

#endif