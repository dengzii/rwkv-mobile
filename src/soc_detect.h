#pragma once

namespace rwkvmobile {

enum platform_type {
    PLATFORM_SNAPDRAGON, // lets add snapdragon support first
    PLATFORM_UNKNOWN,
};

struct snapdragon_soc_id {
    int soc_id;
    const char * soc_partname;
    const char * soc_name;
};

class soc_detect {
    public:
        soc_detect();
        ~soc_detect();

        int detect_platform();

        platform_type get_platform_type();
        const char * get_platform_name();
        const char * get_soc_name();
        const char * get_soc_partname();
    private:
        platform_type platform_type = PLATFORM_UNKNOWN;
        int soc_id = 0;
        const char * soc_name = "Unknown";
        const char * soc_partname = "Unknown";
};

} // namespace rwkvmobile