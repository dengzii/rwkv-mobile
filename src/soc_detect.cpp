#include "soc_detect.h"
#include "logger.h"
#include "commondef.h"
#include <fstream>

#include <iostream>

namespace rwkvmobile {

const char * platform_name[] = {
    "Snapdragon",
    "Unknown",
};

snapdragon_soc_id snapdragon_soc_ids[] = {
    {457, "SM8450", "8 Gen 1"},
    {480, "SM8450_2", "8 Gen 1"},
    {482, "SM8450_3", "8 Gen 1"},
    {530, "SM8475", "8+ Gen 1"},
    {531, "SM8475P", "8+ Gen 1"},
    {540, "SM8475_2", "8+ Gen 1"},
    {519, "SM8550", "8 Gen 2"},
    {557, "SM8650", "8 Gen 3"},
    // TODO: add more
    // TODO: find socid for 8s Gen3
};

soc_detect::soc_detect() {
}

soc_detect::~soc_detect() {
}

int soc_detect::detect_platform() {
#if defined(__ANDROID__)
    std::ifstream file("/sys/devices/soc0/family");
    std::string tmp;
    if (file.is_open()) {
        file >> tmp;
        file.close();
    }

    if (tmp == "Snapdragon") {
        m_platform_type = PLATFORM_SNAPDRAGON;
    } else {
        m_platform_type = PLATFORM_UNKNOWN;
    }

    if (m_platform_type == PLATFORM_SNAPDRAGON) {
        std::ifstream file_soc_id("/sys/devices/soc0/soc_id");
        if (file_soc_id.is_open()) {
            file_soc_id >> m_soc_id;
            file_soc_id.close();
        }

        for (int i = 0; i < sizeof(snapdragon_soc_ids) / sizeof(snapdragon_soc_ids[0]); i++) {
            if (snapdragon_soc_ids[i].soc_id == m_soc_id) {
                m_soc_name = snapdragon_soc_ids[i].soc_name;
                m_soc_partname = snapdragon_soc_ids[i].soc_partname;
                break;
            }
        }
    }
#endif
    return RWKV_SUCCESS;
}

platform_type soc_detect::get_platform_type() {
    return m_platform_type; 
}

const char * soc_detect::get_platform_name() {
    return platform_name[m_platform_type];
}

const char * soc_detect::get_soc_name() {
    return m_soc_name;
}

const char * soc_detect::get_soc_partname() {
    return m_soc_partname;
}

} // namespace rwkvmobile
