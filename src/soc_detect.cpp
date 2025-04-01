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
    {475, "SM7325", "778", "v68"},
    {439, "SM8350", "888", "v68"},
    {457, "SM8450", "8 Gen 1", "v69"},
    {480, "SM8450_2", "8 Gen 1", "v69"},
    {482, "SM8450_3", "8 Gen 1", "v69"},
    {497, "QCM6490", "QCM6490", "v68"},
    {498, "QCS6490", "QCS6490", "v68"},
    {530, "SM8475", "8+ Gen 1", "v69"},
    {531, "SM8475P", "8+ Gen 1", "v69"},
    {540, "SM8475_2", "8+ Gen 1", "v69"},
    {519, "SM8550", "8 Gen 2", "v73"},
    {557, "SM8650", "8 Gen 3", "v75"},
    {614, "SM8635", "8s Gen 3", "v73"},
    {618, "SM8750", "8 Elite", "v79"}
    // TODO: add more
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
                m_htp_arch = snapdragon_soc_ids[i].htp_arch;
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

const char * soc_detect::get_htp_arch() {
    return m_htp_arch;
}

} // namespace rwkvmobile
