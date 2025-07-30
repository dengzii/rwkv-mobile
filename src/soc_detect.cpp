#include "soc_detect.h"
#include "logger.h"
#include "commondef.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstring>

#ifdef __linux__
#include <dirent.h>
#include <vector>
#endif

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
    {536, "SM8550P", "8 Gen 2", "v73"},
    {557, "SM8650", "8 Gen 3", "v75"},
    {603, "QCS8550", "8 Gen 2", "v73"},
    {604, "QCM8550", "8 Gen 2", "v73"},
    {614, "SM8635", "8s Gen 3", "v73"},
    {642, "SM8635", "8s Gen 3", "v73"},
    {618, "SM8750", "8 Elite", "v79"},
    {639, "SM8750P", "8 Elite", "v79"},
    // TODO: add more
};

soc_detect::soc_detect() {
}

soc_detect::~soc_detect() {
}

int soc_detect::detect_platform() {
#ifndef _WIN32
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
#else
    // TODO
#endif

#if defined(_WIN32) && ENABLE_QNN_BACKEND
    // TODO: Detect this
    m_platform_type = PLATFORM_SNAPDRAGON;
    m_htp_arch = "v73";
    m_soc_partname = "SC8380";
    m_soc_name = "X Elite";
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

// From MNN
const std::vector<CPUGroup> get_cpu_groups() {
    std::vector<CPUGroup> cpu_groups;
#ifdef __linux__
    auto read_number = [](std::string &buffer) -> std::vector<int> {
        std::vector<int> numbers;
        std::stringstream ss(buffer);
        int number;
        while (ss >> number) {
            numbers.push_back(number);
        }
        return numbers;
    };

    do {
        DIR* root;
        std::string dir = "/sys/devices/system/cpu/cpufreq";
        if ((root = opendir(dir.c_str())) == NULL) {
            break;
        }
        CPUGroup group;
        struct dirent* ent;
        while ((ent = readdir(root)) != NULL) {
            if (ent->d_name[0] != '.') {
                std::string policyName = dir + "/" + ent->d_name;
                std::string cpus = policyName + "/affected_cpus";
                {
                    std::string buffer;
                    std::ifstream file(cpus, std::ios::binary);
                    if (file.is_open()) {
                        file.seekg(0, std::ios::end);
                        size_t size = file.tellg();
                        file.seekg(0, std::ios::beg);
                        buffer.resize(size);
                        file.read(reinterpret_cast<char*>(buffer.data()), size);
                        file.close();
                    }

                    group.ids = read_number(buffer);
                }
                if (group.ids.empty()) {
                    continue;
                }
                std::string minfreq = policyName + "/cpuinfo_min_freq";
                {
                    std::string buffer;
                    std::ifstream file(minfreq, std::ios::binary);
                    if (file.is_open()) {
                        file.seekg(0, std::ios::end);
                        size_t size = file.tellg();
                        file.seekg(0, std::ios::beg);
                        buffer.resize(size);
                        file.read(reinterpret_cast<char*>(buffer.data()), size);
                        file.close();
                    }
                    group.minFreq = read_number(buffer)[0];
                }
                std::string maxfreq = policyName + "/cpuinfo_max_freq";
                {
                    std::string buffer;
                    std::ifstream file(maxfreq, std::ios::binary);
                    if (file.is_open()) {
                        file.seekg(0, std::ios::end);
                        size_t size = file.tellg();
                        file.seekg(0, std::ios::beg);
                        buffer.resize(size);
                        file.read(reinterpret_cast<char*>(buffer.data()), size);
                        file.close();
                    }
                    group.maxFreq = read_number(buffer)[0];
                }
                cpu_groups.emplace_back(group);
            }
        }
        closedir(root);
        std::sort(cpu_groups.begin(), cpu_groups.end(), [](const CPUGroup& left, const CPUGroup& right) {
            return left.maxFreq > right.maxFreq;
        });
        // for (auto& group : cpu_groups) {
        //     std::string message = "CPU Group: [";
        //     for (int v=0; v<group.ids.size(); ++v) {
        //         message += " " + std::to_string(group.ids[v]) + " ";
        //     }
        //     message += "], " + std::to_string(group.minFreq) + " - " + std::to_string(group.maxFreq);
        //     LOGI("%s\n", message.c_str());
        // }
    } while (false);
#endif
    return cpu_groups;
}

} // namespace rwkvmobile
