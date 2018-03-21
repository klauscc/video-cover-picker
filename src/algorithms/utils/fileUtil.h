#ifndef IQA_FILE_UTIL
#define IQA_FILE_UTIL

#include <boost/filesystem.hpp>
#include <iostream>
#include <vector>

namespace fs = boost::filesystem;

inline void getFileNamesInDir(const std::string path,
                              std::vector<std::string> &filenames) {
  fs::path p(path);
  for (auto i = fs::directory_iterator(p); i != fs::directory_iterator(); ++i) {
    if (!fs::is_directory(i->path())) {
      filenames.push_back(i->path().filename().string());
    } else {
      continue;
    }
  }
}

inline void getDirectoriesInDir(const std::string path,
                              std::vector<std::string> &filenames) {
  fs::path p(path);
  for (auto i = fs::directory_iterator(p); i != fs::directory_iterator(); ++i) {
    if (fs::is_directory(i->path())) {
      filenames.push_back(i->path().filename().string());
    } else {
      continue;
    }
  }
}
inline void splitFileName(const std::string &fileName, std::string &base,
                          std::string &ext) {
  size_t dot_idx = fileName.find_last_of("/\\.");
  base = fileName.substr(0, dot_idx);
  if (dot_idx != fileName.length()) {
    ext = fileName.substr(dot_idx + 1);
  }
}

#endif /* ifndef IQA_FILE_UTIL */
