// µt8.h
// Support header for making C++ look more like mulang

#define fn [[nodiscard]] auto
#define fv void
#define typealias using
#define let const auto
#define var auto
typealias uint = unsigned;
template <typename T> using Ref = const T *_Nonnull;
