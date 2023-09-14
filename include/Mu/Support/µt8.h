// µt8.h
// Support header for making C++ look more like mulang.
// ie lets mutate C++ so it looks more modern

#ifndef _MUTATE_H_
#define _MUTATE_H_

#include <memory>
#include <vector>
#include <optional>

#define fn [[nodiscard]] auto
#define fv void
#define typealias using
#define let const auto
#define var auto
typealias uint = unsigned;
template <typename T> using Ref = const T *_Nonnull;
template <typename T> using MutableRef = T *_Nonnull;
template <typename T> using VectorRef = std::vector<Ref<T>>;
template <typename T> using OptionalRef = std::optional<Ref<T>>;
template <typename T> using OptionalOwnedRef =
  std::optional<std::unique_ptr<T>>;

template <typename T>
struct Defer {
  Defer() = delete;
  Defer(T t, std::function<void(T t)> cleanup): t(t), cleanup(cleanup) {}
  Defer(std::function<void(T t)> cleanup): cleanup(cleanup) {}
  virtual ~Defer() { cleanup(t); }
private:
  T t;
  std::function<void(T t)> cleanup;
};

#endif // _MUTATE_H_
