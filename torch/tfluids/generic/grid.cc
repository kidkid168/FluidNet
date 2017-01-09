// Copyright 2016 Google Inc, NYU.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This is a very, very barebones replication of the Manta grids. It's just
// so that we can more easily transfer KERNEL functions across.
// NOTE: THERE ARE NO CUDA IMPLEMENTATIONS OF THESE. You will need to replicate
// functionally any methods here as flat cuda functions.

#include <iostream>
#include <sstream>
#include <mutex>

class tfluids_(GridBase) {
public:
  explicit tfluids_(GridBase)(THTensor* grid, bool is_3d) :
      is_3d_(is_3d), tensor_(grid) {
    if (grid->nDimension != 4) {
      THError("GridBase: dim must be 4D (even if simulation is 2D).");
    }

    if (!is_3d_ && zsize() != 1) {
      THError("GridBase: 2D grid must have zsize == 1.");
    }
  }

  inline int32_t nchan() const { return tensor_->size[0]; }
  inline int32_t zsize() const { return tensor_->size[1]; }
  inline int32_t ysize() const { return tensor_->size[2]; }
  inline int32_t xsize() const { return tensor_->size[3]; }

  inline int32_t cstride() const { return tensor_->stride[0]; }
  inline int32_t zstride() const { return is_3d_ ? tensor_->stride[1] : 0; }
  inline int32_t ystride() const { return tensor_->stride[2]; }
  inline int32_t xstride() const { return tensor_->stride[3]; }

  inline bool is_3d() const { return is_3d_; }
  
private:
  // Note: Child classes should use getters!
  THTensor* const tensor_;
  const bool is_3d_;
  static std::mutex mutex_;

  // The indices i, j, k are x, y and z respectively.
  // Returns the index of the FIRST channel.
  inline int32_t index3d(int32_t i, int32_t j, int32_t k) const {
    // Bounds checks are slow, but the CUDA implementation is significantly
    // faster and does not have them. The C++ code is largely just a
    // framework to test the CUDA code.
    if (i >= xsize() || j >= ysize() || k >= zsize() ||
        i < 0 || j < 0 || k < 0) {
      std::lock_guard<std::mutex> lock(mutex_);
      std::stringstream ss;
      ss << "GridBase: index3d out of bounds: " << std::endl
         << "  (i, j, k) = (" << i << ", " << j
         << ", " << k << "), size = (" << xsize() << ", " << ysize() << ", "
         << zsize() << ")";
      std::cerr << ss.str() << std::endl << "Stack trace:" << std::endl;
      PrintStacktrace();
      std::cerr << std::endl;
      THError("GridBase: index3d out of bounds");
      return 0;
    }
    return i * xstride() + j * ystride() + k * zstride();
  }

  // The indices i, j, k, c are x, y, z and c respectively.
  // Returns the index of the FIRST channel.
  inline int32_t index4d(int32_t i, int32_t j, int32_t k, int32_t c) const {
    if (i >= xsize() || j >= ysize() || k >= zsize() || c >= nchan() ||
        i < 0 || j < 0 || k < 0 || c < 0) {
      std::lock_guard<std::mutex> lock(mutex_);
      std::stringstream ss;
      ss << "GridBase: index4d out of bounds:" << std::endl
         << "  (i, j, k, c) = (" << i << ", " << j
         << ", " << k << ", " << c << "), size = (" << xsize() << ", "
         << ysize() << ", " << zsize() << ", " << nchan() << ")";
      std::cerr << ss.str() << std::endl << "Stack trace:" << std::endl;
      PrintStacktrace();
      std::cerr << std::endl;
      THError("GridBase: index4d out of bounds");
      return 0;
    }
    return i * xstride() + j * ystride() + k * zstride() + c * cstride();
  }

protected:
  // Use operator() methods in child classes to get at data.
  inline real& data(int32_t i, int32_t j, int32_t k) {
    return THTensor_(data)(tensor_)[index3d(i, j, k)];
  }
  inline real data(int32_t i, int32_t j, int32_t k) const {
    return THTensor_(data)(tensor_)[index3d(i, j, k)];
  }
  inline real& data(int32_t i, int32_t j, int32_t k, int32_t c) {
    return THTensor_(data)(tensor_)[index4d(i, j, k, c)];
  }
  inline real data(int32_t i, int32_t j, int32_t k, int32_t c) const {
    return THTensor_(data)(tensor_)[index4d(i, j, k, c)];
  }
};

std::mutex tfluids_(GridBase)::mutex_;

class tfluids_(FlagGrid) : public tfluids_(GridBase) {
public:
  explicit tfluids_(FlagGrid)(THTensor* grid, bool is_3d) :
      tfluids_(GridBase)(grid, is_3d) {
    if (nchan() != 1) {
      PrintStacktrace();
      THError("FlagGrid: nchan must be 1 (scalar).");
    }
  }

  inline real& operator()(int32_t i, int32_t j, int32_t k) {
    return data(i, j, k);
  }
  
  inline real operator()(int32_t i, int32_t j, int32_t k) const {
    return data(i, j, k);
  }

  inline bool isFluid(int32_t i, int32_t j, int32_t k) const {
    return static_cast<int>(data(i, j, k)) & TypeFluid;
  }

private:
  // These are the same enum values used in Manta. We can't include grid.h
  // from Manta without pull in the entire library.
  enum CellType {
      TypeNone = 0,
      TypeFluid = 1,
      TypeObstacle = 2,
      TypeEmpty = 4,
      TypeInflow = 8,
      TypeOutflow = 16,
      TypeOpen = 32,
      TypeStick = 128,
      TypeReserved = 256,
      TypeZeroPressure = (1<<15)
  };
};

// Our RealGrid is supposed to be like Grid<Real> in Manta.
class tfluids_(RealGrid) : public tfluids_(GridBase) {
public:
  explicit tfluids_(RealGrid)(THTensor* grid, bool is_3d) :
      tfluids_(GridBase)(grid, is_3d) {
    if (nchan() != 1) {
      PrintStacktrace();
      THError("RealGrid: nchan must be 1 (scalar).");
    }
  }

  inline real& operator()(int32_t i, int32_t j, int32_t k) {
    return data(i, j, k);
  }

  inline real operator()(int32_t i, int32_t j, int32_t k) const {
    return data(i, j, k);
  };

  inline real getInterpolatedHi(const tfluids_(vec3)& pos, int order) const {
    switch (order) {
    case 1:
      return interpol(pos);
    case 2:
      PrintStacktrace();
      THError("getInterpolatedHi ERROR: cubic not supported.");
      // TODO(tompson): implement this.
      break;
    default:
      PrintStacktrace();
      THError("getInterpolatedHi ERROR: order not supported.");
      break;
    }
    return 0;
  }

  inline real interpol(const tfluids_(vec3)& pos) const {
    const real px = pos.x - static_cast<real>(0.5);
    const real py = pos.y - static_cast<real>(0.5);
    const real pz = pos.z - static_cast<real>(0.5);
    int32_t xi = (int32_t)px;
    int32_t yi = (int32_t)py;
    int32_t zi = (int32_t)pz;
    real s1 = px - static_cast<real>(xi);
    real s0 = static_cast<real>(1) - s1;
    real t1 = py - static_cast<real>(yi);
    real t0 = static_cast<real>(1) - t1;
    real f1 = pz - static_cast<real>(zi);
    real f0 = static_cast<real>(1) - f1;
    // Clamp to border.
    if (px < static_cast<real>(0)) {
      xi = 0;
      s0 = static_cast<real>(1);
      s1 = static_cast<real>(0);
    }
    if (py < static_cast<real>(0)) {
      yi = 0;
      t0 = static_cast<real>(1);
      t1 = static_cast<real>(0);
    }
    if (pz < static_cast<real>(0)) {
      zi = 0;
      f0 = static_cast<real>(1);
      f1 = static_cast<real>(0);
    }
    if (xi >= xsize() - 1) {
      xi = xsize() - 2;
      s0 = static_cast<real>(0);
      s1 = static_cast<real>(1);
    }
    if (yi >= ysize() - 1) {
      yi = ysize() - 2;
      t0 = static_cast<real>(0);
      t1 = static_cast<real>(1);
    }
    if (zsize() > 1) {
      if (zi >= zsize() - 1) {
        zi = zsize() - 2;
        f0 = static_cast<real>(0);
        f1 = static_cast<real>(1);
      }
    }

    if (is_3d()) {
      return ((data(xi, yi, zi) * t0 + data(xi, yi + 1, zi) * t1) * s0 
          + (data(xi + 1, yi, zi) * t0 +
             data(xi + 1, yi + 1, zi) * t1) * s1) * f0
          + ((data(xi, yi, zi + 1) * t0 + data(xi, yi + 1, zi + 1) * t1) * s0
          + (data(xi + 1, yi, zi + 1) * t0 +
             data(xi + 1, yi + 1, zi + 1) * t1) * s1) * f1;
    } else {
       return ((data(xi, yi, 0) * t0 + data(xi, yi + 1, 0) * t1) * s0
          + (data(xi + 1, yi, 0) * t0 +
             data(xi + 1, yi + 1, 0) * t1) * s1);
    }
  }
};

class tfluids_(MACGrid) : public tfluids_(GridBase) {
public:
  explicit tfluids_(MACGrid)(THTensor* grid, bool is_3d) :
      tfluids_(GridBase)(grid, is_3d) {
    if (nchan() != 2 && nchan() != 3) {
      PrintStacktrace();
      THError("MACGrid: input tensor size[0] is not 2 or 3");
    }
    if (!is_3d && zsize() != 1) {
      PrintStacktrace();
      THError("MACGrid: 2D tensor does not have zsize == 1");
    }
  }

  // Note: as per other functions, we DO NOT bounds check getCentered. You must
  // not call this method on the edge of the simulation domain.
  const tfluids_(vec3) getCentered(int32_t i, int32_t j, int32_t k) const {  
    const real x = static_cast<real>(0.5) * (data(i, j, k, 0) +
                                             data(i + 1, j, k, 0));
    const real y = static_cast<real>(0.5) * (data(i, j, k, 1) +
                                             data(i, j + 1, k, 1));
    const real z = !is_3d() ? static_cast<real>(0) :
        static_cast<real>(0.5) * (data(i, j, k, 2) +
                                  data(i, j, k + 1, 2));
    return tfluids_(vec3)(x, y, z);
  }

  inline const tfluids_(vec3) operator()(int32_t i, int32_t j,
                                         int32_t k) const {
    tfluids_(vec3) ret;
    ret.x = data(i, j, k, 0);
    ret.y = data(i, j, k, 1);
    ret.z = !is_3d() ? static_cast<real>(0) : data(i, j, k, 2);
    return ret;
  };

  inline real operator()(int32_t i, int32_t j, int32_t k, int32_t c) const {
    return data(i, j, k, c);
  };
};
