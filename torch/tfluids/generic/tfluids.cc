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

#ifndef TH_GENERIC_FILE
  #define TH_GENERIC_FILE "generic/tfluids.cc"
#else

#include <assert.h>
#include <memory>

#include "generic/vec3.cc"
#include "generic/grid.cc"

#ifdef BUILD_GL_FUNCS
  #if defined (__APPLE__) || defined (OSX)
    #include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
    #include <OpenGL/glext.h>
  #else
    #include <GL/gl.h>
  #endif

  #ifndef GLUT_API_VERSION
    #if defined(macintosh) || defined(__APPLE__) || defined(OSX)
      #include <GLUT/glut.h>
    #elif defined (__linux__) || defined (UNIX) || defined(WIN32) || defined(_WIN32)
      #include "GL/glut.h"
    #endif
  #endif
#endif

// *****************************************************************************
// LUA MAIN ENTRY POINT FUNCTIONS
// *****************************************************************************

inline real SemiLagrange(tfluids_(FlagGrid)& flags, tfluids_(MACGrid)& vel,
                         tfluids_(RealGrid)& src, real dt, bool is_levelset,
                         int order_space, int32_t i, int32_t j, int32_t k) {
  const real p5 = static_cast<real>(0.5);
  tfluids_(vec3) pos =
      (tfluids_(vec3)((real)i + p5, (real)j + p5, (real)k + p5) -
       vel.getCentered(i,j,k) * dt);
  return src.getInterpolatedHi(pos, order_space);
}

inline real MacCormackCorrect(tfluids_(FlagGrid)& flags, const real old,
                              const real fwd, const real bwd,
                              const real strength, bool is_levelset,
                              int32_t i, int32_t j, int32_t k) {
  real dst = fwd;

  if (flags.isFluid(i, j, k)) {
    // Only correct inside fluid region.
    dst += strength * 0.5 * (old - bwd);
  }
  return dst;
}

static int tfluids_(Main_advectScalar)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack. We also treat 2D advection as 3D (with depth = 1) and
  // no 'w' component for velocity.
  real dt = static_cast<real>(lua_tonumber(L, 1));
  THTensor* tensor_s =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));
  THTensor* tensor_s_tmp =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 5, torch_Tensor));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 6));
  const std::string method = static_cast<std::string>(lua_tostring(L, 7));
  const bool open_bounds = static_cast<bool>(lua_toboolean(L, 8));
  const int32_t boundary_width = static_cast<int32_t>(lua_tointeger(L, 9));
  THTensor* tensor_s_dst =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 10, torch_Tensor));

  // Note: all the checks for sizes and dim are done in init.lua, we're going
  // to do a few here just because we're paranoid.
  if (tensor_u->nDimension != 4) {
    luaL_error(L, "u is not 4D.");
  }
  if (is_3d && tensor_u->size[0] != 3) {
    luaL_error(L, "3D u does not have 3 channels!");
  }
  if (!is_3d && tensor_u->size[0] != 2) {
    luaL_error(L, "2D u does not have 2 channels!");
  }
  const int32_t xdim = static_cast<int32_t>(tensor_u->size[3]);
  const int32_t ydim = static_cast<int32_t>(tensor_u->size[2]);
  const int32_t zdim = static_cast<int32_t>(tensor_u->size[1]);

  if (!is_3d && zdim != 1) {
    luaL_error(L, "2D u does not have unit z dimension!");
  }

  // Wrap the data round our structs (no copy or alloc).
  tfluids_(FlagGrid) flags(tensor_flags, is_3d);
  tfluids_(MACGrid) vel(tensor_u, is_3d);
  tfluids_(RealGrid) src(tensor_s, is_3d);
  tfluids_(RealGrid) dst(tensor_s_dst, is_3d);
  tfluids_(RealGrid) fwd(tensor_s_tmp, is_3d);
  
  if (method != "maccormack" && method != "euler") {
    luaL_error(L, "advectScalar method is not supported.");
  }

  const int32_t order = method == "euler" ? 1 : 2;
  const bool is_levelset = false;  // We never advect them.
  const int order_space = 1;

  int32_t k, j, i;
  const int32_t bnd = 1;
#pragma omp parallel for collapse(3) private(k, j, i)
  for(k = 0; k < zdim; k++) {
    for(j = 0; j < ydim; j++) {
      for(i = 0; i < xdim; i++) {
        if (i < bnd || i > xdim - 1 - bnd ||
            j < bnd || j > ydim - 1 - bnd ||
            (is_3d && (k < bnd || k > zdim - 1 - bnd))) {
          dst(i, j, k) = 0;  // Manta zeros stuff on the border.
          continue;
        }

        // Forward step.
        const real val = SemiLagrange(flags, vel, src, dt, is_levelset,
                                      order_space, i, j, k); 

        if (order == 1) {
          dst(i, j, k) = val;  // Store in the output array
        } else {
          fwd(i, j, k) = val;  // Store in the fwd array.
        }
      }
    }
  }

  if (order == 1) {
    // We're done. The forward Euler step is already in the output array.
    return 0;
  }

  // Otherwise we need to do the backwards step (which is a SemiLagrange step
  // on the forward data - hence we needed to finish the above loops before
  // moving on).
  const real strength = static_cast<real>(1);
#pragma omp parallel for collapse(3) private(k, j, i)
  for(k = 0; k < zdim; k++) { 
    for(j = 0; j < ydim; j++) { 
      for(i = 0; i < xdim; i++) {
        if (i < bnd || i > xdim - 1 - bnd ||
            j < bnd || j > ydim - 1 - bnd ||
            (is_3d && (k < bnd || k > zdim - 1 - bnd))) {
          continue; 
        } 

        const real orig = src(i, j, k);
        const real fwd_val = fwd(i, j, k);

        // Backwards step.
        const real bwd = SemiLagrange(flags, vel, fwd, -dt, is_levelset,
                                      order_space, i, j, k);
        
        // compute correction.
        // TODO(tompson): This kernel is on the entire domain in Manta but not
        // here.
        const real val = MacCormackCorrect(flags, orig, fwd_val, bwd, strength,
                                           is_levelset, i, j, k); 

        // clamp vals.
        // TODO(tompson): This kernel is on the entire domain in Manta but not
        // here.
//        MacCormackClamp(flags, vel, newGrid, orig, fwd, dt);
        dst(i, j, k) = val;
      }
    }
  }

  return 0;  // Recall: number of return values on the lua stack.
}
/*
static int tfluids_(Main_advectVel)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack. We also treat 2D advection as 3D (with depth = 1) and
  // no 'w' component for velocity.
  real dt = static_cast<real>(lua_tonumber(L, 1));
  THTensor* u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* geom =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* u_dst =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));
  const std::string method = static_cast<std::string>(lua_tostring(L, 5));

  bool two_dim = u->size[0] == 2;
  const int32_t xdim = static_cast<int32_t>(geom->size[2]);
  const int32_t ydim = static_cast<int32_t>(geom->size[1]);
  const int32_t zdim = static_cast<int32_t>(geom->size[0]);

  // Get pointers to the tensor data.
  const real* u_data = THTensor_(data)(u);
  const real* geom_data = THTensor_(data)(geom);
  real* u_dst_data = THTensor_(data)(u_dst);

  const real* ux_data = &u_data[0 * xdim * ydim * zdim];
  const real* uy_data = &u_data[1 * xdim * ydim * zdim];
  const real* uz_data = two_dim ? nullptr : &u_data[2 * xdim * ydim * zdim];

  real* ux_dst_data = &u_dst_data[0 * xdim * ydim * zdim];
  real* uy_dst_data = &u_dst_data[1 * xdim * ydim * zdim];
  real* uz_dst_data = two_dim ? nullptr : &u_dst_data[2 * xdim * ydim * zdim];

  // Finally, call the advection routine.
  Int3 dim;   
  dim.x = xdim;
  dim.y = ydim;
  dim.z = zdim;
  if (method == "rk2") {
    tfluids_(Main_advectVelRK2)(dt, ux_data, uy_data, uz_data,
                                geom_data, dim, ux_dst_data, uy_dst_data,
                                uz_dst_data);
  } else if (method == "euler") {
    tfluids_(Main_advectVelEuler)(dt, ux_data, uy_data, uz_data,
                                  geom_data, dim, ux_dst_data, uy_dst_data,
                                  uz_dst_data);
  } else if (method == "maccormack") {
    luaL_error(L, "maccormack not yet implemented.");
  } else {
    luaL_error(L, "Invalid advection method.");
  }
 
  return 0;  // Number of return values on the lua stack.
}

static int tfluids_(Main_vorticityConfinement)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack. We also treat 2D as 3D (with depth = 1) and
  // no 'w' component for velocity.
  const real dt = static_cast<real>(lua_tonumber(L, 1));
  const real scale = static_cast<real>(lua_tonumber(L, 2));
  THTensor* u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* geom =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));
  THTensor* curl =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 5, torch_Tensor));
  THTensor* mag_curl =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 6, torch_Tensor));

  if (u->nDimension != 4 || geom->nDimension != 3 ||
      mag_curl->nDimension != 3) {
    luaL_error(L, "Incorrect input sizes.");
  }

  bool two_dim = u->size[0] == 2;
  const int32_t xdim = static_cast<int32_t>(geom->size[2]);
  const int32_t ydim = static_cast<int32_t>(geom->size[1]);
  const int32_t zdim = static_cast<int32_t>(geom->size[0]);

  if (two_dim && curl->nDimension != 3) {
     luaL_error(L, "Bad curl size.");
  }
  if (!two_dim && curl->nDimension != 4) {
     luaL_error(L, "Bad curl size.");
  }

  if (two_dim && zdim != 1) {
    luaL_error(L, "Incorrect input sizes.");
  }

  if (!two_dim && u->size[0] != 3) {
    luaL_error(L, "Incorrect input sizes.");
  }

  // Get pointers to the tensor data.
  real* u_data = THTensor_(data)(u);
  const real* geom_data = THTensor_(data)(geom);
  real* curl_data = THTensor_(data)(curl);
  real* mag_curl_data = THTensor_(data)(mag_curl);

  real* ux_data = &u_data[0 * xdim * ydim * zdim];
  real* uy_data = &u_data[1 * xdim * ydim * zdim];
  real* uz_data = two_dim ? nullptr : &u_data[2 * xdim * ydim * zdim];

  real* curl_u = &curl_data[0 * xdim * ydim * zdim];
  real* curl_v = two_dim ? nullptr : &curl_data[1 * xdim * ydim * zdim];
  real* curl_w = two_dim ? nullptr : &curl_data[2 * xdim * ydim * zdim];

  // Finally, call the vorticity confinement routine.
  Int3 dim;
  dim.x = xdim;
  dim.y = ydim;
  dim.z = zdim;
  tfluids_(Main_vorticityConfinement)(dt, scale, ux_data, uy_data, uz_data,
                                      geom_data, dim, curl_u, curl_v, curl_w,
                                      mag_curl_data);
 
  return 0;  // Number of return values on the lua stack.
}

// Expose the getScalarInterpValue to the call to the lua stack for debugging.
static int tfluids_(Main_interpField)(lua_State *L) {
  THTensor* field =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* geom =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* pos =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  const bool sample_into_geom = static_cast<bool>(lua_toboolean(L, 4));

  if (field->nDimension != 3 || geom->nDimension != 3) {
    luaL_error(L, "Input field and geom should be 3D.");
  }
  if (pos->nDimension != 1 || pos->size[0] != 3) {
    luaL_error(L, "pos should be 1D and size 3.");
  }
  const int32_t zdim = field->size[0];
  const int32_t ydim = field->size[1];
  const int32_t xdim = field->size[2];
  Int3 dims;
  dims.x = xdim;
  dims.y = ydim;
  dims.z = zdim;

  const real* field_data = THTensor_(data)(field);
  const real* geom_data = THTensor_(data)(geom);
  const real* pos_data = THTensor_(data)(pos);

  tfluids_(vec3) interp_pos;
  interp_pos.x = pos_data[0];
  interp_pos.y = pos_data[1];
  interp_pos.z = pos_data[2];

  const real ret_val = tfluids_(Main_GetScalarInterpValue)(
    field_data, geom_data, interp_pos, dims, sample_into_geom);

  lua_pushnumber(L, static_cast<double>(ret_val));
  return 1;
}

*/

static int tfluids_(Main_drawVelocityField)(lua_State *L) {
/*
#ifdef BUILD_GL_FUNCS
  THTensor* u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));

  const bool flip_y = static_cast<bool>(lua_toboolean(L, 2));

  if (u->nDimension != 5) {
    luaL_error(L, "Input vector field should be 5D.");
  }
  const int32_t nbatch = u->size[0];
  const int32_t nchan = u->size[1];
  const int32_t zdim = u->size[2];
  const int32_t ydim = u->size[3];
  const int32_t xdim = u->size[4];
  const bool two_dim = nchan == 2;
  if (two_dim && zdim != 1) {
    luaL_error(L, "Unexpected zdim for 2D vector field.");
  }

  const real* u_data = THTensor_(data)(u);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glBegin(GL_LINES);
  for (int32_t b = 0; b < nbatch; b++) {
    for (int32_t z = 0; z < zdim; z++) {
      for (int32_t y = 0; y < ydim; y++) {
        for (int32_t x = 0; x < xdim; x++) {
          const int32_t ux_index = b * nchan * zdim * ydim * xdim +
              z * ydim * xdim + y * xdim + x;
          real ux = u_data[ux_index];
          real uy = u_data[ux_index + xdim * ydim * zdim];
          real uz = two_dim ? static_cast<real>(0) :
              u_data[ux_index + 2 * xdim * ydim * zdim];
          // Velocity is in grids / second. But we need coordinates in [0, 1].
          ux = ux / static_cast<real>(xdim - 1);
          uy = uy / static_cast<real>(ydim - 1);
          if (!two_dim) {
            uz = uz / static_cast<real>(zdim - 1);
          }
          // Same for position.
          real px = static_cast<real>(x) / static_cast<real>(xdim - 1);
          real py = static_cast<real>(y) / static_cast<real>(ydim - 1);
          real pz = two_dim ? static_cast<real>(0) :
              (static_cast<real>(z) / static_cast<real>(zdim - 1));
          py = flip_y ? py : static_cast<real>(1) - py;
          uy = flip_y ? -uy : uy;
          glColor4f(0.7f, 0.0f, 0.0f, 1.0f);
          glVertex3f(static_cast<float>(px),
                     static_cast<float>(py),
                     static_cast<float>(pz));
          glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
          glVertex3f(static_cast<float>(px + ux),
                     static_cast<float>(py - uy),
                     static_cast<float>(pz + uz));
        }
      }
    }
  }
  glEnd();
#else
  luaL_error(L, "tfluids compiled without preprocessor def BUILD_GL_FUNCS.");
#endif
  return 0;
  */
  std::cout << "Error: need to sample center of MAC grid." << std::endl;
  exit(-1);
}

static int tfluids_(Main_loadTensorTexture)(lua_State *L) {
#ifdef BUILD_GL_FUNCS
  THTensor* im_tensor =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  if (im_tensor->nDimension != 2 && im_tensor->nDimension != 3) {
    luaL_error(L, "Input should be 2D or 3D.");
  }
  const int32_t tex_id = static_cast<int32_t>(luaL_checkinteger(L, 2));
  if (!lua_isboolean(L, 3)) {
    luaL_error(L, "3rd argument to loadTensorTexture should be boolean.");
  }
  const bool filter = lua_toboolean(L, 3);
  if (!lua_isboolean(L, 4)) {
    luaL_error(L, "4rd argument to loadTensorTexture should be boolean.");
  }
  const bool flip_y = lua_toboolean(L, 4);

  const bool grey = im_tensor->nDimension == 2;
  const int32_t nchan = grey ? 1 : im_tensor->size[0];
  const int32_t h = grey ? im_tensor->size[0] : im_tensor->size[1];
  const int32_t w = grey ? im_tensor->size[1] : im_tensor->size[2];

  if (nchan != 1 && nchan != 3) {
    luaL_error(L, "Only 3 or 1 channels is supported.");
  }

  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, tex_id);

  if (filter) {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  } else {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  }
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

  const real* im_tensor_data = THTensor_(data)(im_tensor);

  // We need to either: a) swizzle the RGB data, b) convert from double to float
  // or c) convert to RGB greyscale for single channel textures). For c) we
  // could use a swizzle mask, but this complicates alpha blending, so it's
  // easier to just convert always at the cost of a copy (which is parallel and
  // fast).

  std::unique_ptr<float[]> fdata(new float[h * w * 4]);
  int32_t c, u, v;
#pragma omp parallel for private(v, u, c) collapse(3)
  for (v = 0; v < h; v++) {
    for (u = 0; u < w; u++) {
      for (c = 0; c < 4; c++) {
        if (c == 3) {
          // OpenMP requires perfectly nested loops, so we need to include the
          // alpha chan set like this.
          fdata[v * 4 * w + u * 4 + c] = 1.0f;
        } else {
          const int32_t csrc = (c < nchan) ? c : 0;
          const int32_t vsrc = flip_y ? (h - v - 1) : v;
          fdata[v * 4 * w + u * 4 + c] =
              static_cast<float>(im_tensor_data[csrc * w * h + vsrc * w + u]);
        }
      }
    }
  }

  const GLint level = 0;
  const GLint internalformat = GL_RGBA32F;
  const GLint border = 0;
  const GLenum format = GL_RGBA;
  const GLenum type = GL_FLOAT;
  glTexImage2D(GL_TEXTURE_2D, level, internalformat, w, h, border,
               format, type, fdata.get());
#else
  luaL_error(L, "tfluids compiled without preprocessor def BUILD_GL_FUNCS.");
#endif
  return 0;
}

/*
static int tfluids_(Main_calcVelocityUpdate)(lua_State *L) {
  THTensor* u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* u_div =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* p =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* geom =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));


  // TODO(tompson): Finish this.
  #error finish calcVelocityUpdate

  // Just do a basic dim assert, everything else goes in the lua code.
  if (delta_u->nDimension != 5 || p->nDimension != 4 || geom->nDimension != 4) {
    luaL_error(L, "Incorrect dimensions. Expect delta_u: 5, p: 4, geom: 4.");
  }
  const int32_t nbatch = delta_u->size[0];
  const int32_t nuchan = delta_u->size[1];
  const int32_t zdim = delta_u->size[2];
  const int32_t ydim = delta_u->size[3];
  const int32_t xdim = delta_u->size[4];

  real* delta_u_data = THTensor_(data)(delta_u);
  const real* geom_data = THTensor_(data)(geom);
  const real* p_data = THTensor_(data)(p);

  int32_t b, c, z, y, x;
#pragma omp parallel for private(b, c, z, y, x) collapse(5)
  for (b = 0; b < nbatch; b++) {
    for (c = 0; c < nuchan; c++) {
      for (z = 0; z < zdim; z++) {
        for (y = 0; y < ydim; y++) {
          for (x = 0; x < xdim; x++) {
            real* cur_delta_u = &delta_u_data[b * nuchan * zdim * ydim * xdim];
            const real* cur_geom = &geom_data[b * zdim * ydim * xdim];
            const real* cur_p = &p_data[b * zdim * ydim * xdim];
            const int32_t pos[3] = {x, y, z};
            const int32_t size[3] = {xdim, ydim, zdim};

            tfluids_(Main_calcVelocityUpdateAlongDim)(
                cur_delta_u, cur_p, cur_geom, pos, size, c);
          }
        }
      }
    }
  }
  return 0;
}

static int tfluids_(Main_calcVelocityUpdateBackward)(lua_State *L) {
  THTensor* grad_p =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* u =
        reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* u_div =
        reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* p =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));
  THTensor* geom =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 5, torch_Tensor));
  THTensor* grad_output =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 6, torch_Tensor));



  // TODO(tompson): Finish this.
  #error finish calcVelocityUpdateBackward


  // Just do a basic dim assert, everything else goes in the lua code.
  if (grad_output->nDimension != 5 || p->nDimension != 4 ||
      geom->nDimension != 4 || grad_p->nDimension != 4) {
    luaL_error(L, "Incorrect dimensions.");
  }
  const int32_t nbatch = grad_p->size[0];
  const int32_t zdim = grad_p->size[1];
  const int32_t ydim = grad_p->size[2];
  const int32_t xdim = grad_p->size[3];
  const int32_t nuchan = grad_output->size[1];

  real* grad_p_data = THTensor_(data)(grad_p);
  const real* geom_data = THTensor_(data)(geom);
  const real* p_data = THTensor_(data)(p);
  const real* grad_output_data = THTensor_(data)(grad_output);

  // We will be accumulating gradient contributions into the grad_p tensor, so
  // we first need to zero it.
  THTensor_(zero)(grad_p);

  // TODO(tompson): I have implemented the following function as a scatter
  // operation. However this requires the use of #pragma omp atomic everywhere.
  // Instead re-write the inner loop code to perform a gather op to avoid the
  // atomic locks.
  int32_t b, c, z, y, x;
#pragma omp parallel for private(b, c, z, y, x) collapse(5)
  for (b = 0; b < nbatch; b++) {
    for (c = 0; c < nuchan; c++) {
      for (z = 0; z < zdim; z++) {
        for (y = 0; y < ydim; y++) {
          for (x = 0; x < xdim; x++) {
            const real* cur_grad_output =
                &grad_output_data[b * nuchan * zdim * ydim * xdim];
            const real* cur_geom = &geom_data[b * zdim * ydim * xdim];
            const real* cur_p = &p_data[b * zdim * ydim * xdim];
            real* cur_grad_p = &grad_p_data[b * zdim * ydim * xdim];
            const int32_t pos[3] = {x, y, z};
            const int32_t size[3] = {xdim, ydim, zdim};

            tfluids_(Main_calcVelocityUpdateAlongDimBackward)(
                cur_grad_p, cur_p, cur_geom, cur_grad_output, pos, size, c);
          }
        }
      }
    }
  }
  return 0;
}

static int tfluids_(Main_calcVelocityDivergence)(lua_State *L) {
  THTensor* u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* geom =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* u_div =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));

  // Just do a basic dim assert, everything else goes in the lua code.
  if (u->nDimension != 5 || u_div->nDimension != 4 || geom->nDimension != 4) {
    luaL_error(L, "Incorrect dimensions.");
  }
  const int32_t nbatch = u->size[0];
  const int32_t nuchan = u->size[1];
  const int32_t zdim = u->size[2];
  const int32_t ydim = u->size[3];
  const int32_t xdim = u->size[4];

  real* u_div_data = THTensor_(data)(u_div);
  const real* geom_data = THTensor_(data)(geom);
  const real* u_data = THTensor_(data)(u);

  int32_t b, z, y, x;
#pragma omp parallel for private(b, z, y, x) collapse(4)
  for (b = 0; b < nbatch; b++) {
    for (z = 0; z < zdim; z++) {
      for (y = 0; y < ydim; y++) {
        for (x = 0; x < xdim; x++) {
          const real* cur_u = &u_data[b * nuchan * zdim * ydim * xdim];
          const real* cur_geom = &geom_data[b * zdim * ydim * xdim];
          real* cur_u_div = &u_div_data[b * zdim * ydim * xdim];
          const int32_t pos[3] = {x, y, z};
          const int32_t size[3] = {xdim, ydim, zdim};

          tfluids_(Main_calcVelocityDivergenceCell)(
              cur_u, cur_geom, cur_u_div, pos, size, nuchan);
        }
      }
    }
  }
  return 0;
}

static int tfluids_(Main_calcVelocityDivergenceBackward)(lua_State *L) {
  THTensor* grad_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* geom =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* grad_output = 
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));

  // Just do a basic dim assert, everything else goes in the lua code.
  if (u->nDimension != 5 || grad_u->nDimension != 5 || geom->nDimension != 4 ||
      grad_output->nDimension != 4) {
    luaL_error(L, "Incorrect dimensions.");
  }
  const int32_t nbatch = u->size[0];
  const int32_t nuchan = u->size[1];
  const int32_t zdim = u->size[2];
  const int32_t ydim = u->size[3];
  const int32_t xdim = u->size[4];

  // We will be accumulating gradient contributions into the grad_u tensor, so
  // we first need to zero it.
  THTensor_(zero)(grad_u);

  real* grad_u_data = THTensor_(data)(grad_u);
  const real* geom_data = THTensor_(data)(geom);
  const real* u_data = THTensor_(data)(u);
  const real* grad_output_data = THTensor_(data)(grad_output);

  int32_t b, z, y, x;
#pragma omp parallel for private(b, z, y, x) collapse(4)
  for (b = 0; b < nbatch; b++) {
    for (z = 0; z < zdim; z++) { 
      for (y = 0; y < ydim; y++) { 
        for (x = 0; x < xdim; x++) {
          const real* cur_u = &u_data[b * nuchan * zdim * ydim * xdim];
          const real* cur_geom = &geom_data[b * zdim * ydim * xdim];
          real* cur_grad_u = &grad_u_data[b * nuchan * zdim * ydim * xdim];
          const real* cur_grad_output =
              &grad_output_data[b * zdim * ydim * xdim];
          const int32_t pos[3] = {x, y, z};
          const int32_t size[3] = {xdim, ydim, zdim};

          tfluids_(Main_calcVelocityDivergenceCellBackward)(
              cur_u, cur_geom, cur_grad_u, cur_grad_output, pos, size, nuchan);
        }
      }
    }
  }
  return 0;
}

*/

static int tfluids_(Main_volumetricUpSamplingNearestForward)(lua_State *L) {
  const int32_t ratio = static_cast<int32_t>(lua_tointeger(L, 1));
  THTensor* input =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* output =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));

  if (input->nDimension != 5 || output->nDimension != 5) {
    luaL_error(L, "ERROR: input and output must be dim 5");
  }

  const int32_t nbatch = input->size[0];
  const int32_t nfeat = input->size[1];
  const int32_t zdim = input->size[2];
  const int32_t ydim = input->size[3];
  const int32_t xdim = input->size[4];

  if (output->size[0] != nbatch || output->size[1] != nfeat ||
      output->size[2] != zdim * ratio || output->size[3] != ydim * ratio ||
      output->size[4] != xdim * ratio) {
    luaL_error(L, "ERROR: input : output size mismatch.");
  }

  const real* input_data = THTensor_(data)(input);
  real* output_data = THTensor_(data)(output);

  int32_t b, f, z, y, x;
#pragma omp parallel for private(b, f, z, y, x) collapse(5)
  for (b = 0; b < nbatch; b++) {
    for (f = 0; f < nfeat; f++) {
      for (z = 0; z < zdim * ratio; z++) {
        for (y = 0; y < ydim * ratio; y++) {
          for (x = 0; x < xdim * ratio; x++) {
            const int64_t iout = output->stride[0] * b + output->stride[1] * f +
                output->stride[2] * z +
                output->stride[3] * y + 
                output->stride[4] * x;
            const int64_t iin = input->stride[0] * b + input->stride[1] * f +
                input->stride[2] * (z / ratio) +
                input->stride[3] * (y / ratio) +
                input->stride[4] * (x / ratio);
            output_data[iout] = input_data[iin];
          }
        }
      }
    }
  }
  return 0;
}

static int tfluids_(Main_volumetricUpSamplingNearestBackward)(lua_State *L) {
  const int32_t ratio = static_cast<int32_t>(lua_tointeger(L, 1));
  THTensor* input =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* grad_output =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* grad_input =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));

  if (input->nDimension != 5 || grad_output->nDimension != 5 ||
      grad_input->nDimension != 5) {
    luaL_error(L, "ERROR: input, gradOutput and gradInput must be dim 5");
  }

  const int32_t nbatch = input->size[0];
  const int32_t nfeat = input->size[1];
  const int32_t zdim = input->size[2];
  const int32_t ydim = input->size[3];
  const int32_t xdim = input->size[4];

  if (grad_output->size[0] != nbatch || grad_output->size[1] != nfeat ||
      grad_output->size[2] != zdim * ratio ||
      grad_output->size[3] != ydim * ratio ||
      grad_output->size[4] != xdim * ratio) {
    luaL_error(L, "ERROR: input : gradOutput size mismatch.");
  }

  if (grad_input->size[0] != nbatch || grad_input->size[1] != nfeat ||
      grad_input->size[2] != zdim || grad_input->size[3] != ydim ||
      grad_input->size[4] != xdim) {
    luaL_error(L, "ERROR: input : gradInput size mismatch.");
  }

  const real* input_data = THTensor_(data)(input);
  const real* grad_output_data = THTensor_(data)(grad_output);
  real * grad_input_data = THTensor_(data)(grad_input);

  int32_t b, f, z, y, x;
#pragma omp parallel for private(b, f, z, y, x) collapse(5)
  for (b = 0; b < nbatch; b++) {
    for (f = 0; f < nfeat; f++) {
      for (z = 0; z < zdim; z++) {
        for (y = 0; y < ydim; y++) {
          for (x = 0; x < xdim; x++) {
            const int64_t iout = grad_input->stride[0] * b +
                grad_input->stride[1] * f +
                grad_input->stride[2] * z +
                grad_input->stride[3] * y +
                grad_input->stride[4] * x;
            float sum = static_cast<real>(0);
            // Now accumulate gradients from the upsampling window.
            for (int32_t zup = 0; zup < ratio; zup++) {
              for (int32_t yup = 0; yup < ratio; yup++) {
                for (int32_t xup = 0; xup < ratio; xup++) {
                  const int64_t iin = grad_output->stride[0] * b +
                      grad_output->stride[1] * f +
                      grad_output->stride[2] * (z * ratio + zup) +
                      grad_output->stride[3] * (y * ratio + yup) +
                      grad_output->stride[4] * (x * ratio + xup);
                  sum += grad_output_data[iin];
                }
              }
            }
            grad_input_data[iout] = sum;
          }
        }
      }
    }
  }
  return 0;
}

static int tfluids_(Main_solveLinearSystemPCG)(lua_State *L) {
  luaL_error(L, "ERROR: solveLinearSystemPCG not defined for CPU tensors.");
  return 0;
}

static const struct luaL_Reg tfluids_(Main__) [] = {
  {"advectScalar", tfluids_(Main_advectScalar)},
//  {"advectVel", tfluids_(Main_advectVel)}, 
//  {"vorticityConfinement", tfluids_(Main_vorticityConfinement)},
//  {"interpField", tfluids_(Main_interpField)},
  {"drawVelocityField", tfluids_(Main_drawVelocityField)},
  {"loadTensorTexture", tfluids_(Main_loadTensorTexture)},
//  {"calcVelocityUpdate", tfluids_(Main_calcVelocityUpdate)},
//  {"calcVelocityUpdateBackward", tfluids_(Main_calcVelocityUpdateBackward)},
//  {"calcVelocityDivergence", tfluids_(Main_calcVelocityDivergence)},
//  {"calcVelocityDivergenceBackward",
//   tfluids_(Main_calcVelocityDivergenceBackward)},
  {"solveLinearSystemPCG", tfluids_(Main_solveLinearSystemPCG)},
  {"volumetricUpSamplingNearestForward",
   tfluids_(Main_volumetricUpSamplingNearestForward)},
  {"volumetricUpSamplingNearestBackward",
   tfluids_(Main_volumetricUpSamplingNearestBackward)},
  {NULL, NULL}  // NOLINT
};

void tfluids_(Main_init)(lua_State *L) {
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, tfluids_(Main__), "tfluids");
}

#endif  // TH_GENERIC_FILE
