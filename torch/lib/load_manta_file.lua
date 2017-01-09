-- Copyright 2016 Google Inc, NYU.
-- 
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
-- 
--     http://www.apache.org/licenses/LICENSE-2.0
-- 
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- This is a little hacky to define these staticlly here, rather than
-- import them from manta/source/grid.h (enum CellType in FlagGrid class).
-- But there's no simple way to link C++ from Manta to lua here (we can't
-- do it through tfluids, because tfluids would need manta as a dependency).
local CELL_TYPE = {
    TypeNone     = 0,
    TypeFluid    = 1,
    TypeObstacle = 2,
    TypeEmpty    = 4,
    TypeInflow   = 8,
    TypeOutflow  = 16,
    TypeOpen     = 32,
    TypeStick    = 128,
    TypeReserved = 256,
    -- 2^10 - 2^14 reserved for moving obstacles
    TypeZeroPressure = bit.lshift(1, 15)  -- Never used in our code.
}

function torch.loadMantaFile(fn)
  assert(paths.filep(fn), "couldn't find ".. fn)
  local file = torch.DiskFile(fn, 'r')
  file:binary()
  local transpose = file:readInt()  -- Legacy. Never used.
  local nx = file:readInt()
  local ny = file:readInt()
  local nz = file:readInt()
  local is3D = file:readInt()
  is3D = is3D == 1
  local numel = nx * ny * nz
  local Ux = torch.FloatTensor(file:readFloat(numel))
  local Uy = torch.FloatTensor(file:readFloat(numel))
  local Uz
  if is3D then
    Uz = torch.FloatTensor(file:readFloat(numel))
  end
  local p = torch.FloatTensor(file:readFloat(numel))
  -- Note: flags are stored as integers but well treat it as floats to make it
  -- easier to sent to CUDA.
  local flags = torch.IntTensor(file:readInt(numel)):float()
  local density = torch.FloatTensor(file:readFloat(numel))

  Ux:resize(1, nz, ny, nx)
  Uy:resize(1, nz, ny, nx)
  if is3D then
    Uz:resize(1, nz, ny, nx)
  end
  p:resize(1, nz, ny, nx)
  flags:resize(1, nz, ny, nx)
  density:resize(1, nz, ny, nx)

  local U
  if is3D then
    U = torch.cat({Ux, Uy, Uz}, 1)
  else
    U = torch.cat({Ux, Uy}, 1)
  end

  -- Ignore the border pixels.
  return p, U, flags, density, is3D
end

