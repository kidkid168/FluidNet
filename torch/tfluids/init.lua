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

local cutorch = require('cutorch')
local torch = require('torch')
local tfluids = require('libtfluids')

tfluids._tmp = {}  -- We'll allocate temporary arrays in here.

-- Advect scalar field 'p' by the input vel field 'u'.
--
-- @param dt - timestep (seconds).
-- @param s - input scalar field to advect, size: (depth x height x width)
-- @param U - input vel field, size: (2/3 x depth x height x width) (MAC Grid)
-- @param flags - input occupancy grid, size: (depth x height x width)
-- @param sDst - Return (pre-allocated) scalar field, same size as p.
-- @param is3D - If the input is 3D or not.
-- @param method - OPTIONAL - "euler", "maccormack" (default).
-- @param openBounds - OPTIONAL - bool indicating boundary. (default: faulse)
-- @param boundaryWidth - OPTIONAL - boundary width. (default 1)
local function advectScalar(dt, s, U, flags, sDst, method, openBounds,
                            boundaryWidth)
  method = method or "maccormack"
  if openBounds == nil then
    openBounds = false
  end
  boundaryWidth = boundaryWidth or 1

  -- Check arguments here (it's easier from lua).
  local is3D = U:size(1) == 3
  assert(s:dim() == 4 and U:dim() == 4 and flags:dim() == 4 and
         sDst:dim() == 4, 'Dimension mismatch')
  assert(s:size(1) == 1, 's is not scalar')
  local d = flags:size(2)
  local h = flags:size(3)
  local w = flags:size(4)
  if not is3D then
    assert(d == 1, '2D velocity field but zdepth > 1')
  end
  assert(s:isSameSizeAs(flags) and s:isSameSizeAs(sDst), 'Size mismatch')
  assert(U:size(2) == d and U:size(3) == h and U:size(4) == w, 'Size mismatch')
  assert((not is3D and U:size(1) == 2) or (is3D and U:size(1) == 3))

  -- Note: the C++ code actually does not need the input to be contiguous but
  -- the CUDA code does. But we'll conservatively constrain both anyway.
  assert(s:isContiguous() and U:isContiguous() and flags:isContiguous() and
         sDst:isContiguous(), 'Input is not contiguous')

  -- If we're using maccormack advection we need a temporary array, however
  -- we should just allocate always since it makes the C++ logic easier.
  tfluids._tmp[torch.type(s)] = tfluids._tmp[torch.type(s)] or s:clone()
  tfluids._tmp[torch.type(s)]:resizeAs(s)
  local sTmp = tfluids._tmp[torch.type(s)]

  s.tfluids.advectScalar(dt, s, U, flags, sTmp, is3D, method, openBounds,
                         boundaryWidth, sDst)
end
rawset(tfluids, 'advectScalar', advectScalar)

--[[

-- Advect vel field 'u' by the input vel field 'u' (and store result in uDst).
--
-- @param dt - timestep (seconds).
-- @param u - input vel field to advect, size: (2/3 x depth x height x width)
-- (MAC Grid)
-- @param flags - input occupancy grid, size: (depth x height x width).
-- @param uDst - Return (pre-allocated) velocity field, same size as u.
-- @param method - OPTIONAL - "euler" or "maccormack" (default).
-- @param openBounds - OPTIONAL - bool indicating boundary type.
-- @param boundaryWidth - OPTIONAL - boundary width.
local function advectVel(dt, u, flags, uDst, method)
  method = method or "maccormack"
  if openBounds == nil then
    openBounds = false
  end
  boundaryWidth = boundaryWidth or 1

  -- Check arguments here (it's easier from lua).
  local twoDim = u:size(1) == 2
  assert(u:dim() == 4 and flags:dim() == 3 and uDst:dim() == 4,
         'Dimension mismatch')
  local d = flags:size(1)
  local h = flags:size(2)
  local w = flags:size(3)
  if twoDim then
    assert(d == 1, '2D velocity field but zdepth > 1')
  end
  assert(u:isSameSizeAs(uDst), 'Size mismatch')
  assert(u:size(2) == d and u:size(3) == h and u:size(4) == w, 'Size mismatch')
  assert((twoDim and u:size(1) == 2) or (not twoDim and u:size(1) == 3))
  assert(u:isContiguous() and flags:isContiguous() and uDst:isContiguous(),
         'Input is not contiguous')

  u.tfluids.advectVel(dt, u, flags, uDst, method)
end
rawset(tfluids, 'advectVel', advectVel)

-- Magnify the vortices in  vel field 'u' by the input vel field 'u' (and 
-- store result in uDst).
--
-- @param dt - timestep (seconds).
-- @param scale - scale of vortex magnifying force
-- @param u - input vel field to effect, size: (2/3 x depth x height x width).
-- @param flags - input occupancy grid, size: (depth x height x width)
-- @param curl - temporary buffer (stores curl of u).  size of u in 3D or size
-- of flags in 2D (curl is a scalar field for the 2D case).
-- @param magCurl - temporary buffer (stores ||curl of u||). size of flags.
local function vorticityConfinement(dt, scale, u, flags, curl, magCurl)
  -- Check arguments here (it's easier from lua).
  local twoDim = u:size(1) == 2
  assert(u:dim() == 4 and flags:dim() == 3, 'Dimension mismatch')
  local d = flags:size(1)
  local h = flags:size(2)
  local w = flags:size(3)
  assert(u:size(2) == d and u:size(3) == h and u:size(4) == w, 'Size mismatch')
  assert((twoDim and u:size(1) == 2) or (not twoDim and u:size(1) == 3))
  assert(u:isContiguous() and flags:isContiguous(), 'input is not contiguous')
  if not twoDim then
    assert(curl:isSameSizeAs(u))
  else
    assert(curl:isSameSizeAs(flags))
  end
  assert(magCurl:isSameSizeAs(flags))

  u.tfluids.vorticityConfinement(dt, scale, u, flags, curl, magCurl)
end
rawset(tfluids, 'vorticityConfinement', vorticityConfinement)

-- Interpolate field (exposed for debugging) using trilinear interpolation.
-- Can only interpolate positions within non-flagsetry cells (will throw an
-- error otherwise). Note: (0, 0, 0) is defined as the CENTER of the first
-- grid cell, so (-0.5, -0.5, -0.5) is the start of the 3D grid and (dimx + 0.5,
-- dimy + 0.5, dimz + 0.5) is the last cell.
-- @param field - Tensor of size (depth x height x width).
-- @param flags - occupancy grid of size (depth x height x width).
-- @param pos - Tensor of size (3).
-- @param sampleIntoGeom - OPTIONAL see argument description for advectScalar.
local function interpField(field, flags, pos, sampleIntoGeom)
  if sampleIntoGeom == nil then
    sampleIntoGeom = true
  end
  assert(torch.isTensor(field) and torch.isTensor(flags) and torch.isTensor(pos))
  assert(field:dim() == 3 and flags:dim() == 3, '4D tensor expected')
  assert(field:isSameSizeAs(flags), 'Size mismatch')
  assert(pos:dim() == 1 and pos:size(1) == 3)
  assert(pos[1] >= -0.5 and pos[1] <= flags:size(3) - 0.5,
      'pos[1] out of bounds')
  assert(pos[2] >= -0.5 and pos[2] <= flags:size(2) - 0.5,
      'pos[2] out of bounds')
  assert(pos[3] >= -0.5 and pos[3] <= flags:size(1) - 0.5,
      'pos[3] out of bounds')

  return field.tfluids.interpField(field, flags, pos, sampleIntoGeom)
end
rawset(tfluids, 'interpField', interpField)

-- flipY will render the field upside down (required to get alignment with
-- grid = 0 on the bottom of the OpenGL context).
local function drawVelocityField(U, flipY)
  if flipY == nil then
    flipY = false
  end
  assert(U:dim() == 5)  -- Expected batch input.
  assert(U:size(2) == 2 or U:size(2) == 3)
  local twoDim = U:size(2) == 2
  assert(not twoDim or U:size(3) == 1)
  U.tfluids.drawVelocityField(U, flipY)
end
rawset(tfluids, 'drawVelocityField', drawVelocityField)

-- loadTensorTexture performs a glTexImage2D call on the imTensor data (and
-- handles correct re-swizzling of the data).
local function loadTensorTexture(imTensor, texGLID, filter, flipY)
  if flipY == nil then
    flipY = true
  end
  assert(imTensor:dim() == 2 or imTensor:dim() == 3)
  imTensor.tfluids.loadTensorTexture(imTensor, texGLID, filter, flipY)
end
rawset(tfluids, 'loadTensorTexture', loadTensorTexture)

-- calcVelocityUpdate assumes the input is batched.
-- Calculates: U = Udiv - grad(p)
-- With some special handling for flagsetry cells!
local function calcVelocityUpdate(U, UDiv, p, flags)
  assert(U:dim() == 5 and UDiv:dim() == 5 and p:dim() == 4 and flags:dim() == 4)
  local nbatch = U:size(1)
  local twoDim = U:size(2) == 2
  local zdim = U:size(3)
  local ydim = U:size(4)
  local xdim = U:size(5)
  if not twoDim then
    assert(U:size(2) == 3, 'Bad number of velocity slices')
  end
  assert(UDiv:isSameSizeAs(U))
  assert(p:isSameSizeAs(flags))
  assert(p:size(1) == nbatch)
  assert(p:size(2) == zdim)
  assert(p:size(3) == ydim)
  assert(p:size(4) == xdim)
  assert(p:isContiguous() and U:isContiguous() and flags:isContiguous() and
         UDiv:isContiguous())
  deltaU.tfluids.calcVelocityUpdate(U, UDiv, p, flags)
end
rawset(tfluids, 'calcVelocityUpdate', calcVelocityUpdate)

--]]

-- Calculate velocity update using cuSPARSE (NVIDIA) library's PCG
-- implementation. Note: there's no backward version of this function.
--
-- @param deltaU: The velocity update to zero the divergence of U.
-- @param p: The p field from the previous timestep.
-- @param flags: The current flagsetry field.
-- @param U: The DIVERGENT velocity field for the current timestep.
local function solveLinearSystemPCG(deltaU, p, flags, U)
  assert(deltaU:dim() == 5 and p:dim() == 4 and flags:dim() == 4)
  local nbatch = deltaU:size(1)
  local twoDim = deltaU:size(2) == 2
  local zdim = deltaU:size(3)
  local ydim = deltaU:size(4)
  local xdim = deltaU:size(5)
  if not twoDim then
    assert(deltaU:size(2) == 3, 'Bad number of velocity slices')
  end
  assert(deltaU:isSameSizeAs(U))
  assert(p:isSameSizeAs(flags))
  assert(p:size(1) == nbatch)
  assert(p:size(2) == zdim)
  assert(p:size(3) == ydim)
  assert(p:size(4) == xdim)
  assert(p:isContiguous() and deltaU:isContiguous() and flags:isContiguous() and
         U:isContiguous())
  deltaU.tfluids.solveLinearSystemPCG(deltaU, p, flags, U)
end
rawset(tfluids, 'solveLinearSystemPCG', solveLinearSystemPCG)

--[[

-- Calculates the partial derivative of calcVelocityUpdate w.r.t. p.
-- It DOES NOT calculate the partial derivative w.r.t velocity.
local function calcVelocityUpdateBackward(gradP, U, UDiv, p, flags, gradOutput)
  assert(gradP:dim() == 4, 'gradP must be 4D')
  assert(U:dim() == 5 and UDiv:dim() == 5, 'U and UDiv must be 5D')
  assert(p:dim() == 4, 'p must be 4D')
  assert(flags:dim() == 4, 'flags must be 4D')
  assert(gradOutput:dim() == 5, 'gradOutput must be 5D')
  assert(gradP:isSameSizeAs(p) and gradP:isSameSizeAs(flags))
  local nbatch = gradP:size(1)
  local zdim = gradP:size(2)
  local ydim = gradP:size(3)
  local xdim = gradP:size(4)
  local twoDim = gradOutput:size(2) == 2
  assert(U:isSameSizeAs(UDiv))
  assert(U:isSameSizeAs(gradOutput))
  if not twoDim then
    assert(gradOutput:size(2) == 3, 'Bad number of velocity slices')
  else
    assert(zdim == 1, 'zdim is too large')
  end
  assert(p:isContiguous() and gradP:isContiguous() and flags:isContiguous()
         and gradOutput:isContiguous() and U:isContiguous() and
         UDiv:isContiguous())
  gradP.tfluids.calcVelocityUpdateBackward(gradP, U, UDiv, p, flags, gradOutput)
end
rawset(tfluids, 'calcVelocityUpdateBackward', calcVelocityUpdateBackward)

-- calcVelocityDivergence assumes the input is batched.
local function calcVelocityDivergence(U, flags, UDiv)
  assert(U:dim() == 5 and flags:dim() == 4 and UDiv:dim() == 4)
  assert(flags:isSameSizeAs(UDiv))
  local nbatch = U:size(1)
  local twoDim = U:size(2) == 2
  local zdim = U:size(3)
  local ydim = U:size(4)
  local xdim = U:size(5)
  if not twoDim then
    assert(U:size(2) == 3, 'Bad number of velocity slices')
  end
  assert(flags:size(1) == nbatch)
  assert(flags:size(2) == zdim)
  assert(flags:size(3) == ydim)
  assert(flags:size(4) == xdim)
  assert(U:isContiguous() and flags:isContiguous())
  U.tfluids.calcVelocityDivergence(U, flags, UDiv)
end
rawset(tfluids, 'calcVelocityDivergence', calcVelocityDivergence)

-- Calculates the partial derivative of calcVelocityDivergence.
local function calcVelocityDivergenceBackward(gradU, U, flags, gradOutput)
  assert(gradU:dim() == 5 and U:dim() == 5 and flags:dim() == 4 and
         gradOutput:dim() == 4)
  assert(gradU:isSameSizeAs(U))
  assert(gradOutput:isSameSizeAs(flags))
  local nbatch = flags:size(1)
  local zdim = flags:size(2)
  local ydim = flags:size(3)
  local xdim = flags:size(4)
  local twoDim = U:size(2) == 2
  if not twoDim then
    assert(U:size(2) == 3, 'Bad number of velocity slices')
  else
    assert(zdim == 1, 'zdim is too large')
  end
  assert(U:size(1) == nbatch)
  assert(U:size(3) == zdim)
  assert(U:size(4) == ydim)
  assert(U:size(5) == xdim)
  assert(U:isContiguous() and gradU:isContiguous() and flags:isContiguous())
  U.tfluids.calcVelocityDivergenceBackward(gradU, U, flags, gradOutput)
end
rawset(tfluids, 'calcVelocityDivergenceBackward',
       calcVelocityDivergenceBackward)

--]]

include('volumetric_up_sampling_nearest.lua')

-- Also include the test framework.
include('test_tfluids.lua')

return tfluids
