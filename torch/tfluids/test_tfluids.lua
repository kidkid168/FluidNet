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

-- Adhoc testing for our utility functions.

-- You can easily test specific units like this:
-- qlua -ltfluids -e "tfluids.test{'calcVelocityDivergenceCUDA'}"

-- Or to test everything:
-- qlua -ltfluids -e "tfluids.test()"

local cutorch = require('cutorch')
local nn = require('nn')

dofile('../lib/load_manta_file.lua')  -- For torch.loadMantaFile()

torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(8)

-- Create an instance of the test framework
local precision = 1e-6
local loosePrecision = 1e-4
local mytester = torch.Tester()
local test = torch.TestSuite()
local times = {}
local profileTimeSec = 1
local jac = nn.Jacobian

local function profileCuda(func, name, args)
  local tm = {}
  times[name] = tm

  -- Profile CPU (convert tensors to float for a fair comparison).
  for key, value in pairs(args) do
    if torch.isTensor(value) then
      args[key] = value:float()
    end
  end
  local a = torch.Timer()
  local count = 0
  while a:time().real < profileTimeSec do
    count = count + 1
    func(unpack(args))
  end
  tm.cpu = a:time().real / count

  -- Profile GPU.
  for key, value in pairs(args) do
    if torch.isTensor(value) then
      args[key] = value:cuda()
    end
  end
  a:reset()
  while a:time().real < profileTimeSec do
    count = count + 1
    func(unpack(args))
  end   
  tm.gpu = a:time().real / count
end

function test.advectScalar()
  for dim = 2, 3 do
    -- Load the pre-advection Manta file for this test.
    local fn = 'test_data/' .. dim .. 'd_initial.bin'
    local _, U, flags, density, is3D = torch.loadMantaFile(fn)
    assert(is3D == (dim == 3))

    -- Now do advection using the 2 parameters and check against Manta.
    for order = 1, 2 do
      for _, openBounds in pairs({false, true}) do
        -- Load the Manta ground truth.
        local openStr
        if openBounds then
          openStr = 'True'
        else
          openStr = 'False'
        end
        fn = ('test_data/' .. dim .. 'd_advect_openBounds_' .. openStr ..
              '_order_' .. order ..'.bin')
        local _, UDiv, flagsDiv, densityManta, is3D = torch.loadMantaFile(fn)
        assert(is3D == (dim == 3))

        -- Make sure that Manta didn't change the flags.
        assert((flags - flagsDiv):abs():max() == 0, 'Flags changed!')

        -- Perform our own advection.
        local dt = 0.1  -- Unfortunately hard coded for now.
        local boundaryWidth = 0  -- Also shouldn't be hard coded.
        local method
        if order == 1 then
          method = 'euler'
        else
          method = 'maccormack'
        end
        -- Note: the clone's here are to make sure every inner loops
        -- sees completely independent data.
        local densityAdv =
            torch.rand(unpack(density:size():totable())):typeAs(density)
        tfluids.advectScalar(dt, density:clone(), U:clone(), flags:clone(),
                             densityAdv, method, openBounds, boundaryWidth)
        local err = densityManta - densityAdv

        mytester:assertlt(err:abs():max(), precision,
                          ('Error: tfluids.advectScalar dim ' .. dim ..
                           ', order ' .. order .. ', openBounds ' ..
                           tostring(openBounds)))
      end
    end
  end
end

--[[

function test.vorticityConfinementCUDA()
  -- TODO(tompson): Write a test of the forward function.

  -- Test that the float and cuda implementations are the same.
  local nchan = {2, 2, 3, 3}
  local d = {1, 1, torch.random(32, 64), torch.random(32, 64)}
  local w = torch.random(32, 64)
  local h = torch.random(32, 64)
  local scale = torch.uniform(0.5, 1)  -- in [0.5, 1]
  local dt = torch.uniform(0.5, 1)
  local case = {'2D', '2DGEOM', '3D', '3DGEOM'}
  local incGeom = {false, true, false,  true}

  for testId = 1, #nchan do
    -- 2D and 3D cases.
    local geom
    if not incGeom[testId] then
      geom = torch.zeros(d[testId], h, w):float()
    else
      geom = torch.rand(d[testId], h, w):gt(0.8):float()
    end

    local U = torch.rand(nchan[testId], d[testId], h, w):float()
    local magCurl = geom:clone():fill(0)
    local curl
    if nchan[testId] == 2 then
      curl = geom:clone()
    else
      curl = U:clone()
    end
    local UCPU = U:clone()

    -- Perform the function on the CPU.
    tfluids.vorticityConfinement(dt, scale, UCPU, geom, curl, magCurl)

    -- Perform the function on the GPU.
    local UGPU = U:cuda()
    local curlGPU = curl:cuda():fill(0)
    local magCurlGPU = magCurl:cuda():fill(0)
    tfluids.vorticityConfinement(dt, scale, UGPU, geom:cuda(), curlGPU,
                                 magCurlGPU)

    -- Compare the results.
    local maxErr = (curlGPU:float() - curl):abs():max()
    mytester:assertlt(maxErr, precision,
                      ('vorticityConfinementZeroGeom CUDA curl ERROR ' ..
                       case[testId]))
    maxErr = (magCurlGPU:float() - magCurl):abs():max()
    mytester:assertlt(maxErr, precision,
                      ('vorticityConfinementZeroGeom CUDA magCurl ERROR ' ..
                       case[testId]))
    maxErr = (UCPU - UGPU:float()):abs():max()
    mytester:assertlt(maxErr, precision,
                      ('vorticityConfinementZeroGeom CUDA ERROR ' ..
                       case[testId]))

    profileCuda(tfluids.vorticityConfinement,
                'vorticityConfinementZeroGeom' .. case[testId],
                {dt, scale, U, geom, curl, magCurl})
  end
end

function test.calcVelocityUpdateCUDA()
  -- NOTE: The forward function test is split between:
  -- torch/utils/test_calc_velocity_update.lua and
  -- torch/lib/modules/test_velocity_update.lua

  -- Test that the float and cuda implementations are the same.
  local batchSize = torch.random(1, 3)
  local nchan = {2, 3}
  local d = {1, torch.random(32, 64), 1}
  local w = torch.random(32, 64)
  local h = torch.random(32, 64)
  local case = {'2D', '3D'}

  for testId = 1, #nchan do
    -- 2D and 3D cases.
    local geom = torch.rand(batchSize, d[testId], h, w):gt(0.8):float()
    local p = torch.rand(batchSize, d[testId], h, w):float()
    local outputCPU =
        torch.rand(batchSize, nchan[testId], d[testId], h, w):float()

    -- Perform the function on the CPU.
    tfluids.calcVelocityUpdate(outputCPU, p, geom)

    -- Perform the function on the GPU.
    local outputGPU = outputCPU:clone():fill(math.huge):cuda()
    tfluids.calcVelocityUpdate(outputGPU, p:cuda(), geom:cuda())

    -- Compare the results.
    local maxErr = (outputCPU - outputGPU:float()):abs():max()
    mytester:assertlt(maxErr, precision,
                      'calcVelocityUpdate CUDA ERROR ' .. case[testId])

    -- Now test the backwards call.
    local gradOutput =
      torch.rand(batchSize, nchan[testId], d[testId], h, w):float()

    local gradPCPU = p:clone():fill(math.huge)
    tfluids.calcVelocityUpdateBackward(gradPCPU, p, geom, gradOutput)

    local gradPGPU = p:clone():fill(math.huge):cuda()
    tfluids.calcVelocityUpdateBackward(gradPGPU, p:cuda(), geom:cuda(),
                                       gradOutput:cuda())

    maxErr = (gradPCPU - gradPGPU:float()):abs():max()
    mytester:assertlt(maxErr, precision,
                      'calcVelocityUpdateBackward CUDA ERROR ' .. case[testId])

    profileCuda(tfluids.calcVelocityUpdate,
                'calcVelocityUpdate' .. case[testId],
                {outputCPU, p, geom})

    profileCuda(tfluids.calcVelocityUpdateBackward,
                'calcVelocityUpdateBackward' .. case[testId],
                {gradPCPU, p, geom, gradOutput})
  end
end

--]]

--[[
function test.advectScalarCUDA()
  -- Test that the float and cuda implementations are the same.
  for dim = 2, 3 do
    local dt = torch.uniform(0.5, 1)
    local d
    if dim == 2 then
      d = 1
    else
      d = torch.random(32, 64)
    end
    local h = torch.random(32, 64)
    local w = torch.random(32, 64)
    local methods = {'euler', 'maccormack'}

    local p = torch.rand(d, h, w):float()
    -- Mul rand by 10 to get reasonable coverage over multiple cells.
    local u = torch.rand(dim, d, h, w):mul(5):float()
    local geom = torch.rand(d, h, w):gt(0.8):float()

    p:cmul(1 - geom)  -- Zero out geometry cells.

    for _, method in pairs(methods) do
      local caseStr = ('advectScalar - method: ' .. method .. ', dim: ' ..
                       dim)

      -- Perform advection on the CPU.
      local pDst = p:clone():fill(0)
      tfluids.advectScalar(dt, p, u, geom, pDst, method)

      -- Perform advection on the GPU.
      local pDstGPU = p:clone():fill(0):cuda()
      tfluids.advectScalar(dt, p:cuda(), u:cuda(), geom:cuda(), pDstGPU,
                           method)
      
      local maxErr = (pDst - pDstGPU:float()):abs():max()
      -- TODO(tompson): It's troubling that we need loose precision here.
      -- Maybe it's OK, but it is never-the-less surprising.

      -- NOTE: This might actually fail sometimes (rarely, but I've seen
      -- it happen). This is because floating point roundoff can cause rays
      -- to brush past geometry cells but not actually trigger intersections. 
      mytester:assertlt(maxErr, loosePrecision, 'ERROR: ' .. caseStr)

      profileCuda(tfluids.advectScalar, caseStr,
                  {dt, p, u, geom, pDst, method})
    end
  end
end
--]]

--[[
function test.advectVelCUDA()
  -- Test that the float and cuda implementations are the same.
  for dim = 2, 3 do
    local dt = torch.uniform(0.5, 1)
    local d
    if dim == 2 then
      d = 1
    else
      d = torch.random(32, 64)
    end
    local h = torch.random(32, 64)
    local w = torch.random(32, 64)
    local methods = {'euler', 'maccormack'}  

    -- Mul rand by 10 to get reasonable coverage over multiple cells.
    local u = torch.rand(dim, d, h, w):mul(5):float()
    local geom = torch.rand(d, h, w):gt(0.8):float()

    for _, method in pairs(methods) do
      local caseStr = ('advectVel - method: ' .. method .. ', dim: ' .. dim)

      -- Perform advection on the CPU.
      local uDst = u:clone():fill(0) 
      tfluids.advectVel(dt, u, geom, uDst, method)

      -- Perform advection on the GPU.
      local uDstGPU = u:clone():fill(0):cuda()
      tfluids.advectVel(dt, u:cuda(), geom:cuda(), uDstGPU, method)
      
      local maxErr = (uDst - uDstGPU:float()):abs():max()
      -- TODO(tompson): It's troubling that we need loose precision here.
      -- Maybe it's OK, but it is never-the-less surprising.

      -- NOTE: This might actually fail sometimes (rarely, but I've seen
      -- it happen). This is because floating point roundoff can cause rays
      -- to brush past geometry cells but not actually trigger intersections. 
      mytester:assertlt(maxErr, loosePrecision, 'ERROR: ' .. caseStr)

      profileCuda(tfluids.advectVel, caseStr, {dt, u, geom, uDst, method})
    end
  end
end
--]]

--[[
function test.solveLinearSystemPCGCUDA()
  local batchSize = torch.random(1, 3)
  local w = torch.random(32, 64)
  local h = torch.random(32, 64)
  for dim = 2, 3 do
    local d
    if dim == 3 then
      d = torch.random(32, 64)
    else
      d = 1
    end

    -- Create some random inputs.
    local geom = torch.rand(batchSize, d, h, w):gt(0.8):cuda()
    local p = torch.rand(batchSize, d, h, w):cuda()
    local U = torch.rand(batchSize, dim, d, h, w):cuda()

    -- Allocate the output tensor.
    local deltaU = torch.CudaTensor(batchSize, dim, d, h, w)

    -- Call the forward function.
    tfluids.solveLinearSystemPCG(deltaU, p, geom, U)

    -- TODO(kris): Check the output.
  end
end
--]]

--[[

function test.calcVelocityDivergenceCUDA()
  -- NOTE: The forward and backward function tests are in:
  -- torch/lib/modules/test_velocity_divergence.lua

  -- Test that the float and cuda implementations are the same.
  local batchSize = torch.random(1, 3)
  local nchan = {2, 3}
  local d = {1, torch.random(32, 64)}
  local w = torch.random(32, 64)
  local h = torch.random(32, 64)
  local case = {'2D', '3D'}

  for testId = 1, 2 do
    local geom = torch.rand(batchSize, d[testId], h, w):gt(0.8):float()
    local U = torch.rand(batchSize, nchan[testId], d[testId], h, w):float()
    local UDivCPU = geom:clone():fill(0)
    
    -- Perform the function on the CPU.
    tfluids.calcVelocityDivergence(U, geom, UDivCPU)

    -- Perform the function on the GPU. 
    local UDivGPU = UDivCPU:clone():fill(math.huge):cuda()
    tfluids.calcVelocityDivergence(U:cuda(), geom:cuda(), UDivGPU)

    -- Compare the results.
    local maxErr = (UDivCPU - UDivGPU:float()):abs():max()
    mytester:assertlt(maxErr, precision,
                      'calcVelocityDivergence CUDA ERROR ' .. case[testId])
  
    -- Now test the backwards call.
    local gradOutput =
      torch.rand(batchSize, d[testId], h, w):float()
  
    local gradUCPU = U:clone():fill(math.huge)
    tfluids.calcVelocityDivergenceBackward(gradUCPU, U, geom, gradOutput)
  
    local gradUGPU = U:clone():fill(math.huge):cuda()
    tfluids.calcVelocityDivergenceBackward(gradUGPU, U:cuda(), geom:cuda(),
                                       gradOutput:cuda())

    maxErr = (gradUCPU - gradUGPU:float()):abs():max()
    mytester:assertlt(
        maxErr, precision, 'calcVelocityDivergenceBackward CUDA ERROR ' ..
        case[testId])

    profileCuda(tfluids.calcVelocityDivergence,
                'calcVelocityDivergence' .. case[testId], {U, geom, UDivCPU})
    
    profileCuda(tfluids.calcVelocityDivergenceBackward,
                'calcVelocityDivergenceBackward' .. case[testId],
                {gradUCPU, U, geom, gradOutput})
  end
end 

function test.interpField()
  local d = torch.random(8, 16)
  local h = torch.random(8, 16)
  local w = torch.random(8, 16)
  local eps = 1e-6

  local geom = torch.zeros(d, h, w)
  local field = torch.rand(d, h, w)

  -- Center of the first cell.
  local pos = torch.Tensor({0, 0, 0})
  local val = tfluids.interpField(field, geom, pos)
  mytester:asserteq(val, field[{1, 1, 1}], 'Bad interp value')

  -- Center of the last cell.
  pos = torch.Tensor({w - 1, h - 1, d - 1})
  val = tfluids.interpField(field, geom, pos)
  mytester:asserteq(val, field[{-1, -1, -1}], 'Bad interp value')
  
  -- The corner of the grid should also be the center of the first cell.
  pos = torch.Tensor({-0.5 + eps, -0.5 + eps, -0.5 + eps})
  val = tfluids.interpField(field, geom, pos)
  mytester:asserteq(val, field[{1, 1, 1}], 'Bad interp value')

  -- The right edge of the first cell should be the average of the two.
  pos = torch.Tensor({0.5, 0, 0})
  val = tfluids.interpField(field, geom, pos)
  mytester:asserteq(val, field[{1, 1, {1, 2}}]:mean(), 'Bad interp value')

  -- The top edge of the first cell should be the average of the two.
  pos = torch.Tensor({0, 0.5, 0})
  val = tfluids.interpField(field, geom, pos)
  mytester:asserteq(val, field[{1, {1, 2}, 1}]:mean(), 'Bad interp value')

  -- The back edge of the first cell should be the average of the two.
  pos = torch.Tensor({0, 0, 0.5})
  val = tfluids.interpField(field, geom, pos)
  mytester:asserteq(val, field[{{1, 2}, 1, 1}]:mean(), 'Bad interp value')

  -- The corner of the first cell should be the average of all the neighbours.
  pos = torch.Tensor({0.5, 0.5, 0.5})
  val = tfluids.interpField(field, geom, pos)
  mytester:asserteq(val, field[{{1, 2}, {1, 2}, {1, 2}}]:mean(),
                    'Bad interp value')

  -- TODO(tompson,kris): Is this enough test cases?
end

--]]

function test.VolumetricUpSamplingNearest()
  local batchSize = torch.random(1, 5)
  local nPlane = torch.random(1, 5)
  local widthIn = torch.random(5, 8)
  local heightIn = torch.random(5, 8)
  local depthIn = torch.random(5, 8)
  local ratio = torch.random(1, 3)

  local module = tfluids.VolumetricUpSamplingNearest(ratio)

  local input = torch.rand(batchSize, nPlane, depthIn, heightIn, widthIn)
  local output = module:forward(input):clone()

  assert(output:dim() == 5)
  assert(output:size(1) == batchSize)
  assert(output:size(2) == nPlane)
  assert(output:size(3) == depthIn * ratio)
  assert(output:size(4) == heightIn * ratio)
  assert(output:size(5) == widthIn * ratio)

  local outputGT = torch.Tensor():resizeAs(output)
  for b = 1, batchSize do
    for f = 1, nPlane do
      for z = 1, depthIn * ratio do
        local zIn = math.floor((z - 1) / ratio) + 1
        for y = 1, heightIn * ratio do
          local yIn = math.floor((y - 1) / ratio) + 1
          for x = 1, widthIn * ratio do
            local xIn = math.floor((x - 1) / ratio) + 1
            outputGT[{b, f, z, y, x}] = input[{b, f, zIn, yIn, xIn}]
          end
        end
      end
    end
  end

  -- Note FPROP should be exact (it's just a copy).
  mytester:asserteq((output - outputGT):abs():max(), 0, 'error on fprop')

  -- Generate a valid gradInput (we'll use it to test the GPU implementation).
  local gradOutput = torch.rand(batchSize, nPlane, depthIn * ratio,
                                heightIn * ratio, widthIn * ratio);
  local gradInput = module:backward(input, gradOutput):clone()

  -- Perform the function on the GPU.
  module:cuda()
  local outputGPU = module:forward(input:cuda()):double()
  mytester:assertle((output - outputGPU):abs():max(), precision,
                    'error on GPU fprop')

  local gradInputGPU =
      module:backward(input:cuda(), gradOutput:cuda()):double()
  mytester:assertlt((gradInput - gradInputGPU):abs():max(), precision,
                    'error on GPU bprop')

  -- Check BPROP is correct.
  module:double()
  local err = jac.testJacobian(module, input)
  mytester:assertlt(err, precision, 'error on bprop')
end

-- Now run the test above
mytester:add(test)

function tfluids.test(tests, seed, gpuDevice)
  local curDevice = cutorch.getDevice()
  -- By default don't test on the primary device.
  gpuDevice = gpuDevice or 1
  cutorch.setDevice(gpuDevice)
  print('Testing on gpu device ' .. gpuDevice)
  print(cutorch.getDeviceProperties(gpuDevice))

  -- randomize stuff.
  local seed = seed or (1e5 * torch.tic())
  print('Seed: ', seed)
  math.randomseed(seed)
  torch.manualSeed(seed)
  cutorch.manualSeed(seed)
  mytester:run(tests)

  numTimes = 0
  for _, _ in pairs(times) do
    numTimes = numTimes + 1
  end

  if numTimes > 0 then
    print ''
    print('-----------------------------------------------------------------' ..
          '-------------')
    print('| Module                                                       ' ..
          '| Speedup     |')
    print('-----------------------------------------------------------------' ..
          '-------------')
    for module, tm in pairs(times) do
      local str = string.format('| %-60s | %6.2f      |', module,
                                (tm.cpu / tm.gpu))
      print(str)
    end
    print('-----------------------------------------------------------------' ..
          '-------------')
  end

  cutorch.setDevice(curDevice)

  return mytester
end

