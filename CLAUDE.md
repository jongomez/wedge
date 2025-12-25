# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Wedge?

Wedge is a WebGL2-based neural network inference engine that compiles and executes TensorFlow.js models directly on the GPU using custom GLSL fragment shaders. It's an alternative runtime to TensorFlow.js's built-in WebGL backend.

## Commands

This is an npm workspaces monorepo. All commands run from the root directory.

```bash
# Install dependencies
npm install

# Run development servers
npm run -w apps/site dev          # Main site app (http://localhost:3000)
npm run -w packages/core dev      # Core package dev server
npm run -w benchmarks dev         # Benchmarks app

# Build
npm run -w apps/site build
npm run -w packages/core build

# Lint
npm run -w packages/core lint
npm run -w apps/site lint
```

## Architecture

### Execution Flow

1. **Model Loading**: TensorFlow.js GraphModel loaded via `loadGraphModel()`
2. **Graph Construction**: Model converted to internal `Graph` representation with `GraphNode` objects
3. **Compilation**: Graph nodes ordered for execution, WebGL programs compiled
4. **Execution**: For each operation - bind input textures, set uniforms, render to framebuffer
5. **Output**: Final texture read back to CPU as Float32Array

### Core Package (`packages/core/src/`)

**Main Runtime**: `backends/webgl/WedgeWebGL.ts`
- `WedgeWebGL` class manages WebGL context, graph compilation, and execution
- `opNodeWithProgramMap`: Maps operation names to compiled WebGL programs
- `run(inputRawData)`: Execute inference and return Float32Array

**Operation Implementations**: `backends/webgl/ops/`
Each operation (conv2D, depthwiseConv2D, relu, arithmetic, ResizeBilinear) has:
- `init.ts` - Setup and initialization
- `output.ts` - Output texture configuration
- `webGLShader.ts` - GLSL fragment shader code

**Graph System**: `graph/`
- `GraphNode`: Single operation with inputs, outputs, params
- `Graph`: Full computation graph with placeholders, weights, ordered nodes

**Tensor System**: `tensor/`
- `TensorWebGL`: GPU-backed tensor stored as WebGL textures
- Data formats: NHWC, HWC, VEC

### Supported Operations

Defined in `backends/webgl/types.ts`:
- Conv2D, _FusedConv2D, DepthwiseConv2D variants
- AddV2, Mul (arithmetic)
- Relu (activation)
- Pad, PadV2, MirrorPad (padding)
- ResizeBilinear, Identity

**When adding a new operation:**
1. Add types to `backends/webgl/types.ts` (OpName union)
2. Create op folder in `backends/webgl/ops/` with init.ts, output.ts, webGLShader.ts
3. Register in `backends/webgl/webGLData.ts` switch statement
4. Add to `SUPPORTED_OPS` in `apps/site/app/model/page.tsx` for visualization
5. For LayersModel support, add mapping in `modelHelpers.ts` mapLayerClassesToOpName()

### Site App (`apps/site/`)

Next.js 16 app with pages:
- `/` - Landing page
- `/model` - Model visualization (load model, view layers, see support status)
- `/tests` - Test suite with sidebar navigation
- `/classification` - Camera demo (WIP)

### Path Aliases

Site app tsconfig uses these path aliases:
- `@/*` → local files
- `@wedge/core/*` → `packages/core/src/*`
- `@wedge/ui-react/*` → `packages/ui-react/src/*`

## Key Files

| File | Purpose |
|------|---------|
| `packages/core/src/backends/webgl/WedgeWebGL.ts` | Main runtime class |
| `packages/core/src/backends/webgl/types.ts` | WebGLOpNode, OpName, WedgeOptions |
| `packages/core/src/graph/types.ts` | GraphNode, Graph interfaces |
| `packages/core/src/backends/webgl/ops/conv2D/webGLShader.ts` | Example shader |
| `apps/site/app/model/page.tsx` | Model visualization UI |

## Shader Coordinate Conventions

The codebase uses specific coordinate conventions that must be followed:

### `convertFlatToHWC3D(flatIndex, width, height, depth)`
- Returns `ivec3(x, y, z)` where for square dimensions: x=w, y=h, z=c
- Called with `(outputDims.x, outputDims.y, paddedDepth)` where outputDims=(H, W, C)
- The naming is confusing: "width" parameter is actually H, "height" is W

### Flat Index Formula
```glsl
flatIndex = y * inputDims.x * paddedDepth + x * paddedDepth + z
// Which translates to: h * H * paddedC + w * paddedC + c
```

### Important Implementation Notes

**4D vs 3D shapes**: Input tensors from LayersModel are often 4D [N,H,W,C] even with `hasBatchDimension: false`. Shaders must detect and handle this:
```typescript
const inputIs4D = input.originalShape.length === 4;
const inputH = inputIs4D ? input.originalShape[1] : input.originalShape[0];
```

**Channel padding**: Channels are padded to multiples of 4 for texture storage. Use `paddedInputDepth = ((inputDims.z + 3) / 4) * 4`.

**Output reading**: The `removePadChannels()` function in `transforms.ts` handles both channel padding AND texture spatial padding when reading output.

**LayersModel attributes**: In `create.ts`, `getAttrParams()` stores padding under key `'pad'` not `'padding'`. Check both when reading attributes.
