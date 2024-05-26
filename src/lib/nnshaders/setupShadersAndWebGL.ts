


export function initWebGL(width: number, height: number, viewportMaxSize: number): InitWebGLReturn {
  const canvas = new OffscreenCanvas(width, height);

  var gl = canvas.getContext('webgl2');

  if (!gl) {
    throw new Error("Unable to initialize WebGL2. Your browser may not support it.");
  }

  // FIXME (maybe): Should the gl viewport have the same size as the output texture and / or the canvas?
  gl.viewport(0, 0, viewportMaxSize, viewportMaxSize);

  // checkFloatTextureSupport(this.gl);

  const maxTextureImageUnits = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS);
  console.log('Maximum texture image units:', maxTextureImageUnits);

  const max2DTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  console.log('Maximum 2D texture size:', max2DTextureSize);

  const max3DTextureSize = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE);
  console.log('Maximum 3D texture size:', max3DTextureSize);

  const maxArrayTextureLayers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS);
  console.log('Maximum array texture layers:', maxArrayTextureLayers);

  const maxFragmentUniformVectors = gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS);
  console.log('Maximum fragment uniform vectors:', maxFragmentUniformVectors);

  const maxDrawBuffers = gl.getParameter(gl.MAX_DRAW_BUFFERS);
  console.log('Maximum draw buffers:', maxDrawBuffers);

  // This one is important. Without it, we can't use floating point outputs on fragment shaders.
  const colorBufferFloat = gl.getExtension('EXT_color_buffer_float');

  if (colorBufferFloat === null) {
    console.error('EXT_color_buffer_float extension is not supported.');
  } else {
    console.log('EXT_color_buffer_float extension is supported.');
  }

  const maxColorAttachments = gl.getParameter(gl.MAX_COLOR_ATTACHMENTS)
  console.log('Maximum color attachments:', maxColorAttachments);

  const debugInfo = gl.getExtension("WEBGL_debug_renderer_info");

  if (!debugInfo) {
    console.warn("Error: WEBGL_debug_renderer_info is not supported.");
  } else {
    const vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
    console.log("Vendor:", vendor);

    const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
    console.log("Renderer:", renderer);
  }

  return { canvas, gl, maxColorAttachments };
}

import { initVertexShaderBuffer } from "./buffersAndTextures";
import { InitWebGLReturn, ProgramInfo } from "./types";

// Creates a WebGLProgram from two compiled WebGLShaders consisting of a vertex shader and a fragment shader.
// These shaders are compiled and linked to create a WebGLProgram.
function createWebGLProgramFromShaders(gl: WebGL2RenderingContext, vsSource: string, fsSource: string): WebGLProgram {
  var vertexShader = compileShader(gl, gl.VERTEX_SHADER, vsSource);
  var fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fsSource);

  // Create the shader program
  var shaderProgram = gl.createProgram();

  if (!shaderProgram) {
    throw new Error("Error creating shaderProgram.");
  }

  if (!vertexShader) {
    throw new Error("Error creating vertexShader.");
  }

  if (!fragmentShader) {
    throw new Error("Error creating fragmentShader.");
  }

  gl.attachShader(shaderProgram, vertexShader);
  gl.attachShader(shaderProgram, fragmentShader);
  gl.linkProgram(shaderProgram);

  // If creating the shader program failed, alert
  if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
    throw new Error('Unable to initialize the shader program: ' + gl.getProgramInfoLog(shaderProgram));
  }

  return shaderProgram;
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  var shader = gl.createShader(type);

  if (!shader) {
    throw new Error("Error calling gl.createShader(type).");
  }

  // Send the source to the shader object.
  gl.shaderSource(shader, source);

  // Compile the shader program.
  gl.compileShader(shader);

  // See if it compiled successfully.
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const errorMessage = 'An error occurred compiling the shaders: ' + gl.getShaderInfoLog(shader);
    console.error(errorMessage);
    console.error('Shader source:', source);
    gl.deleteShader(shader);
    throw new Error(errorMessage);
  }

  return shader;
}

export function createWebGLProgram(gl: WebGL2RenderingContext, vsSource: string, fsSource: string): ProgramInfo {
  // Initialize a shader program.
  var webGLProgram = createWebGLProgramFromShaders(gl, vsSource, fsSource);

  // Collect all the info needed to use the shader program.
  // Look up the locations of the attributes and uniforms used by our shader
  var programInfo: ProgramInfo = {
    program: webGLProgram,
    attribLocations: {
      vertexPosition: gl.getAttribLocation(webGLProgram, 'aVertexPosition'),
    },
  };

  // Handle vertex shader stuff.
  var positionBuffer = initVertexShaderBuffer(gl);

  {
    const numComponents = 2; // pull out 2 values per iteration (x and y coordinates).
    const type = gl.FLOAT; // the data in the buffer is 32bit floats.
    const normalize = false; // don't normalize.
    const stride = 0; // how many bytes to get from one set of values to the next.
    // 0 = use type and numComponents above.
    const offset = 0; // how many bytes inside the buffer to start from.
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

    // Specify how the vertex attribute receives its data from the buffer.
    gl.vertexAttribPointer(
      programInfo.attribLocations.vertexPosition, // location (index) of the vertex attribute in the shader program.
      numComponents,
      type,
      normalize,
      stride,
      offset);

    // Enable the vertex attribute array so that the vertex attribute / buffer is used when drawing
    // (by default, this is disabled).
    gl.enableVertexAttribArray(
      programInfo.attribLocations.vertexPosition);
  }

  return programInfo;
}


