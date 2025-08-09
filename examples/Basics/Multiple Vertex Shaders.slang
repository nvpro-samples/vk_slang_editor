// Shows how to use vertex and fragment shaders.
//
// This shader renders the skybox using a compute shader,
// then rasterizes a reflective model over it.
// It also shows briefly how you can define multiple shading
// passes in vk_slang_editor.

RWTexture2D<float4> texFrame; // Output texture
Sampler2D<float4> texDaytime; // HDR environment map
uniform float iTime;

// Camera uniforms
uniform float4x4 iWorldToClip;
uniform float3 iEye;
uniform float3x3 iFragCoordToDirection;

// This compute shader renders the skybox.
// See the Basics > Perspective example for how iFragCoordToDirection
// works.
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 thread: SV_DispatchThreadID)
{
	float3 direction = mul(float3(thread, 1.), iFragCoordToDirection);
	texFrame[thread] = float4(sky(direction), 1.0); // sky() is defined below
}

// Vertex shader input and output structs
struct Vertex{
	float3 position : POSITION;
	float3 normal : NORMAL;
};
struct VsOutput{
	Vertex vertex;
	float4 svPosition : SV_Position;
};

// A basic vertex shader.
// These run once per vertex; their function output is a struct
// that defines both the inputs to the next stage, as well as
// system value attributes like SV_Position (the clip-space position
// of the vertex).
[shader("vertex")]
VsOutput vertexMain(Vertex input)
{
	VsOutput o;
	o.vertex = input;
	// Transform the vertices to clip space for perspective projection:
	o.svPosition = mul(float4(o.vertex.position, 1.0), iWorldToClip);
	return o;
}

[shader("fragment")]
// The input to the fragment shader is all the fields of the vertex
// shader output, except the system value (SV_) ones.
// That's just `Vertex`!
float4 fragmentMain(Vertex v)
{
	// Get the ray from the point on the surface to the camera
	const float3 toEye = iEye - v.position;
	// Reflect along the normal
	const float3 n = v.normal;
	const float3 light = -reflect(toEye, n);
	return float4(sky(light), 1.0);
}

// If you define another vertex shader, vk_slang_editor will
// realize you want another pass over all the meshes in the scene,
// and create a new rendering pipeline.
[shader("vertex")]
VsOutput vertex2(Vertex input, uint vertexIdx : SV_VertexID)
{
	VsOutput o;
	o.vertex = input;
	// Add some waves to the shape:
	float displacement = sin(40.0 * o.vertex.position.y + iTime);
	o.vertex.position.xz *= 1.0 + 0.1 * displacement;
	// Here we translate it by a few units:
	o.vertex.position += float3(3, 0, 0);
	// Perspective projection:
	o.svPosition = mul(float4(o.vertex.position, 1.0), iWorldToClip);
	return o;
}

// vk_slang_editor tries to infer how you want to create rendering passes
// based on the shaders in the file and how they're ordered.
// `vertexMain` and `fragmentMain` above form a complete pass; since the
// last pass already had a vertex shader, adding `vertex2` started a new
// one.
// Now, when the fragment shader below is defined, it matches with
// an empty slot in `vertex2`'s pass. If it's not defined, then
// `vertex2`'s is missing a fragment shader, so its pass will use the
// last one that was defined -- `fragmentMain`.
// You can also see this effect in how the passes are listed in the
// Performance window.
//
// Try it out by changing this `#if 0` to an `#if 1`!
// And please note that this behavior is specific to vk_slang_editor;
// in your own engine, you can create passes in any way you want.
#if 0
[shader("fragment")]
float4 surfaceWorldPosition(Vertex input, uint triangleID : SV_PrimitiveID)
{
	const float3 pos = input.position;
	const float3 triangleColor = 0.5 + 0.5
		* sin(triangleID * float3(0.6, 1.0, 1.6));
	return float4(triangleColor, 1.0);
}
#endif

// Utility functions
float3 sky(float3 direction)
{
	direction = normalize(direction);
	float2 uv = float2(atan2(direction.z, direction.x), acos(direction.y));
	uv *= float2(0.5 / float.getPi(), 1.0 / float.getPi());
	return texDaytime.Sample(uv).rgb;
}
