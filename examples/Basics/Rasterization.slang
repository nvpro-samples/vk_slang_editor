// Shows how to use vertex and fragment shaders by rasterizing a single object.
#define VARIANT 0

// Camera uniforms
uniform float4x4 iViewProjection;

// Vertex shader input and output structs.
struct Vertex{
	// Here are all the attributes vk_slang_editor can load
	// from glTF files.
	// You don't need to specify all of these.
	float3 position : POSITION;
	float3 normal : NORMAL;
	float4 tangent : TANGENT; // .w = bitangent sign
	float2 uv0 : TEXCOORD0;
	float2 uv1 : TEXCOORD1; // Two sets of UVs
	float4 color : COLOR;
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
	o.svPosition = mul(float4(o.vertex.position, 1.0), iViewProjection);
	return o;
}

[shader("fragment")]
// The input to the fragment shader is all the fields of the vertex
// shader output, except the system value (SV_) ones.
// That's just `Vertex`!
float4 fragmentMain(Vertex v,
	// There are many different system value inputs you can use in vertex
	// shaders; here are a few!
	// You can find more at https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/a2-01-spirv-target-specific.html#system-value-semantics.
	float2 fragCoord : SV_Position, // Pixel coordinate
	float3 barycentrics : SV_Barycentrics,
	uint triangleId : SV_PrimitiveID
)
{
#if VARIANT == 0
	// Here's the position we passed down from the vertex shader
	return float4(v.position, 1.0);
#elif VARIANT == 1
	// Normals
	return float4(v.normal, 1.0);
#elif VARIANT == 2
	// Tangents
	return float4(v.tangent.xyz, 1.0);
#elif VARIANT == 3
	// Bitangents (aka binormals), according to the glTF specification
	float3 bitangent = cross(v.normal, v.tangent.xyz) * v.tangent.w;
	return float4(bitangent, 1.0);
#elif VARIANT == 4
	// Texture coordinates
	return float4(v.uv0, 0.0, 1.0);
#elif VARIANT == 5
	// Shadertoy works by drawing a triangle or two over the entire
	// screen and using a pixel shader, but there's no reason you can't
	// do this with more general meshes:
	uint2 xy = uint2(fragCoord);
	bool sierpinski = ((xy.x & xy.y) == 0);
	float3 color = float3(sierpinski);
	return float4(color, 1.0);
#elif VARIANT == 6
	// Barycentric coordinates; useful for wireframes!
	return float4(barycentrics, 1.0);
#else
	// Triangle IDs.
	float3 color = fract(float3(0.18, 0.33, 0.45) * triangleId);
	return float4(color, 1.0);
#endif
}
