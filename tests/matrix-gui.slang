// Tests that matrices appear correctly in the Shader Parameters window.
// You should see a very distorted shaderball.

// Camera uniforms
uniform float4x4 iViewProjection;
uniform half4x3 customMat;

struct Vertex{
	float3 position : POSITION;
};
struct VsOutput{
	Vertex vertex;
	float4 svPosition : SV_Position;
};

[shader("vertex")]
VsOutput vertexMain(Vertex input)
{
	VsOutput o;
	o.vertex = input;
	o.vertex.position = mul(float4(o.vertex.position, 1.0), customMat);
	// Transform the vertices to clip space for perspective projection:
	o.svPosition = mul(float4(o.vertex.position, 1.0), iViewProjection);
	return o;
}

[shader("fragment")]
float4 fragmentMain(Vertex v)
{
	return float4(v.position, 1.0);
}
