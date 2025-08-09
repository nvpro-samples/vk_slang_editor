// This doesn't really work well. But geometry shaders do seem to function,
// at least.

uniform float kDistance;
// Camera uniforms
uniform float4x4 iViewProjection;
uniform float3 iEye;

// Vertex shader input and output structs.
struct Vertex{
	float3 position : POSITION;
	float3 normal : NORMAL;
};

// Vertex shader does nothing
[shader("vertex")]
Vertex vertexMain(Vertex input)
{
	return input;
}

struct FragmentInput{
	float3 normal : NORMAL;
};

struct GsOutput{
	FragmentInput fi = FragmentInput();
	float4 svPosition : SV_Position;	
};

float4 transformVertex(float3 v)
{
	return mul(float4(v, 1.0), iViewProjection);
}

// Geometry shader transforms vertices
[shader("geometry")]
[maxvertexcount(15)]
void geometryMain(triangle Vertex input[3], inout TriangleStream<GsOutput> outStream)
{
	// Copy this triangle to the output
	for(int i = 0; i < 3; i++)
	{
		GsOutput output;
		output.fi.normal = input[i].normal;
		output.svPosition = transformVertex(input[i].position);
		outStream.Append(output);
	}
	
	// Then for each edge, if the first vertex normal is facing towards
	// the camera but the second is facing away, create a fin:
	for(int edge = 0; edge < 3; edge++)
	{
		const Vertex v0 = input[edge];
		const Vertex v1 = input[(edge + 1) % 3];
		const float3 mid = (v0.position + v1.position) * 0.5f;
		const float3 midToEye = iEye - mid;
		if(dot(v0.normal, midToEye) >= 0
		&& dot(v1.normal, midToEye) < 0)
		{
			GsOutput output = {};
			output.svPosition = transformVertex(v0.position);
			outStream.Append(output);
			output.svPosition = transformVertex(v1.position);
			outStream.Append(output);
			output.svPosition = transformVertex(v0.position + kDistance * v0.normal);
			outStream.Append(output);
			
			outStream.Append(output);
			output.svPosition = transformVertex(v1.position);
			outStream.Append(output);
			output.svPosition = transformVertex(v1.position + kDistance * v1.normal);
			outStream.Append(output);
		}
	}
}

[shader("fragment")]
// The input to the fragment shader is all the fields of the vertex
// shader output, except the system value (SV_) ones.
// That's just `Vertex`!
float4 fragmentMain(FragmentInput fi)
{
	// Basic wrap shading
	float c = 0.0;
	const float l2 = dot(fi.normal, fi.normal);
	if(l2 > 0.0)
	{
		c = dot(fi.normal / sqrt(l2), float3(0.0, 1.0, 0.0)) * 0.5 + 0.5;
	}
	return float4(float3(c), 1.0);
}
