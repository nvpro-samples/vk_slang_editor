// A simple tessellation shader.
// Since mesh shaders are generally faster at doing this nowadays,
// this will only be in the tests (to verify we can handle domain
// and hull shaders) until there's a mesh shader example.

// Camera uniforms
uniform float4x4 iViewProjection;
uniform float2 iResolution;

// GUI controls
uniform bool kConstantTessellation;
// These are used if kConstantTessellation is true:
uniform float kEdgeTessellationFactor;
uniform float kInsideTessellationFactor;
// And this is the goal pixels per tessellated edge if
// kConstantTessellation is false:
uniform float kGoalPixelsPerEdge;

// Vertex shader input and output structs.
struct Vertex{
	float3 position : POSITION;
};
struct VsOutput{
	Vertex original;
};

// The vertex shader does nothing; matrix transforms will happen
// in the domain shader.
[shader("vertex")]
Vertex vertexMain(Vertex input)
{
	return input;
}

// The hull shader (aka tessellation control shader in GLSL)
// turns patches (triangles in this case) into
// data for higher-order surfaces. In this case, we'll keep the
// same control points as before, but increase the tessellation level.
// See  https://www.khronos.org/opengl/wiki/Tessellation_Control_Shader

struct PatchConstantOutput{
	float edges[3] : SV_TessFactor;
	float inside : SV_InsideTessFactor;
};

PatchConstantOutput patchConstantMain(
	InputPatch<Vertex, 3> input
)
{
	PatchConstantOutput output;
	if(kConstantTessellation)
	{
		output.edges[0] = output.edges[1] = output.edges[2]
		                = kEdgeTessellationFactor;
		output.inside = kInsideTessellationFactor;
	}
	else
	{
		// There are much more sophisticated ways to do adaptive
		// subdivision; for the sake of this example, we just
		// subdivide each edge to try to reach a goal number of
		// pixels per subdivided edge.
		// First, project each point to the screen
		// (the offset doesn't matter here so we don't add it):
		float2 screenPositions[3];
		for(int i = 0; i < 3; i++)
		{
			const float4 hom = mul(float4(input[i].position, 1.0), iViewProjection);
			screenPositions[i] = (hom.xy / hom.w) * iResolution / 2.0;
		}
		// Now compute the edge factors; edge `i` connects the vertices
		// with the two indices not equal to `i`.
		for(int i = 0; i < 3; i++)
		{
			const float dist = distance(screenPositions[(i + 1)%3],
			                            screenPositions[(i + 2)%3]);
			// We clamp just for safety here:
			output.edges[i] = clamp(dist / kGoalPixelsPerEdge, 1.0, 32.0);
		}
		// For the interior, this isn't too bad as a starting point.
		output.inside = min3(output.edges[0], output.edges[1], output.edges[2]);
	}
	return output;
}

[shader("hull")]
[domain("triangle")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("patchConstantMain")]
Vertex hullMain(
	InputPatch<Vertex, 3> ip,
	uint i : SV_OutputControlPointID
)
{
	Vertex output;
	output.position = ip[i].position;
	return output;
}

struct DomainShaderOutput{
	float4 svPosition : SV_Position;
}

// Then the domain shader (aka tessellation evaluation shader in GLSL)
// evaluates where each tessellated vertex should be.
[shader("domain")]
[domain("triangle")]
DomainShaderOutput domainMain(
	PatchConstantOutput input,
	float3 bary : SV_DomainLocation,
	OutputPatch<Vertex, 3> patch
)
{
	const float3 pos = bary[0] * patch[0].position
	                 + bary[1] * patch[1].position
	                 + bary[2] * patch[2].position;
	// Now we apply the perspective transform:
	DomainShaderOutput output;
	output.svPosition = mul(float4(pos, 1.0), iViewProjection);
	return output;
}

// Approximates how much `v` changes over a pixel in screenspace.
float smoothFwidth(float v)
{
	return length(float2(ddx_fine(v), ddy_fine(v)));
}

[shader("fragment")]
float4 fragmentMain(float3 barycentrics : SV_Barycentrics,)
{
	float c = 1.0;
	for(int edge = 0; edge < 3; edge++)
	{
		const float b = barycentrics[edge];
		// This uses finite differences to approximate how much
		// `b` has to change to move a pixel in screen space.
		const float width = smoothFwidth(b);
		// If we didn't want to have antialiasing, the standard
		// approach would be to have a line like
		// ` if(b < 1.5 * width) c = 0.0; `
		// However, without too much extra cost, we can use a smooth
		// falloff function. This looks good because it's pretty
		// close to the integral of the Mitchell-Netravali
		// antialiasing kernel.
		// You can also use this same idea for SDFs;
		// https://blog.pkh.me/p/44-perfecting-anti-aliasing-on-signed-distance-functions.html
		// gives a good overview.
		// Note that we'll still have aliasing on silhouette edges;
		// the usual fixes for this are MSAA or post-process
		// antialiasing filters like DLSS or FXAA.
		c *= smoothstep(0.0, 1.5 * width, b);
	}
	// This sqrt is an approximate conversion from linear to sRGB;
	// this matches what should happen photometrically, so the
	// edge quality will look better.
	return float4(sqrt(float3(c)), 1.0);
}
