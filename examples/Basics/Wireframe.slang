// A simple fragment shader-based antialiased wireframe shader.

// Camera uniforms
uniform float4x4 iViewProjection;

// Vertex shader input and output structs.
struct Vertex{
	float3 position : POSITION;
};
struct VsOutput{
	float4 svPosition : SV_Position;
};

// A basic vertex shader.
[shader("vertex")]
VsOutput vertexMain(Vertex input)
{
	VsOutput o;
	// Transform the vertices to clip space for perspective projection:
	o.svPosition = mul(float4(input.position, 1.0), iViewProjection);
	return o;
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
