// Renders Julia sets using the mouse position.
// Try changing the shader structure and parameters to see what
// images you can get!
RWTexture2D<float4> texFrame; // Output texture
uniform float2 iResolution; // Screen size
uniform int4 iMouse; // .xy = pos in pixels, .z = LMB, .w = RMB

// Returns whether the input point escaped the Julia iteration.
// `i` will be the iteration counter, and `z` will be updated to be
// the iterate position in the complex plane.
bool julia(inout float2 z, const float2 c, out int i)
{
	for(i = 0; i < 64; i++)
	{
		if(dot(z, z) > 256.0) // Did we escape?
		{
			return true;
		}
		// Complex multiplication: z = z^2 + c
		z = c + float2(z.x * z.x - z.y * z.y, 2 * z.x * z.y);
	}
	return false;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void render(uint2 thread: SV_DispatchThreadID)
{
	const float2 uv = (float2(thread) - .5 * iResolution.xy)/iResolution.y;
	float2 z = uv * lerp(3.0, 1.0, iMouse.z);
	const float2 c = 2.0 * (iMouse.xy / iResolution - .5);
	int i;
	float3 color = float3(0.0);
	if(julia(z, c, i))
	{
		// Continuous color scheme
		color = 0.5 + 0.5 * sin(i + float3(0, .5, 1) - log2(log2(dot(z,z))));
	}
	texFrame[thread] = float4(color, 1.0);
}
