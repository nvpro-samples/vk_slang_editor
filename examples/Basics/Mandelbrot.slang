// Shows basic functions and `for` loops using a Mandelbrot set.
// Try changing the shader structure and parameters to see what
// images you can get!
RWTexture2D<float4> texFrame; // Output texture
uniform float2 iResolution; // Screen size

// Returns whether the input point escaped the Mandelbrot iteration.
// `i` will be the iteration counter, and `z` will be updated to be
// the iterate position in the complex plane.
bool mandelbrot(inout float2 z, out int i)
{
	const float2 c = z;
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
	// Center the screen
	float2 uv = (float2(thread) - .5 * iResolution.xy)/iResolution.y;
	float2 z = 2.5 * uv - float2(.5,0); // Zoom out and adjust center
	int i;
	float3 color = float3(0.0);
	if(mandelbrot(z, i))
	{
		// Continuous color scheme
		color = 0.5 + 0.5 * sin(i + float3(0, .5, 1) - log2(log2(dot(z,z))));
	}
	texFrame[thread] = float4(color, 1.0);
}
