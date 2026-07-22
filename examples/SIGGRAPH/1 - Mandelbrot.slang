// Renders a Mandelbrot set. Try modifying this!
RWTexture2D<float4> texFrame; // Output texture
uniform float  iTime; // In seconds
uniform float2 iResolution; // Screen size
uniform float4 iMouse; // .xy = mouse position in pixels, .zw = left/right mouse buttons

static const uint kMaxIterations = 64;

uint mandelbrot(inout float2 z)
{
	const float2 c = z;
	for(uint i = 0; i < kMaxIterations; i++)
	{
		if(dot(z, z) > 256.0)
		{
			return i;
		}
		z = float2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y)
			+ c;
	}
	return kMaxIterations;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void render(uint2 thread: SV_DispatchThreadID)
{
	float2 uv = float2(thread) / iResolution;
	uv = 4.0 * uv - 2.0;
	uint i = mandelbrot(uv);
	float3 color = float3(0.0);
	if(i != kMaxIterations)
	{
		color = 0.5 + 0.5 * sin(i + float3(0, .5, 1)
			- log2(log2(dot(uv,uv))));
	}
	texFrame[thread] = float4(color, 1.0);
}
