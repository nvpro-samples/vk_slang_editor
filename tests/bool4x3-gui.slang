RWTexture2D<float4> texFrame; // Output texture
uniform float iTime; // In seconds
uniform bool4x3 invert; // 4 rows, 3 columns
uniform float2 iResolution; // Screen size

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 thread: SV_DispatchThreadID)
{
	float2 uv = float2(thread) / iResolution;
	float4 color = float4(uv, 0.5 + 0.5 * sin(iTime), 1.0);
	const int2 dim = int2(invert[0].getCount(), invert.getCount());
	const int2 id = clamp(int2(uv * dim), 0, 3);
	if(invert[id.y][id.x])
	{
		color.rgb = 1.0 - color.rgb;
	}
	texFrame[thread] = color;
}
