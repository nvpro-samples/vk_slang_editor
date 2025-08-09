RWTexture2D<float4> texFrame; // Output texture

struct MaterialLayer
{
	Texture2D t[3];
	float4 c;
	SamplerState s;
}
struct Material
{
	int n;
	MaterialLayer layers[5];
}
MaterialLayer mat;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 thread: SV_DispatchThreadID)
{
  float4 color = float4(0.67, 0.09, 0.17, 1.0);
  texFrame[thread] = color;
}
