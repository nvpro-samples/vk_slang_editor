// Demonstrates how to transform screen coordinates to ray origins and directions.
//
// Some things to note:
// - You'll see a Camera pane appear in the same tab group as
// Pipeline Statistics and Performance.
// - You'll also see a small transform gizmo at the bottom-left
// of the screen; that indicates the coordinate system the
// camera matrices use.

uniform uint2 iResolution; // Screen size

#if 1 // Change this to #if 0 to see the other method
// Easier (and more precise) way using vk_slang_editor's
// FragCoordToDirection and Eye uniforms
uniform float3 iEye;
uniform float3x3 iFragCoordToDirection;
void getRay(float2 fragCoord, out float3 origin, out float3 direction)
{
	origin = iEye;
	direction = mul(float3(fragCoord, 1.), iFragCoordToDirection);
}

#else
// Classic way using ClipToWorld.
// You'll notice that this also incorporates near-plane clipping.
uniform float4x4 iInverseViewProj; // inverse(mul(view, proj))
uniform float3 iEye;

float3 positionFromClip(float3 clip)
{
	float4 homogeneous = mul(float4(clip, 1.), iInverseViewProj);
	return homogeneous.xyz / homogeneous.w;
}

void getRay(float2 fragCoord, out float3 origin, out float3 direction)
{
	// UV:
	float2 uv = fragCoord / iResolution;
	// To clip-space:
	const float2 clipXY = 2 * uv - 1;
	// Get a point on the front clip plane:
	origin = positionFromClip(float3(clipXY, 0.));
	origin = iEye;
	// Then get a point anywhere further back in the volume:
	direction = positionFromClip(float3(clipXY, 0.999)) - origin;
}
#endif

// Rendering demo
RWTexture2D<float4> texFrame; // Output texture
uniform uint iFrame;
static const float TMAX = asfloat(0x7f800000); // infinity

struct HitResult
{
	float t;
	float3 normal;
	float3 albedo;
	__init()
	{
		t = TMAX;
		normal = albedo = 0;
	}
	[mutating]
	void update(float newT, float3 newNormal, float3 newAlbedo)
	{
		if(newT < t)
		{
			t = newT;
			normal = newNormal;
			albedo = newAlbedo;
		}
	}
}

HitResult raytraceScene(float3 o, float3 d)
{
	HitResult hit;
	// Sky
	hit.albedo = lerp(float3(1), float3(0.5,0.7,1), .5+.5*d.y);
	// Plane at y == -1: (o + d * t).y == -1
	{
		float t = (-1-o.y)/d.y;
		if(t > 0)
		{
			float3 p = o + d * t;
			int checkerboard = (int(floor(p.x)) ^ int(floor(p.z))) & 1;
			float3 albedo = lerp(0.2, 1.0, checkerboard);
			hit.update(t, float3(0,1,0), albedo);
		}
	}
	// Sphere of radius 1 at (0,0,0):
	// ||o + d * t||^2 == 1
	// -> dot(o, o) - 1 + 2 * dot(o, d) t + dot(d, d) t^2 == 0
	{
		float a = dot(d, d);
		float b = 2 * dot(o, d);
		float c = dot(o, o) - 1;
		float det = b * b - 4 * a * c;
		if(det > 0)
		{
			float t = (-b - sqrt(det)) / (2 * a);
			if(t > 0)
			{
				float3 p = o + d * t;
				float3 normal = p;
				hit.update(t, normal, float3(1.));
			}
		}
	}
	return hit;
}

float3 pathtrace(float3 o, float3 d)
{
	float3 throughput = 1.0;
	for(int bounce = 0; bounce < 128; bounce++)
	{
		HitResult hit = raytraceScene(o, d);
		throughput *= hit.albedo;
		if(hit.t == TMAX) {
			return throughput;
		}
		o = o + d * hit.t;
		if(dot(d, hit.normal) > 0) hit.normal *= -1;
		d = reflect(d, hit.normal);
		o += 1e-4 * d; // Move away from surface
	}
	// Never hit a light source
	return float3(0.);
}

// Condensed version of pcg_output_rxs_m_xs_32_32, with simple conversion to floating-point [0,1].
static uint rngState;
float rng()
{
	rngState = rngState * 747796405 + 1;
	uint word = ((rngState >> ((rngState >> 28) + 4)) ^ rngState) * 277803737;
	word      = (word >> 22) ^ word;
	return float(word) / exp2(32.);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 thread: SV_DispatchThreadID)
{
	// Jittering:
	rngState = (thread.y * iResolution.x) + thread.x;
	const float2 jitter = float2(rng(), rng()) - .5;
	
	// Ray:
	float3 origin, direction;
	getRay(float2(thread) + jitter, origin, direction);
	direction = normalize(direction);
	
	const float3 color = pathtrace(origin, direction);
	// Approximate linear to sRGB
	texFrame[thread] = float4(sqrt(color), 1.);
}
