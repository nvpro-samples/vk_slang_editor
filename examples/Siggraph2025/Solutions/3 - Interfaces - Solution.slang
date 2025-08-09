// The solution to 3 - Interfaces.

// This imports interface_scene.slang next to this file as a module.
// (this uses a path only because we're in a different folder than
// the puzzle)
import "../interface_scene.slang";

// Here's the solution:
struct MirrorBrdf : IBrdf 
{
	static float3 sample(HitInfo hit, inout RNG rng)
	{
		return reflect(hit.direction, hit.normal);
	}
}

// The start of interface_scene has these types:
#if 0
public struct HitInfo {
  public float3 direction; // Input direction, pointing into the surface.
  public float3 normal;    // Surface normal.
}

public interface IBrdf {
  // Randomly samples and returns a new direction.
  // `rng` is an optional state for a random number generator.
  // You do not need it to solve this puzzle.
  static float3 sample(HitInfo hit, inout RNG rng);
}
#endif

// Example diffuse BRDF.
struct DiffuseBrdf : IBrdf
{
	static float3 sample(HitInfo hit, inout RNG rng)
	{
		// Uniform sphere to oriented cosine-weighted hemisphere
		// using Peter Shirley's trick
		return normalize(hit.normal + uniformSampleSphere(rng.get2D()));
	}
}


























//-----------------------------------------------------------------------
// The rest of the code below is the compute path tracer.
// The one part of it that may be interesting is that we can create
// the scene and call into it like this:
//
//	example::Scene<DiffuseBrdf, MirrorBrdf> scene;
//	float3 color = scene.pathtrace(ray, rng);
//
// You don't need to look at it to solve the puzzle!







#define MOTION
#ifdef MOTION
uniform float iTime;
float getRotation()
{
	return 0.07 * iTime;
}
#else
float getRotation() { return 0.0f; }
#endif

float3 uniformSampleSphere(float2 u)
{
	const float z = 2.f * u.x - 1.f;
	const float r = sqrt(max(0.F, 1.F - z * z));
	const float phi = 2.f * float.getPi() * u.y;
	return float3(r * float2(cos(phi), sin(phi)), z);
}


RWTexture2D<float4> texFrame; // Output texture
[format("rgba32f")]
RWTexture2D<float3> texAccumulation; // Linear-space accumulation
uniform uint2 iResolution;
uniform float2 fResolution;
uniform uint iFrame; // Frame counter
uniform float fTime;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 thread: SV_DispatchThreadID)
{
	// Seed RNG
	uint2 seed = thread;
#ifndef MOTION
	seed.y += iFrame * iResolution.y;
#endif
	
	float3 color = 0.0f;
	const uint kNumSamples = 8;
	for(uint sampleIndex = 0; sampleIndex < kNumSamples; sampleIndex++)
	{
		RNG rng = RNG(seed, sampleIndex, kNumSamples);
		// Jitter over a pixel
		const float2 pixel = float2(thread) + rng.get2D() - .5;
		const float2 uv = (2.0 * pixel - fResolution.xy) / fResolution.y;
		// Right-handed Z-up coordinate system, same as Blender's
		example::Ray ray;
		ray.origin = float3(0, -4, 4);
		static const float kSqrtP5 = sqrt(.5);
		ray.direction = float3(
			uv.x,
			 kSqrtP5 -kSqrtP5 * uv.y,
			-kSqrtP5 -kSqrtP5 * uv.y
		);
		ray.direction = normalize(ray.direction);
		
		const float r = getRotation();
		const float2x2 rotMatrix = float2x2(cos(r), -sin(r),
		                                    sin(r), cos(r));
		ray.origin.xy    = mul(ray.origin.xy, rotMatrix);
		ray.direction.xy = mul(ray.direction.xy, rotMatrix);

		example::Scene<DiffuseBrdf, MirrorBrdf> scene;
		color += scene.pathtrace(ray, rng);
	}
	color /= float(kNumSamples);
	
#ifndef MOTION
	// Exponential moving average over time
	if(iFrame != 0)
	{
		const float3 previous = texAccumulation[thread].rgb;
		color = lerp(previous, color, max(1./float(iFrame + 1),1e-3));
	}
#endif
	texAccumulation[thread] = color;
	texFrame[thread] = float4(sqrt(color), 1.0);
}
