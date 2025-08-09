// 3D flame fractal with optional shared memory optimization.
// Based on https://wrighter.xyz/blog/2023_08_17_flame_fractals_in_comp_shader
// Ported to Slang and warp operations added by Nia Bickford

// Define this to get a slight improvement in speed by reducing shader
// divergence by using shared memory, at the expense of a more diffuse
// look for some of the arcs. See the places where this define appears
// below for an explanation of how this works.
#define WARP_OPTIMIZATION

static const uint BLOCK_SIZE_X = 16;
static const uint BLOCK_SIZE_Y = 16;
// This must be a power of 2:
static const uint BLOCK_SIZE = BLOCK_SIZE_X * BLOCK_SIZE_Y;

#ifdef WARP_OPTIMIZATION
groupshared float3 block_particles[BLOCK_SIZE];
#endif

uniform float iTime; // In seconds
uniform uint2 iResolution; // Screen size
uniform float2 fResolution;

// r32f means we use VK_FORMAT_R32_SFLOAT;
// globallycoherent means different threads can write to the same texel
// using atomics.
[format("r32f")] globallycoherent RWTexture2D<float> texAccumulate;
// We use rgba8 here just to show off an 8-bit backbuffer:
[format("rgba8")] RWTexture2D<float4> texFrame;

// The usual camera projection code
uniform float4x4 iView;
uniform float4x4 iProj;
uniform float3 iEye;
float2 projectParticle(float3 p, out float particle_depth)
{
	float4 w = mul(float4(p, 1.), iView);
	particle_depth = -w.z;
	w = mul(w, iProj);
	return w.xy / w.w;
}

// Random number generation
static uint rngState;
// Based on pcg_output_rxs_m_xs_32_32
uint hash(uint w)
{
	w = ((w >> ((w >> 28) + 4)) ^ w) * 277803737;
	return (w >> 22) ^ w;
}
float rng()
{
	rngState  = rngState * 747796405 + 1;
	return float(hash(rngState)) / exp2(32.f);
}

float2 randomPointInDisk()
{
	const float u1 = 2. * float.getPi() * rng();
	const float u2 = rng();
	return sqrt(u2) * float2(sin(u1), cos(u1));
}

// 2x2 matrix rotation
float2x2 rot(float angle)
{
	const float c = cos(angle);
	const float s = sin(angle);
	return float2x2(c, -s, s, c);
}

// Clears the accumulation texture every frame, since it's not cleared
// by default
[shader("compute")]
[numthreads(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1)]
void clear(uint2 thread: SV_DispatchThreadID)
{
	texAccumulate[thread] = 0;
}

uniform float kAperture;
uniform float kFocusDepthAdjust;

[shader("compute")]
[numthreads(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1)]
void main(uint2 thread: SV_DispatchThreadID,
          uint2 threadInGroup: SV_GroupThreadID)
{
	const float t = 0.1 * iTime;
	const float focusDepth = length(iEye);
	const uint gidLinear = threadInGroup.y * BLOCK_SIZE_X + threadInGroup.x;
	
	// Seed the RNG
	const uint particleIdx = thread.y * iResolution.x + thread.x;
	rngState = particleIdx;
	
	// Random initial position
	float3 p = float3(rng(), rng(), rng());
	
	const uint kMaxItersPerParticle = 25;
	for(uint i = 0; i < kMaxItersPerParticle; i++)
	{
		// On each iteration, choose a random transformation.
		// We get chaotic behavior by making each branch here
		// scatter particles away from each other, but keeping
		// them mostly within a bounded area.
#ifndef WARP_OPTIMIZATION
		const float r = rng();
#else
		// Because of shader divergence, if different particles
		// in a warp choose different branches, the warp has to run
		// the code for all branches.
		// We can get better performance by making all particles in
		// a warp go through the same branch.
		// But if we do that naively, we'll get a 'clumpy' look, since
		// groups of 32 particles will always go through the same
		// transformation.
		// To avoid this, we first shuffle particles between warps using
		// shared memory.
		GroupMemoryBarrierWithGroupSync(); // Wait for all warps to get here
		block_particles[gidLinear] = p;
		// Our permutation will be pretty simple:
		// x -> k * x mod BLOCK_SIZE, for k coprime to BLOCK_SIZE will
		// work. Since BLOCK_SIZE is a power of 2, we just need an
		// odd number:
		const uint k = 2 * hash(i) + 1;
		GroupMemoryBarrierWithGroupSync(); // Wait for all warps to finish writing
		p = block_particles[(k * gidLinear) % BLOCK_SIZE];
		// Make all particles in this warp go through the branch chosen
		// by the first particle in the warp.
		float r = rng();
		r = WaveReadLaneFirst(r);
#endif
		if(r < .3)
		{
			p = floor(p) + pow(fract(p), pow(sin(t), 4));
			p += 0.3;
			p.xz = mul(p.xz, rot(5.5 + 0.5 * sin(t)));
			p /= clamp(dot(p, p), -.2, 4.);
		}
		else if(r < .66)
		{
			p.xz = mul(p.xz, rot(5.2 + sin(float(particleIdx)*1e-5)));
			p.xy = mul(p.xy, rot(3.0 + 0.5 * cos(t)));
			p += float3(-1., .4, 0.);
			p /= clamp(dot(-p,p), -3.2, 1.);
			p *= float3(2, 1.5, 1.2) * 1.5;
		}
		else
		{
			p -= float3(-.2, .2, .2);
			p /= clamp(dot(p, p),-4.5, 10.);
			p *= float3(2, 1.5, 1.2) * 3.1;
		}
		
		// Splat the particle
		float depth;
		float2 ndc = projectParticle(p, depth);
		// Depth of field
		ndc += kAperture
			* (1. - kFocusDepthAdjust * focusDepth/depth)
			* randomPointInDisk() / iResolution;
		
		// Depth rejection; replace with if(true) for trippiness
		if(depth > 0.)
		{
			int2 intP = int2((.5 * ndc + .5) * fResolution);
			if(intP.x > 0 && intP.y > 0 && intP.x < iResolution.x && intP.y < iResolution.y)
			{
				texAccumulate.InterlockedAddF32(uint2(intP), .1f);
			}
		}
	}
}

uniform float3 paletteScale;
uniform float3 paletteBias;

float3 palette(float c)
{
	return 0.5 + 0.5 * sin(2.f * float.getPi()
	  * mad(c, paletteScale, paletteBias));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void display(uint2 thread: SV_DispatchThreadID)
{
	float count = texAccumulate[thread];
	float tonemapped = log(1. + count) * .1;
	float3 color = palette(tonemapped);
	color *= count;
	color = color / (1. + color);
	color = sqrt(color);
	
	texFrame[thread] = float4(color, 1.);
}
