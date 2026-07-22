// Puzzle: Fill in the 4 missing types, based on the comments.
// We've filled in one type to get you started.

// A shader parameter of type `uint`:
uniform uint kNumParticles;

// A readable and writable buffer of `Particle`s.
// (Slang lets us define the Particle struct anywhere in the file.)
// This one has been completed for you.
RWStructuredBuffer<Particle> particles;

// A readable and writable buffer of `uint`s:
RWStructuredBuffer<uint> tileParticleIds;

// A 2D readable and writable texture.
RWTexture2D texFrame;

// A 2D combined texture and sampler.
Sampler2D texPuzzleMedian; 

// Once you've added the 4 missing types,
// press F3 to compile and run!
// If there are no errors in the Diagnostics window,
// you've solved it correctly.


















//-----------------------------------------------------------------------
// The rest of the text below implements the shader that appears
// when you solve the puzzle;
// don't look at this!






















struct Particle
{
	float2 uv;
	float2 velocity;
}

// Computes the median of three numbers.
float median(float3 v)
{
    return max(min(v.x, v.y), min(max(v.x, v.y), v.z));
}

// Some additional parameters we didn't want to include in the puzzle.
globallycoherent RWByteAddressBuffer tileStarts;
uniform uint2 iResolution;
uniform uint iFrame;
uniform float iTime;

// Our main texture here is square; this function fits
// it on the screen with uniform scaling.
// Coordinates inside [0,1)^2 are inside the texture.
float2 screenToUv(uint2 thread)
{
	float scaleFactor = 1.f / min(iResolution.x, iResolution.y);
	return (float2(thread) - .5 * iResolution) * scaleFactor + .5;
}

// Inverse of screenToUv.
float2 uvToScreen(float2 uv)
{
	float rcpScaleFactor = min(iResolution.x, iResolution.y);
	return (uv - .5) * rcpScaleFactor + .5 * iResolution;
}

// Must match the tile size used for "tileData".
static const uint kTileSize = 16;

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

[shader("compute")]
[numthreads(128, 1, 1)]
void clear(uint2 thread: SV_DispatchThreadID)
{
	if(any(thread >= iResolution)) return;
	
	const uint pixel1D = thread.y * iResolution.x + thread.x;
	rngState = pixel1D;
	if(iFrame == 0 && pixel1D < kNumParticles)
	{
		// Randomize particles
		Particle particle;
		particle.uv = float2(rng(), rng());
		particle.velocity = float2(rng(), rng()) * 2. - 1.;
		particle.velocity *= .1;
		
		particles[pixel1D] = particle;
	}
	
	const uint2 numTiles = getNumTiles();
	if(pixel1D < (kTileStart + kTileStride * numTiles.x * numTiles.y)/4)
	{
		tileStarts.Store(4 * pixel1D, 0);
	}
}

// Abstraction for working with `tileStarts`.
// Note that we could just as well do this with AppendStructuredBuffer,
// but we didn't mention that in detail in the lab so people might
// not know about it.
static const uint kTileCounter = 0;
static const uint kTileStart   = sizeof(uint);
static const uint kTileStride  = sizeof(uint2);
uint2 getNumTiles()
{
	return uint2(iResolution + kTileSize - 1) / kTileSize;
}
uint tile1DToByte(uint tile1D)
{
	return kTileStart + kTileStride * tile1D;
}
uint tile2DToByte(uint2 tile)
{
	const uint2 numTiles = getNumTiles();
	return tile1DToByte(tile.y * numTiles.x + tile.x);
}
bool tile2DOk(uint2 tile)
{
	return all(tile >= 0) && all(tile < getNumTiles());
}

// Count how many particles lie within each screen-sized tile.
// Tile data -> uint2(count, 0)
[shader("compute")]
[numthreads(128, 1, 1)]
void computeTileSizes(uint2 thread: SV_DispatchThreadID)
{
	// For each particle...
	if(any(thread >= iResolution)) return;
	const uint particleId = thread.y * iResolution.x + thread.x;
	if(particleId >= kNumParticles) return;
	Particle particle = particles[particleId];
	
	const int2 tile = int2(uvToScreen(particle.uv)) / kTileSize;
	if(!tile2DOk(tile)) return;
	tileStarts.InterlockedAdd(tile2DToByte(uint2(tile)), 1);
}

// Assign tiles to offsets within tileParticleIds.
// Note that this method is non-deterministic. In fact, there are
// few enough tiles that we could have a single warp do a prefix sum
// over all tiles and have it be performant!
// But for the design of this lab, we use the separate approach
// because it lets us justify using RWByteAddressBuffer more easily.
// Tile data -> uint2(start, start)
[shader("compute")]
[numthreads(128, 1, 1)]
void computeTileOffsets(uint2 thread: SV_DispatchThreadID)
{
	if(any(thread >= iResolution)) return;
	const uint tile = thread.y * iResolution.x + thread.x;
	const uint2 numTiles = getNumTiles();
	if(tile >= numTiles.x * numTiles.y) return;
	
	const uint tileByte = tile1DToByte(tile);
	uint particlesInTile = tileStarts.Load(tileByte);
	uint tileOffset;
	tileStarts.InterlockedAdd(kTileCounter, particlesInTile, tileOffset);
	tileStarts.Store2(tileByte, uint2(tileOffset, tileOffset));
}

// Bin particles into tiles.
// Tile data -> uint2(start, end)
[shader("compute")]
[numthreads(128, 1, 1)]
void binParticles(uint2 thread: SV_DispatchThreadID)
{
	// For each particle...
	if(any(thread >= iResolution)) return;
	const uint particleId = thread.y * iResolution.x + thread.x;
	if(particleId >= kNumParticles) return;
	Particle particle = particles[particleId];
	
	const int2 tile = int2(uvToScreen(particle.uv)) / kTileSize;
	if(!tile2DOk(tile)) return;
	uint writeIndex;
	tileStarts.InterlockedAdd(tile2DToByte(uint2(tile)) + sizeof(uint),
		1, writeIndex);
	tileParticleIds[writeIndex] = particleId;
}

// Draws an antialiased circle for a particle on top of a given
// premultiplied color.
void compParticle(inout float4 color, Particle particle, uint2 thread)
{
	const float d = length(uvToScreen(particle.uv) - float2(thread));
	const float opacity = smoothstep(3, 2, d);
	const float3 rgb = float3(pow(abs(particle.velocity), .8), 0.0);
	color = lerp(color, float4(rgb, 1.0), opacity);
}

struct Entry
{
	int id = -1;
	float dist = 1e6;
};
// The number of closest particles to find.
static const uint kNumClosest = 6;
// Safety factor for particle searches.
static const uint kMaxParticlesPerTile = 32;
// Finds the NumClosest particles closest to `uv`,
// ignoring any particles that had a uv coordinate exactly equal
// to `uv`.
Array<Entry, N> findClosest<uint N>(float2 uv, int2 myTile)
{
	Array<Entry, N> closest;
	for(int n = 0; n < N; n++)
	{
		closest[n] = Entry();
	}
	
	for(int dy = -1; dy <= 1; dy++)
	{
		for(int dx = -1; dx <= 1; dx++)
		{
			const int2 otherTile = int2(myTile) + int2(dx, dy);
			if(any(otherTile < 0) || any(otherTile >= getNumTiles()))
				continue;
			const uint2 otherData = tileStarts.Load2(
				tile2DToByte(otherTile));
			const uint bound = min(otherData.y,
				otherData.x + kMaxParticlesPerTile);
			for(uint j = otherData.x; j < bound; j++)
			{
				const uint otherId = tileParticleIds[j];
				const Particle other = particles[otherId];
				if(all(other.uv == uv)) continue;
				
				// Insertion sort by distance
				const float d = length(other.uv - uv);
				int n = 0;
				for(; n < kNumClosest; n++)
				{
					const float tmp = closest[n].dist;
					if(tmp <= d) break;
					if(n > 0) closest[n-1] = closest[n];
				}
				if(n > 0) closest[n-1] = {otherId, d};
			}
		}
	}
	
	return closest;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void render(uint2 thread: SV_DispatchThreadID)
{
	if(any(thread >= iResolution)) return;
	
	float3 bg = float3(1.);
	float4 fg = float4(0.);
	
	const uint2 myTile = thread/kTileSize;
	
	// Since lines from particles in adjacent tiles can cross
	// this pixel, we need an outer "for each particle in each
	// adjacent tile..." loop
	for(int dy = -1; dy <= 1; dy++)
	{
		for(int dx = -1; dx <= 1; dx++)
		{
			const int2 tile = int2(myTile) + int2(dx, dy);
			if(!tile2DOk(tile)) continue;
			const uint2 tileData = tileStarts.Load2(tile2DToByte(tile));
			const uint bound = min(tileData.y,
				tileData.x + kMaxParticlesPerTile);
			for(uint j = tileData.x; j < bound; j++)
			{
				const uint id = tileParticleIds[j];
				const Particle p = particles[id];
				// Composite particle to foreground
				compParticle(fg, p, thread);
				
				// Find the `N` closest particles:
				const Array<Entry, kNumClosest> closest
					= findClosest<kNumClosest>(p.uv, tile);
				
				// Now draw lines to the closest particles:
				const float2 myPx = uvToScreen(p.uv);
				const float2 deltaPx = float2(thread) - myPx;
				for(int n = 0; n < kNumClosest; n++)
				{
					const int otherId = closest[n].id;
					if(otherId < 0) continue;
					const Particle other = particles[otherId];
					const float2 otherPx = uvToScreen(other.uv);
					
					const float2 lineDir = otherPx - myPx;
					const float lineLen = length(lineDir);
					const float along = dot(deltaPx, lineDir) / lineLen;
					const float across = dot(deltaPx,
						float2(-lineDir.y, lineDir.x)) / lineLen;
					
					float opacity = smoothstep(1.5, 0, abs(across));
					opacity *= smoothstep(-.5, .5, along);
					opacity *= smoothstep(lineLen + .5, lineLen - .5, along);
					opacity *= 0.5;
					bg *= (1 - opacity);
				}
			}
		}
	}
	
	// Composite:
	bg = bg * (1. - fg.a) + fg.rgb;
	// Write to screen, with approx. linear -> sRGB correction
	
	// Did you get an error here? Then your type for `texFrame`
	// is wrong.
	// We want a type that (1) only lists the texture, (2) is
	// Readable and Writeable, and (3) is 2D.
	texFrame[thread] = float4(sqrt(bg), 1.);
}

// Returns a gradient pointing towards the text; we'll use this
// to steer particles.
// Outside the text, the gradient is normalized; inside, it is 0.
float2 textGradient(float2 uv)
{
	// Near the edges of the texture, point back inwards
	const float2 rect = float2(0.02, 0.17);
	if(any(uv < rect) || any(uv >= 1-rect))
	{
		const float2 fromEdge = uv - clamp(uv, rect, 1-rect);
		return normalize(-fromEdge);
	}
	// Did you get an error message here? Then your type for
	// `texPuzzleMedian` is wrong.
	// Hint: Texture = texture but no sampler,
	// SamplerState = sampler but no texture,
	// Sampler = combined texture and sampler (new in Slang)
	const float center = median(texPuzzleMedian.SampleLevel(uv, 0).rgb);
	if(center > 0.5)
	{
		return 0.0;
	}
	// Elsewhere, use finite differences.
	// Worth noting that the choice to use a median SDF in this
	// puzzle may have been a poor one -- we could have encoded all
	// this as a texture instead!
	const float dEps = 2e-3;
	const float xp = median(texPuzzleMedian.SampleLevel(
	                        uv + float2(dEps, 0), 0).rgb);
	const float yp = median(texPuzzleMedian.SampleLevel(
	                        uv + float2(0, dEps), 0).rgb);
	const float2 grad = normalize(float2(xp, yp)-center);
	if(any(isnan(grad))) return float2(1,0);
	return grad;
}

static const float kRepulsionDist  = 0.005;
static const float kRepulsionForce = 0.2;

// Moves particles.
[shader("compute")]
[numthreads(128, 1, 1)]
void animate(uint2 thread: SV_DispatchThreadID)
{
	if(any(thread >= iResolution)) return;
	const uint particleId = thread.y * iResolution.x + thread.x;
	if(particleId >= kNumParticles) return;
	
	Particle particle = particles[particleId];
	
	const float dt = 5e-3;
	
	float2 acceleration = textGradient(particle.uv);
	
	// Also, find the closest particles and try to move away from them.
	int2 myTile = int2(uvToScreen(particle.uv) / kTileSize);
	Array<Entry, 1> closest = findClosest<1>(particle.uv, myTile);
	if(closest[0].id >= 0)
	{
		const float2 closestDelta = particle.uv - particles[closest[0].id].uv;
		const float len = length(closestDelta);
		acceleration += kRepulsionForce * smoothstep(kRepulsionDist, 0, len) * normalize(closestDelta);
	}
	
	particle.velocity += dt * acceleration;
	particle.velocity *= 0.995; // Some friction
	particle.uv += dt * particle.velocity;
	
	// Sanitize positions
	if(any(isnan(particle.uv))) particle.uv = float2(0.0);
	if(any(isnan(particle.velocity))) particle.velocity = float2(0.0);
	
	particles[particleId] = particle;
}

// If you're developing this puzzle, these debug visualizations
// might come in handy.
#if 0
[shader("compute")]
[numthreads(16, 16, 1)]
void debugTexture(uint2 thread: SV_DispatchThreadID)
{
	if(any(thread >= iResolution)) return;
	float2 uv = screenToUv(thread);
	float2 grad = textGradient(uv);
	float4 color = float4(0.5 + 0.5 * grad, 0.0, .5);
	
	// Debug tile info
	const uint2 tile = thread / kTileSize;
	const uint tileByte = tile2DToByte(tile);
	const uint2 tileData = tileStarts.Load2(tileByte);
	
	int len = tileData.y - tileData.x;
	color.b = float(abs(len)) / 4.0;
	if(false && tileData.y > tileData.x)
	{
		// Debug tiled particle IDs
		const uint firstParticleId = tileParticleIds[tileData.x];
		Particle firstParticle = particles[firstParticleId];
		color.rg = firstParticle.uv;
	}
	
	float4 input = texFrame[thread];
	texFrame[thread] = lerp(input, color, color.a);
}
#endif

#if 0
[shader("compute")]
[numthreads(128, 1, 1)]
void debugParticles(uint2 thread: SV_DispatchThreadID)
{
	if(any(thread >= iResolution)) return;
	const uint particleId = thread.y * iResolution.x + thread.x;
	if(particleId >= kNumParticles) return;
	
	Particle particle = particles[particleId];
	const float2 pixel = uvToScreen(particle.uv);
	texFrame[uint2(pixel)] = float4(0.0,0.0,1.0,1.0);
}
#endif