// Slang's autodifferentiation makes generating normals from procedural displacement easier.
//
// This shader renders the skybox using a compute shader,
// then rasterizes a reflective model over it.
// It also shows briefly how you can define multiple shading
// passes in vk_slang_editor.

RWTexture2D<float4> texFrame; // Output texture
Sampler2D<float4> texDaytime; // HDR environment map
uniform float iTime;

// Camera uniforms
uniform float4x4 iWorldToClip;
uniform float3 iEye;
uniform float3x3 iFragCoordToDirection;

// Shader parameters
uniform float3 frequencyScale;
uniform float3 frequencyOffset;
uniform float waveScale;

// Displaces a vertex and gives its new position, given its
// position `v` and normal `n`.
//
// This can be as complicated as you like; we'll use Slang's
// autodiff feature to automatically compute its derivatives
// and correctly compute normals in the fragment shader.
//
// The default one we provide simulates oceam waves.
// Computing normals for this surface by hand is fairly
// complex and error-prone, but Slang can do it for us.
// Try modifying this function yourself and see what sorts
// of shapes you can make!
[Differentiable]
float3 displace(float3 v, no_diff float3 n)
{
	float3 displacement = 0.0;
	// Multiple waves
	for(int i = 0; i < 4; i++)
	{
		const float3 frequency = fma(no_diff randomFrequency3D() - .5,
			frequencyScale, frequencyOffset);
		const float phase = no_diff randomUniform();
		
		const float input = dot(v, frequency) + iTime + phase;
		
		// Ocean waves have both vertical and horizontal motion;
		// this creates their characteristic peaks and troughs.
		// This is the amount of motion that moves in the same plane
		// as `frequency`:
		const float lateral        = waveScale * cos(input);
		// And this is the amount of motion that moves perpendicular
		// to it -- we choose to use the normal here so it more closely
		// corresponds to moving the surface in and out:
		const float transverse     = waveScale * sin(input);
		const float3 lateralDir    = normalize(frequency);
		const float3 transverseDir = normalize(n - dot(n, lateralDir));
		displacement += lateral * lateralDir
			+ transverse * transverseDir;
	}
	return v + displacement;
}

// This compute shader renders the skybox.
// See the Basics > Perspective example for how iFragCoordToDirection
// works.
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 thread: SV_DispatchThreadID)
{
	float3 direction = mul(float3(thread, 1.), iFragCoordToDirection);
	texFrame[thread] = float4(sky(direction), 1.0); // sky() is defined below
}

// Vertex shader input and output structs
struct Vertex{
	float3 position : POSITION;
	float3 normal : NORMAL;
	float4 tangent : TANGENT;
};
struct VsOutput{
	Vertex original;
	float3 originalBitangent; // For convenience
	float4 svPosition : SV_Position;
};

[shader("vertex")]
VsOutput vertexMain(Vertex input)
{
	initRng();
	VsOutput o;
	// Copy original attributes
	o.original = input;
	o.originalBitangent = cross(input.normal, input.tangent.xyz)
		* input.tangent.w;
	// Displacement
	float3 pos = displace(input.position, input.normal);
	// Transform the vertices to clip space for perspective projection:
	o.svPosition = mul(float4(pos, 1.0), iWorldToClip);
	return o;
}

[shader("fragment")]
// The input to the fragment shader is all the fields of the vertex
// shader output, except the system value (SV_) ones.
// That's just `Vertex`!
float4 fragmentMain(VsOutput v)
{
	initRng();
	// Compute the new normal based on the derivatives of the
	// displacement.
	// Our original tangent plane was
	// `pos + u * tangent + v * bitangent`.
	// Displaced, we get the surface
	// `displace(pos + u * tangent + v * bitangent)`.
	// So our new tangent is the derivative of `displace` in the direction 
	// of `tangent`, and our new bitangent is the derivative of `displace`
	// in the direction of `bitangent`.
	// Slang's autodiff functionality makes this easy!
	
	const float3 newTangent = fwd_diff(displace)(
		diffPair(v.original.position, v.original.tangent.xyz),
		v.original.normal
	).d;
	
	const float3 newBitangent = fwd_diff(displace)(
		diffPair(v.original.position, v.originalBitangent),
		v.original.normal
	).d;
	
	// Then take the cross product of these and normalize
	// to get the new normal:
	const float3 newNormal = normalize(
		cross(newTangent, newBitangent) * v.original.tangent.w
	);
	
	// Get the ray from the point on the surface to the camera
	const float3 toEye = iEye - v.original.position;
	// Reflect along the normal
	float3 light = -reflect(toEye, newNormal);
	// If we reflected into the surface, reflect it back out
	if(dot(light, v.original.normal) < 0.0)
	{
		light = reflect(light, v.original.normal);
	}
	return float4(sky(light), 1.0);
}

//--------------------------------------------------------------------------
// Utility functions
static uint g_rngState;
void initRng()
{
	g_rngState = 123;
}
// Returns a uniform random value in [0,1].
float randomUniform()
{
  // Condensed version of pcg_output_rxs_m_xs_32_32,
  // with simple conversion to floating-point [0,1].
  g_rngState  = g_rngState * 747796405 + 1;
  uint word   = ((g_rngState >> ((g_rngState >> 28) + 4)) ^ g_rngState) * 277803737;
  word        = (word >> 22) ^ word;
  return float(word) / exp2(32.0);
}
// Returns a random number from a non-uniform distribution.
float randomFrequency()
{
	// This samples a random frequency from a simplified version
	// of the JONSWAP model, https://www.codecogs.com/library/engineering/fluid_mechanics/waves/spectra/jonswap.php,
	// with wp=1, alpha = 1, beta = 1, gamma = 1, and g = 1.
	return pow(-log(max(1e-6, randomUniform())), .25);
}
// Returns a float3 where each component has the same distribution as
// randomFrequency().
float3 randomFrequency3D()
{
	return float3(randomFrequency(), randomFrequency(), randomFrequency());
}

float3 sky(float3 direction)
{
	direction = normalize(direction);
	float2 uv = float2(atan2(direction.z, direction.x), acos(direction.y));
	uv *= float2(0.5 / float.getPi(), 1.0 / float.getPi());
	return texDaytime.Sample(uv).rgb;
}
