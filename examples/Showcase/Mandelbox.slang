// Renders a Mandelbox fractal using Mikael Hvidtfeldt Christensen's distance estimator.
// http://blog.hvidtfeldts.net/index.php/2011/11/distance-estimated-3d-fractals-vi-the-mandelbox/

RWTexture2D<float4> texFrame; // Output texture
uniform uint2 iResolution; // Screen size

uniform float3 iEye;
uniform float3x3 iFragCoordToDirection;
void getRay(float2 fragCoord, out float3 origin, out float3 direction)
{
	origin = iEye;
	direction = mul(float3(fragCoord, 1.), iFragCoordToDirection);
}

uniform float4x4 iView;

uniform float kToleranceFactor;

float getTolerance(float depth)
{
	return kToleranceFactor * depth;
}

uniform float kScale;
uniform float kMinRadius;
uniform float kFixedRadius;
uniform float kFoldingLimit;

void sphereFold(inout float3 z, inout float dz)
{
	const float r2 = dot(z, z);
	const float kFixedRadius2 = kFixedRadius * kFixedRadius;
	const float kMinRadius2 = kMinRadius * kMinRadius;
	float factor = 1.0f;
	if(r2 < kMinRadius2)
	{
		// linear inner scaling
		factor = kFixedRadius2 / kMinRadius2;
	}
	else if(r2 < kFixedRadius2)
	{
		// sphere inversion
		factor = kFixedRadius2 / r2;
	}
	z *= factor;
	dz *= factor;
}

void boxFold(inout float3 z, inout float dz)
{
	z = clamp(z, -kFoldingLimit, kFoldingLimit) * 2.0 - z;
}

uniform int kMandelboxIterations;

// Distance estimator
float map(float3 p)
{
	const float3 c = p;
	float dr = 1.0;
	for(int n = 0; n < kMandelboxIterations; n++) {
		boxFold(p, dr);
		sphereFold(p, dr);
		p = kScale * p + c;
		dr = dr * abs(kScale) + 1.0;
	}
	return length(p) / abs(dr);
}

static const float kInfinity = asfloat(0x7f800000);
uniform float kTMax;
uniform int kTraceIterations;

// Over-relaxation sphere tracing loop
float trace(float3 pos, float3 dir, out float candidateError)
{
	float omega = 1.5;
	const float pixelRadius = getTolerance(1.);
	float t = 0.0;
	candidateError = kInfinity;
	float candidateT = 0.0;
	float previousRadius = 0.0;
	float stepLength = 0.0;
	
	t = 0.0f;
	for(int i = 0; i < kTraceIterations; i++)
	{
		const float radius = map(pos + t * dir);
		
		const bool sorFail = omega > 1 && (radius + previousRadius) < stepLength;
		if(sorFail)
		{
			stepLength -= omega * stepLength;
			omega = 1;
		}
		else
		{
			stepLength = radius * omega;
		}
		
		previousRadius = radius;
		const float error = radius / t;
		
		if(!sorFail && error < candidateError)
		{
			candidateT = t;
			candidateError = error;
		}
		
		if(!sorFail && error < pixelRadius || t > kTMax)
			break;
		t += stepLength;
	}
	if(t > kTMax || candidateError > pixelRadius) return kInfinity;
	return t;
}

// Gets the normal the classic way
float3 getNormal(float distAtCenter, float3 p)
{
	const float2 eps = float2(1e-3,0.);
	float3 deltas = float3(map(p + eps.xyy),
	                       map(p + eps.yxy),
	                       map(p + eps.yyx)) - distAtCenter;
	return normalize(deltas);
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

// Statix-style AO
uniform int kAoSamples;
uniform float kAoFactor;
uniform float kAoDistanceFalloff; // say, 0.5
uniform float kAoDistanceScale; // say, 4
float getAo(float3 p, float3 n, float aoDelta)
{
	float sum = 0.0;
	float maxSum = 0.0;
	float factor = 1.0;
	float sampleDist = aoDelta;
	for(int i = 1; i <= kAoSamples; i++)
	{
		const float realDist = map(p + sampleDist * n);
		sum    += factor * (sampleDist - realDist);
		maxSum += factor * sampleDist;
		
		factor *= kAoDistanceFalloff;
		sampleDist *= kAoDistanceScale;
	}
	return 1. - kAoFactor * sum / maxSum;
}

uniform float3 kAmbientColor;
uniform float3 kLightDirection;
uniform float kDiffuse;
uniform float kSpecular;
uniform float kRoughness;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 thread: SV_DispatchThreadID)
{
	rngState = thread.y * iResolution.x + thread.x;
	float2 jittered = float2(thread) + float2(rng(), rng()) - .5;
	float3 pos, dir;
	getRay(jittered, pos, dir);
	dir = normalize(dir);
	
	float smallestError;
	const float t = trace(pos, dir, smallestError);
	pos = pos + dir * t;
	const float distAtCenter = t * smallestError;
	
	const float3 nrm = getNormal(distAtCenter, pos);
	const float3 lightDir = normalize(kLightDirection);
	const float wrapLighting = 0.6 + 0.4 * dot(dir, lightDir);
	float3 color = 0;
	if(t >= kTMax)
	{
		// Skybox
		color = lerp(0., kAmbientColor, wrapLighting);
	}
	else
	{
		const float NdotL = dot(nrm, lightDir);
		color = 0.5 + 0.5 * NdotL;
		float tolerance = getTolerance(t);
		const float ao = getAo(pos, nrm, 2 * tolerance);
		color *= ao;
		color *= kAmbientColor;
		
		// Direct glossy lighting + soft shadowing
		if(NdotL > 0.0){
			const float tLight = trace(pos + 4 * tolerance * nrm, lightDir, smallestError);
			const float visibility = (tLight < kTMax) ? 0 : min(1., sqrt(smallestError));
			
			const float3 h = normalize(-dir + lightDir);
			const float VdotH = dot(nrm, -dir);
			const float NdotH = dot(nrm, h);
			const float kAlpha2 = pow(kRoughness, 4);
			const float temp = NdotH * NdotH * (kAlpha2 - 1) + 1;
			float specular = kAlpha2 / (float.getPi() * temp * temp);
			specular *= 0.25 / VdotH;
			
			const float diffuse = NdotL;
			
			const float material = kSpecular * specular + kDiffuse * diffuse;
			
			color += (1-kAmbientColor) * material * visibility;
		}
	}
	
	float3 srgbish = sqrt(color);
	srgbish += (float3(rng()) - .5) / 255.;
	
	texFrame[thread] = float4(srgbish, 1.);
}
