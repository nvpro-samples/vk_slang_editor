// Slang can compile mostly unmodified GLSL shaders. This black hole shader is almost entirely GLSL, except for 5 new lines.
// The only additional lines needed to compile with Slang are the entrypoint annotations,
// tagged with "// #GLSL->SLANG".

// Originally by Nia Bickford:
// https://www.shadertoy.com/view/4s3SzN
// Adapted to a single GLSL file with compute shaders,
// as if compiling for SPIR-V.
// Gaussian blur parameters have been modified as well.
// Original description follows:

// This is a somewhat ad-hoc gravitational raytracer made as part of my final project
// for UCLA's Math 32BH.
// This slightly newer version now has some rough Gaussian lens flare added to work
// with some of the newer features in Shadertoy.

// Slang sees the #version 460 directive here and uses this as its
// cue to load in its GLSL compatibility module, glsl.meta.slang.
// You can see the contents of this module online at
// https://github.com/shader-slang/slang/blob/master/source/slang/glsl.meta.slang
#version 460
#extension GL_EXT_shader_image_load_formatted : require 

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Uniforms
layout(binding=1) uniform image2D texFrame;
layout(binding=2) uniform image2D texPing;
layout(binding=3) uniform image2D texPong;
layout(binding=4) uniform sampler2D texStarmap;
uniform vec2 iResolution;
uniform float iTime;

#define PI 3.1415926535
#define MATTE_Y 0.39

[shader("compute")] // #GLSL->SLANG: Add entrypoint attribute
void mainImage()
{
	uvec2 thread = gl_GlobalInvocationID.xy;
	if(any(greaterThan(thread, iResolution))) return;
	vec2 fragCoord = vec2(thread);
	fragCoord.y = iResolution.y - fragCoord.y;
	
	vec2 uv = 2.0*(fragCoord.xy-0.5*iResolution.xy) / iResolution.x;
    
    // Matte to widescreen
    // We'll matte again in the final pass;
    // this just saves some processing time.
    vec2 uv2 = 2.0*(fragCoord-0.5*iResolution.xy)/iResolution.x;
    if(abs(uv2.y) > MATTE_Y)
    {
    	imageStore(texFrame, ivec2(thread), vec4(0.0));
    	return;
    }
    
    uv*=6.0; // Ad-hoc view factor
    
    // Camera controls
    // Get spherical coordinates from the camera ray (uv.x,1,uv.y)
    float cd=4.0+6.0*(1.0-pow(1.1,-(2.0*iTime+25.0) ));
    // Initial direction
    vec3 nF=vec3(0.4,uv.x,uv.y);
    
    // Astronomical aberration
    float beta=0.0;
    float betap=sqrt(1.-beta*beta);
    nF=vec3(-nF.x*betap,-nF.y+beta,-nF.z*betap)/(1.-beta*nF.y);
    nF*=3.0;
    nF.y/=1.01; // Ad-hoc: Make black hole more circular
    
    // Initialize the state!
    float a = 0.9;
    
    // For the moment, the camera's coordinates will be (cd,pi/2,0), it'll be at
    // rest with respect to the FIDO (i.e. we're not accounting for special-relativistic
    // phenomena), and the ray's canonical momenta will be some multiple of nF.
    float lr=cd;
    float lo=PI*(0.49-0.001*pow(1.1,-iTime));
    float lp=0.0+0.04*iTime;
    float pr=nF.x;
    float po=nF.z; //we use 'o' because it looks like theta.
    float b =(nF.y+3.);
    
    // Set timestep and local variables to default values
    float dt=0.1;
    
    float drdz=0.0;
    float dodz=0.0;
    float dpdz=0.0;
    float dprdz=0.0;
    float dpodz=0.0;
    
    float q=po*po+cos(lo)*cos(lo)*(b*b/(sin(lo)*sin(lo))-a*a); 
    
    // Object we hit and the accumulated color
    int exitCode=0;
    vec4 accCol=vec4(0.0,0.0,0.0,0.0);
    // Main rendering loop
    for(int i=0;i<300;i++){

        // Compute Kerr black hole parameters
        // (se the paper for more details as to where this came from)
        float co=cos(lo);
        float so=sin(lo);
        
        float p2=lr*lr+a*a*co*co;
        float D=lr*lr-2.0*lr+a*a;
        
        // Early exit: are we in the black hole?
        if(D*p2<0.0){
            exitCode=1;
            break;
        }
        
        drdz=pr*D/p2;
        dodz=po/p2;
        
        float P=lr*lr+a*a-a*b;
        float R=P*P-D*((b-a)*(b-a)+q);
        
        dpdz=(a*(P-D)+b*D/(so*so))/(D*p2);
        dprdz=(lr*D*(-R+pr*pr*D*D)+p2*((1.0-lr)*R+D*(2.0*lr*P+(1.0-lr)*((b-a)*(b-a)+q+pr*pr*D))))/(D*D*p2*p2);
        dpodz=(b*b*D*p2*co/(so*so*so)+a*a*so*co*(R-D*(pr*pr*D+p2)))/(D*p2*p2);
        
        
        // Change our timestep depending on our curvature and position
        dt=min(min(0.5/abs(dprdz),0.1/abs(dpodz)),0.01+4.0*abs(mod(lo,PI)-0.5*PI) );
        
        // Also exit early if our curvature's too high (this just hides some artifacts)
        if(min(0.5/abs(dprdz),0.5*.5/abs(dpodz))<0.001){
            exitCode=1;
            break;
        }
        
        dt=clamp(dt,0.005,0.5);
        
        // Have we escaped the region of influence of the black hole?
        if(lr+dt*drdz>16.0){
            //call it
            dt=(16.0-lr)/drdz;
            lr+=dt*drdz;
        	lo+=dt*dodz;
        	lp+=dt*dpdz;
            break;
        }
        
        
        //if we might have just passed through the ring
        lo=mod(lo,PI);
        //lo+dt*dodz=Pi/2
        float tdt=(PI*0.5-lo)/dodz;
        if((-1.*dt<=tdt && tdt<=1.*dt) || (0.<=(PI*0.51-lo)/dodz && (PI*0.49-lo)/dodz<=dt)){
            float tr=lr+tdt*drdz; // Ring radius
            if(tr>3. && tr<8.){
                // Ring brightness - This is largely ad-hoc:
				float lines=(smoothstep(3.,3.5,tr)*(1.-smoothstep(7.4,8.4,tr)));
                lines *= 0.4;
                // The original shader used a texture here, but
                // the effect doesn't really rely on it:
                // lines*=texture(iChannel1,vec2(mod(tr+0.00*drdz,2.0)/2.0,mod(lp,PI)/PI ),20.0*abs(dodz)).r;
                
                float tp=lp+tdt*dpdz;
                
                lines*=(1.-smoothstep(0.0,0.001,mod(tp,0.2))*(1.-smoothstep(0.009,0.01,mod(lp,0.2))));
                lines*=16.0*pow(abs(dodz),1.0)*exp(abs(dodz))*exp(abs(dpdz))*exp(dt); // ad-hoc attenuation factor
                lines*=0.68;
                lines=pow(lines,0.45);
                
				vec4 tex=vec4(lines*0.2+0.1*dpdz,lines*0.08,lines*0.04-0.1*dpdz,3.0*lines*exp(dt));
                accCol.rgb=(accCol.rgb+tex.rgb*tex.a*(1.-accCol.a));
                accCol.a=accCol.a*(1.-tex.a);
                
            }
        }
        
        // Update position
        lr+=dt*drdz;
        lo+=dt*dodz;
        lp+=dt*dpdz;
        
        pr+=dt*dprdz;
        po+=dt*dpodz;
    }
    
    vec3 col;
    if(exitCode==1){
        col=vec3(0.,0.,0.);
    }else{
        // turn phi and theta into equiangular lookups
        float eqPhi=mod(lp,2.*PI);
        eqPhi=(eqPhi/(2.*PI));
        float eqTheta=1.-mod(lo,PI);
        eqTheta/=PI;
        
        
        // To avoid issues when sampling the equirectangular map,
        // we call textureGrad but correct for wraparound:
        const vec2 p = vec2(eqPhi,eqTheta);
        vec2 dpdx = dFdx(p);
        dpdx.x -= round(dpdx.x);
        vec2 dpdy = dFdy(p);
        dpdy.x -= round(dpdy.x);
        // The sqrt here is an ad-hoc linear->sRGB conversion.
        col = 2.0 * sqrt(textureGrad(texStarmap, p, dpdx, dpdy).rgb);
        
        //hey, color grading!
        col.r*=0.8;
        col.g*=0.8;
        col*=(0.25*col.r+0.5*col.g+0.25*col.b); 
    }
   
    //blend in accCol
    col=accCol.rgb+col*(1.-accCol.a);
        
    // More color grading
    col.r=0.5*col.r+0.5*smoothstep(0.,1.,col.r);
    col.b=0.7*col.b+0.3*(col.b-smoothstep(0.,1.,col.b));
    col.b+=0.1*col.r;
    col.g*=0.8;
    
    col.b*=1.2;
    col.g*=1.3;
    
    col = clamp(col,0.0,10.0);
    imageStore(texFrame, ivec2(thread), vec4(col,1.0));
}

// Buffer B
vec3 GaussianBlur(image2D tex, ivec2 texel, ivec2 dx, int width, float sigma){
    float totalWeight = 0.0;
    vec3 totalCol = vec3(0.0);
    for(int i = -width; i <= width; i++)
    {
    	float weight = exp(-float(i * i) / (2.0 * sigma * sigma))
    	             / (sigma * sqrt(2.0 * PI));
    	vec3 col = imageLoad(tex, texel + i * dx).rgb;
    	totalWeight += weight;
    	totalCol += col * weight;
    }

	return totalCol / totalWeight;
}

[shader("compute")] // #GLSL->SLANG: Add entrypoint attribute
void blurY()
{
	if(any(greaterThan(gl_GlobalInvocationID.xy, iResolution))) return;
	vec3 gb = GaussianBlur(texFrame, ivec2(gl_GlobalInvocationID.xy), ivec2(0, 1), 9, 4.0);
	imageStore(texPing, ivec2(gl_GlobalInvocationID.xy), vec4(gb, 1.0));
}

[shader("compute")] // #GLSL->SLANG: Add entrypoint attribute
void blurX()
{
	if(any(greaterThan(gl_GlobalInvocationID.xy, iResolution))) return;
	vec3 gb = 0.25 * GaussianBlur(texPing, ivec2(gl_GlobalInvocationID.xy), ivec2(1, 0), 9, 4.0);
	gb += 0.75 * GaussianBlur(texPing, ivec2(gl_GlobalInvocationID.xy), ivec2(8, 0), 16, 8.0);
	imageStore(texPong, ivec2(gl_GlobalInvocationID.xy), vec4(gb, 1.0));
}

[shader("compute")] // #GLSL->SLANG: Add entrypoint attribute
void blurY2()
{
	if(any(greaterThan(gl_GlobalInvocationID.xy, iResolution))) return;
	vec3 gb = GaussianBlur(texPong, ivec2(gl_GlobalInvocationID.xy), ivec2(0, 8), 9, 4.0);
	imageStore(texPing, ivec2(gl_GlobalInvocationID.xy), vec4(gb, 1.0));
}

vec3 manualLinearSample(image2D image, vec2 resolution, vec2 uv)
{
	vec2 pixels = resolution * uv;
	vec2 fRoot = floor(pixels);
	vec2 delta = pixels - fRoot;
	ivec2 iRoot = ivec2(fRoot);
	vec3 v00 = imageLoad(image, iRoot).rgb;
	vec3 v01 = imageLoad(image, iRoot + ivec2(0,1)).rgb;
	vec3 v10 = imageLoad(image, iRoot + ivec2(1,0)).rgb;
	vec3 v11 = imageLoad(image, iRoot + ivec2(1,1)).rgb;
	return mix(mix(v00, v01, delta.y), mix(v10, v11, delta.y), delta.x);
}

[shader("compute")] // #GLSL->SLANG: Add entrypoint attribute
void composite()
{
	uvec2 thread = gl_GlobalInvocationID.xy;
	if(any(greaterThan(thread, iResolution))) return;
	
	// Matte to widescreen
	vec2 uv = vec2(thread) / iResolution;
	vec2 uv2 = 2.0 * (thread - 0.5 * iResolution.xy) / iResolution.x;
	if(abs(uv2.y) > MATTE_Y)
	{
		imageStore(texFrame, ivec2(thread), vec4(0.0, 0.0, 0.0, 1.0));
		return;
	}
	
	vec3 source = imageLoad(texFrame, ivec2(thread)).rgb; 
	vec3 gb = imageLoad(texPing, ivec2(thread)).rgb;
	// Add in a reverse image!
	vec3 gbFlipped = manualLinearSample(texPing, iResolution, vec2(0.5)+0.55*(vec2(0.5)-uv));
	vec3 col = source + 0.2 * gb + 0.1 * gbFlipped;
	// Final tonemapping
	col = tanh(col);
	imageStore(texFrame, ivec2(thread), vec4(col, 1.0));
}
