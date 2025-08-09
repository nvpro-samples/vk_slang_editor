module interface_scene;

// For the purposes of this lab, we've put the types you need up at the start.

public struct HitInfo {
  public float3 direction; // Input direction, pointing into the surface.
  public float3 normal;    // Surface normal.
}

// Contains functions for implementing BRDFs.
// BRDFs are Bidirectional Radiance Distribution Functions;
// they tell you how light bounces off a surface.
public interface IBrdf {
  // Randomly samples and returns a new direction.
  // `rng` is an optional state for a random number generator.
  // You do not need it to solve this puzzle.
  static float3 sample(HitInfo hit, inout RNG rng);
}

//------------------------------------------------------------------------------
// The rest of this file is not needed to solve the puzzle. But it might be
// interesting to look at.

internal uint pcgHash(uint x) {
  // Condensed version of pcg_output_rxs_m_xs_32_32
  x = ((x >> ((x >> 28) + 4)) ^ x) * 277803737;
  return (x >> 22) ^ x;
}

// https://andrew-helmer.github.io/permute/,
// https://graphics.pixar.com/library/MultiJitteredSampling/paper.pdf
internal uint32_t kenslerHash(uint idx, uint mask, uint seed) {
  idx ^= seed;
  idx *= 0xe170893d;
  idx ^= seed >> 16;
  idx ^= (idx & mask) >> 4;
  idx ^= seed >> 8;
  idx *= 0x0929eb3f;
  idx ^= seed >> 23;
  idx ^= (idx & mask) >> 1;
  idx *= 1 | seed >> 27;
  idx *= 0x6935fa69;
  idx ^= (idx & mask) >> 11;
  idx *= 0x74dcb303;
  idx ^= (idx & mask) >> 2;
  idx *= 0x9e501cc3;
  idx ^= (idx & mask) >> 2;
  idx *= 0xc860a3df;
  idx &= mask;
  idx ^= idx >> 5;
  return idx;
}

internal uint32_t kenslerPermute(uint idx, uint len, uint seed) {
  uint mask = len - 1;
  mask |= mask >> 1;
  mask |= mask >> 2;
  mask |= mask >> 4;
  mask |= mask >> 8;
  mask |= mask >> 16;

  do {
    idx = kenslerHash(idx, mask, seed);
  } while (idx >= len);
  return (idx + seed) % len;
}

// A simple random number generator.
public struct RNG {
  // See https://pbr-book.org/4ed/Sampling_and_Reconstruction/Stratified_Sampler;
  // this version is much simplified and might not work as well.
  private uint m_state;
  private uint m_spp;
  private uint m_sampleIdx;

  [mutating]
  private uint getNonStratifiedUint() {
    m_state = m_state * 747796405 + 1;
    return pcgHash(m_state);
  }

  [mutating]
  public float get1D() {
    const uint stratum = kenslerPermute(m_sampleIdx, m_spp, getNonStratifiedUint());
    const float delta = float(getNonStratifiedUint()) / exp2(32.0);
    return (float(stratum) + delta) / float(m_spp);
  }

  [mutating]
  public float2 get2D() { return fract(float2(get1D(), get1D())); }

  public __init(uint2 seed, uint sampleIndex, uint samplesPerPixel) {
    m_state = seed.x * 40507 + seed.y;
    m_sampleIdx = sampleIndex;
    m_spp = samplesPerPixel;
  }
}

// Takes as input a vector of 2 2-bit integers, and returns
// a 4-bit integer equal to
// bit_reverse(bit_interleave(xy.x ^ xy.y, xy.y)).
public float2 bayer2to4(uint2 xy) {
  uint t = ((xy.y & 3) & 17) ^ (xy.x & 3); // --01--ab
  t = (t ^ (t << 3)) & 0xAAAAAAAAu;        // 1-0-b-a-
  return (t >> 5) | (t & 0xF);             // ----b1a0
}

namespace example {

// The scene for the interfaces lab.
// DiffuseBrdfT and MirrorBrdfT are the types of the BRDFs; both must satisfy
// the IBrdf interface.
// The pathtracer will use them to look up how to reflect light off materials.
//
// In practice, there are a lot of ways to write path tracers!
// You might store your materials in a Shader Binding Table, and instead use
// types like these to make sure your lights all use the same interface,
// for instance.
public struct Scene<DiffuseBrdfT, MirrorBrdfT>
    where DiffuseBrdfT : IBrdf
    where MirrorBrdfT : IBrdf
{
  [mutating] public float3 pathtrace(Ray ray, RNG rng)
  {
    float3 throughput = float3(1.0);
    for (int segment = 0; segment < 8; segment++) {
      Intersection hit = sceneIntersect(ray);
      if (hit.missed()) {
        // Sky
        const float3 sky = 0.5 + 0.5 * dot(float3(0.0, 1.0, 1.0), ray.direction);
        return throughput * sky;
      }

      // Evaluate material
      HitInfo info;
      info.direction = ray.direction;
      ray.origin = ray.at(hit.t);
      throughput *= sceneAlbedo(hit.id);
      info.normal = sceneNormal(hit.id, ray.origin);

      if (sceneUsesMirrorBrdf(hit.id)) {
        ray.direction = MirrorBrdfT::sample(info, rng);
      } else {
        ray.direction = DiffuseBrdfT::sample(info, rng);
      }
    }
    // Didn't hit the sky after a few bounces; return 0.
    return float3(0.0, 0.0, 0.0);
  }
};

//------------------------------------------------------------------------------
// Support code

static const float kInfinity = asfloat(0x7f800000);
static const uint32_t kIdNone = 0xFFFFFFFF;

struct Intersection {
  float t = kInfinity;
  uint32_t id = 0xFFFFFFFFu;

  bool missed() { return id == kIdNone; }

  [mutating]
  void update(uint32_t newId, float newT) {
    if (newT >= 0 && newT < t) {
      t = newT;
      id = newId;
    }
  }
};

public struct Ray {
  public float3 origin;
  public float3 direction;
  public float3 at(float t) { return mad(direction, float3(t), origin); }
};

// Returns the t-value of the intersection of the infinite line
// with the plane given by
// `dot(planeNormal, p) == planeDist`.
float intersectPlane(Ray ray, float3 planeNormal, float planeDist) {
  // dot(planeNormal, o + d * t) == planeDist
  // -> dot(planeNormal, o) + t * dot(planeNormal, d) == planeDist
  // -> t = (planeDist - dot(planeNormal, o)) / dot(planeNormal, d)
  return (planeDist - dot(planeNormal, ray.origin)) / dot(planeNormal, ray.direction);
}

float2 sortLoHi(float2 v) { return float2(min(v.x, v.y), max(v.x, v.y)); }

struct OBB {
  // Note that these are more parameters than we need;
  // technically, the sides of an OBB must all be perpendicular,
  // so there's only 3 (position) + 3 (side lengths) + 3 (rotation)
  // degrees of freedom.
  float3 corner;
  float3 edges[3];

  static OBB fromVertices(float3 v000, float3 v100, float3 v010, float3 v001) {
    OBB obb;
    obb.corner = v000;
    obb.edges = { v100 - v000, v010 - v000, v001 - v000 };
    return obb;
  }

  // Returns the t-value of the intersection of the OBB with
  // the ray.
  // The returned t-value may be negative; i.e. this assumes
  // the camera is outside of the box.
  // On miss, returns infinity.
  float intersect(Ray ray) {
    ray.origin -= corner;
    float2 tCloseFar;
    float tFar;
    [ForceUnroll]
    for (int i = 0; i < 3; i++) {
      const float3 edge = edges[i];
      float2 slab = float2(intersectPlane(ray, edge, 0.0), intersectPlane(ray, edge, dot(edge, edge)));
      slab = sortLoHi(slab);

      if (i == 0) {
        tCloseFar = slab;
      } else {
        tCloseFar = float2(max(tCloseFar.x, slab.x), min(tCloseFar.y, slab.y));
      }
    }

    if (tCloseFar.x >= tCloseFar.y) {
      return kInfinity;
    } else {
      return tCloseFar.x;
    }
  }

  // Returns the normal at a given position on the OBB.
  float3 getNormal(float3 pos) {
    // Not sure if this is the best algorithm; just something
    // I came up with on the spot
    pos -= corner;

    float bestDistance;
    float3 bestNormal;
    [ForceUnroll]
    for (int side = 0; side < 3; side++) {
      const float3 edge = edges[side];

      [ForceUnroll]
      for (int offset = 0; offset < 2; offset++) {
        const float3 pointOnSurface = float(offset) * edge;
        float dist = dot(edge, pos) - dot(edge, pointOnSurface);
        dist = abs(dist);
        if ((offset == 0 && side == 0) || (dist < bestDistance)) {
          bestDistance = dist;
          bestNormal = edge * (2.0 * float(offset) - 1.0);
        }
      }
    }

    return normalize(bestNormal);
  }
};

// Oriented bounding boxes for the scene.
// Normally we'd store these in a buffer;
// this is just so that everything fits in the shader.
static const Array<OBB, 12> boxes = {
  // Base mirror
  OBB::fromVertices(float3(-3.8, -3.8, -.1), float3(3.8, -3.8, -.1), float3(-3.8, 3.8, -.1), float3(-3.8, -3.8, 0)),
  OBB::fromVertices(float3(-3.14125, 2.31521, 2.3319), float3(-3.14125, 2.493, 2.50978),
                    float3(-3.14125, 1.77504, 2.87178), float3(0.064309, 2.31521, 2.3319)),
  OBB::fromVertices(float3(-0.074819, 0.012733, 0.022451), float3(-0.074819, -0.33438, 0.36938),
                    float3(-0.074819, 0.55572, 0.56572), float3(1.5926, 0.012733, 0.022451)),
  OBB::fromVertices(float3(-2.37629, -2.51016, 4.51441), float3(-2.37629, -2.00268, 4.45354),
                    float3(-2.37629, -2.4802, 4.76412), float3(2.38713, -2.51016, 4.51441)),
  OBB::fromVertices(float3(1.58642, -1.75502, 1.89817), float3(1.58642, -1.62135, 2.03191),
                    float3(1.58642, -2.798, 2.9406), float3(2.0141, -1.75502, 1.89817)),
  OBB::fromVertices(float3(2.17882, 1.04052, 1.05612), float3(1.96775, 1.21617, 1.23187),
                    float3(2.17882, -0.359464, 2.45535), float3(2.62362, 1.30762, 1.32337)),
  OBB::fromVertices(float3(2.76825, 0.527473, 3.10508), float3(2.3191, 0.863692, 3.44148),
                    float3(2.76825, -0.872507, 4.50431), float3(2.92132, 0.629658, 3.20732)),
  OBB::fromVertices(float3(-1.95729, 0.900007, 1.06288), float3(-2.66476, 1.27992, 1.443),
                    float3(-1.95729, -2.47693, 4.43801), float3(-1.89548, 0.957525, 1.12043)),
  OBB::fromVertices(float3(-2.05192, -2.42544, 2.68788), float3(-2.05192, -2.41734, 2.69598),
                    float3(-2.05192, -2.78048, 3.04272), float3(-1.54802, -2.42544, 2.68788)),
  OBB::fromVertices(float3(1.44463, -0.299888, 1.45692), float3(1.32486, 0.057503, 1.64493),
                    float3(0.653857, -0.85905, 2.01608), float3(1.59362, -0.3398, 1.6277)),
  OBB::fromVertices(float3(-0.760991, -1.83972, 4.13919), float3(-1.82613, -1.03842, 3.59822),
                    float3(-0.704892, -1.43268, 4.63164), float3(-0.501835, -1.6314, 3.93748)),
  OBB::fromVertices(float3(1.10778, 0.194421, 2.41099), float3(1.038, 0.477712, 3.12152),
                    float3(-0.802987, -1.57592, 2.92917), float3(1.44571, -0.123496, 2.57093))
};

Intersection sceneIntersect(Ray ray) {
  Intersection intersection = {};
  // Our scene is entirely composed of OBBs.
  // We unroll this so we don't store to temporary memory.
  [ForceUnroll]
  for (int i = 0; i < boxes.getCount(); i++) {
    intersection.update(i, boxes[i].intersect(ray));
  }

  return intersection;
}

float3 sceneNormal(uint32_t id, float3 pos) {
  OBB obb;
  // This is again to avoid storing to global memory;
  // we instead get a lot of `if`s.
  // In a real scene, we'd just look this up from a
  // storage buffer, of course!

  [ForceUnroll]
  for (uint32_t i = 0; i < boxes.getCount(); i++) {
    if (i == id) {
      obb = boxes[i];
    }
  }

  return obb.getNormal(pos);
}

bool sceneUsesMirrorBrdf(uint32_t id) { return (id == 0); }

float3 sceneAlbedo(uint32_t id) {
  if (id == 0) {
    return float3(0.0, 1.0, 1.0);
  } else if (id <= 8) {
    return float3(1.0, 0.0, 0.0);
  }
  return float3(1.0, 1.0, 0.0);
}
}
