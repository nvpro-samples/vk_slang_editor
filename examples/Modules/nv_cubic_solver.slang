/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

// A fast, robust cubic polynomial root finding library for Slang.
// Version 2025-08-07

module nv_cubic_solver;

// Public API
namespace nv_cubic_solver
{

// Evaluates sum(k[i] x^i, i in [0,2]).
public float evaluateQuadratic(const float k[3], float x);

// Evaluates sum(k[i] x^i, i in [0,3]).
public float evaluateCubic(const float k[4], float x);

// Finds the first root of the cubic equation
// f(x) := sum(k[i] x^i, i in [0,3]) == 0
// within the range [x0, x1], if it exists. If it doesn't, returns
// infinity (or NaN, if something goes wrong).
// "k" stands for "koefficient" here, because we need a variable named c
// to match the equations in Yuksel's paper.
public float solveCubic(const float k[4], float x0, float x1)
{
	Diagnostics unused;
	return solveCubic(k, x0, x1, unused);
}

public struct Diagnostics{
	public int caseChosen;
	public float x0;
	public float x1;
	public int iterations;
};

// Same as solveCubic, but provides debugging info.
public float solveCubic(const float k[4], float x0, float x1, out Diagnostics diagnostics);

public static const float c_infinity = asfloat(0x7f800000u);

//------------------------------------------------------------------------------
// Internal defines

#define NV_CUBIC_SOLVER_ANALYTIC 0
#define NV_CUBIC_SOLVER_YUKSEL 1

// This define makes the cubic solver use Cem Yuksel's cubic solver from http://www.cemyuksel.com/research/polynomials/ ,
// instead of the analytical method from Numerical Recipes.
// Set NV_CUBIC_SOLVER to 0 (analytic) or 1 (Yuksel) to choose your cubic solver.
// 0 is the analytical method from Numerical Recipes, while 1 is Cem Yuksel's cubic solver reimplemented from the
// paper http://www.cemyuksel.com/research/polynomials/ , with additional tweaks. Generally, Yuksel is a better choice.
#ifndef NV_CUBIC_SOLVER
	#define NV_CUBIC_SOLVER NV_CUBIC_SOLVER_YUKSEL
#endif

// The Yuksel cubic solver will try to find a root within
// (x1 - x0) * NV_CUBIC_SOLVER_YUKSEL_TOLERANCE of the true root.
#define NV_CUBIC_SOLVER_YUKSEL_TOLERANCE 0.001f
// It turns out a low iteration count is perfectly OK here!
// Voxel rendering can get away with as low as 4 iterations.
#define NV_CUBIC_SOLVER_YUKSEL_MAX_ITERATIONS 8

static const float c_twoPi = 6.2831853f;

public float evaluateCubic(const float k[4], float x)
{
	float y = fma(k[3], x, k[2]);
	y = fma(y, x, k[1]);
	y = fma(y, x, k[0]);
	return y;
}

public float evaluateQuadratic(const float k[3], float x)
{
	float y = fma(k[2], x, k[1]);
	y = fma(y, x, k[0]);
	return y;
}

// Returns true iff a and b have different signs, or if either is 0.
bool isDifferentSignOrZero(const float a, const float b)
{
	// This (mul, cmp) might be faster than Yuksel's
	// (a < 0) != (b < 0) (cmp, cmp, cmp)?
	return a * b <= 0;
}

public float solveCubic(const float k[4], float x0, float x1, out Diagnostics diagnostics)
{
	diagnostics.caseChosen = 0;
	diagnostics.x0 = x0;
	diagnostics.x1 = x1;
	diagnostics.iterations = 0;

	// Fast path: If k[0] is near 0 and x0 is 0, then 0 may as well be
	// a root.
	// This avoids numerical instability here later on.
	if ((x0 == 0.f) && (abs(k[0]) < 1e-7f))
	{
		diagnostics.caseChosen = 1;
		return 0.0f;
	}

#if NV_CUBIC_SOLVER == NV_CUBIC_SOLVER_YUKSEL
	// This approach is based on CubicFirstRoot in Cem Yuksel's
	// implementation of his fast and robust cubic solving paper,
	// with some modifications to make it easier for the compiler to
	// generate convergent code.

	// Controls the precision of the solve.
	// A factor of 0.01 is typically good enough for voxel rendering.
	const float epsError = (x1 - x0) * 0.001f;

	// Solve the quadratic equation f'(x) == 0 to find the points where
	// f'(x) changes sign.
	// Over each of these intervals, f(x) will be either monotonically
	// increasing, decreasing, or constant.
	// Our derivative is 3 k[3] x^2 + 2 k[2] x + k[1]:
	const float a = k[3] * 3;
	const float bDiv2 = k[2];
	const float c = k[1];

	// You might be familiar with the "b^2 - 4ac" term in the quadratic
	// formula; this is (b^2 - 4ac)/4.
	const float discriminantDiv4 = bDiv2 * bDiv2 - a * c;

	if (discriminantDiv4 == 0.f) // Uncommon
	{
		// The cubic is either linear or a constant!
		if (c == 0.0f)
		{
			// In this branch, it's a constant! It has solutions everywhere
			// if it's 0, and none otherwise.
			if (k[0] == 0.0f)
			{
				diagnostics.caseChosen = 2;
				return x0;
			}
			else
			{
				diagnostics.caseChosen = 3;
				return c_infinity;
			}
		}
		else
		{
			// Linear system, c[1] x + c[0] == 0
			const float x = -k[0] / k[1];
			if (x0 <= x && x <= x1)
			{
				diagnostics.caseChosen = 4;
				return x;
			}
			else
			{
				diagnostics.caseChosen = 5;
				return c_infinity;
			}
		}
	}

	// We'll eventually do an iterative solve over an interval.
	// We'll modify x0, x1, y0, and y1 to be the endpoints of this smaller
	// interval.
	float y0 = evaluateCubic(k, x0);
	float y1 = evaluateCubic(k, x1);
	// Divide into cases:
	if (discriminantDiv4 < 0.f)
	{
		diagnostics.caseChosen = 6;
		// f'(x) == 0 has no real roots, so the cubic as a whole is
		// monotonic, and our interval is [x0, x1].
		if (!isDifferentSignOrZero(y0, y1))
		{
			diagnostics.caseChosen = 7;
			return c_infinity;
		}
	}
	else
	{
		// f'(x) == 0 has two real roots; call them xa and xb, with xa <= xb.
		// Compute them:
		const float q = -(bDiv2 + ((bDiv2 > 0.f) ? 1.f : -1.f) * sqrt(discriminantDiv4));
		const float rv0 = q / a; // If a == 0, then this is NaN; we'll remove it in the next step.
		const float rv1 = c / q; // We're guaranteed q > 0, so this can't be NaN.
		// Sort the two roots. In the NaN case, we get rv1 for both:
		const float xa = (rv0 < rv1 ? rv0 : rv1);
		const float xb = (rv1 < rv0 ? rv0 : rv1);

		// Now, find the first interval between the points in
		// sort(x0, xa, xb, x1)
		// that has a root.
		// There are 6 possible orders:
		// 01ab -> test [x0, x1]
		// a01b -> test [x0, x1]
		// ab01 -> test [x0, x1]
		// 0a1b -> test [x0, xa] and [xa, x1]
		// 0ab1 -> test [x0, xa], [xa, xb], and [xb, x1]
		// a0b1 -> test [x0, xb] and [xb, x1]

		if (xa >= x1 || xb <= x0 || (xa <= x0 && xb >= x1))
		{
			diagnostics.caseChosen = 8;
			// Test [x0, x1]
			if (!isDifferentSignOrZero(y0, y1))
			{
				diagnostics.caseChosen = 9;
				return c_infinity; // The cubic doesn't change sign in this interval, so there's no root.
			}
		}
		else if (xa > x0)
		{
			// 0a1b or 0ab1
			// Test [x0, xa]:
			const float ya = evaluateCubic(k, xa);
			if (isDifferentSignOrZero(y0, ya))
			{
				diagnostics.caseChosen = 10;
				// It's [x0, xa]!
				x1 = xa;
				y1 = ya;
			}
			else if (xb < x1)
			{
				// 0ab1 -> test [xa, xb] and [xb, x1]:
				const float yb = evaluateCubic(k, xb);
				if (isDifferentSignOrZero(ya, yb))
				{
					diagnostics.caseChosen = 11;
					x0 = xa;
					y0 = ya;
					x1 = xb;
					y1 = yb;
				}
				else if (isDifferentSignOrZero(yb, y1))
				{
					diagnostics.caseChosen = 12;
					x0 = xb;
					y0 = yb;
				}
				else
				{
					diagnostics.caseChosen = 13;
					return c_infinity;
				}
			}
			else
			{
				// 0a1b -> test [xa, x1]:
				if (isDifferentSignOrZero(ya, y1))
				{
					diagnostics.caseChosen = 14;
					x0 = xa;
					y0 = ya;
				}
				else
				{
					diagnostics.caseChosen = 15;
					return c_infinity;
				}
			}
		}
		else
		{
			// a0b1 -> test [x0, xb] and [xb, x1]:
			const float yb = evaluateCubic(k, xb);
			if (isDifferentSignOrZero(y0, yb))
			{
				diagnostics.caseChosen = 16;
				x1 = xb;
				y1 = yb;
			}
			else if (isDifferentSignOrZero(yb, y1))
			{
				diagnostics.caseChosen = 17;
				x0 = xb;
				y0 = yb;
			}
			else
			{
				diagnostics.caseChosen = 18;
				return c_infinity;
			}
		}
	}

	// We now know there's a root in the range [x0, x1]. Find it!
	const float deriv[3] = { c, 2 * bDiv2, a };

	// Yuksel's code does 16 safety iterations of Newton before moving
	// to the more robust method. For now, let's just start with the more
	// robust Newton-or-bisect method.
	// Start with the midpoint, because we might be able to exit early:
	float xr = (x0 + x1) * 0.5f;
	const float epsError2 = epsError * 2.f;
	if (x1 - x0 <= epsError2)
	{
		return xr;
	}

	float yr = evaluateCubic(k, xr);
	int safetyCounter;
	for (safetyCounter = 0; safetyCounter < NV_CUBIC_SOLVER_YUKSEL_MAX_ITERATIONS; safetyCounter++)
	{
		// Split to one of the two sides
		if (isDifferentSignOrZero(y0, yr))
		{
			// Change to [x0, xr]
			x1 = xr;
		}
		else
		{
			// Change to [xr, x1]
			x0 = xr;
		}
		// Try a Newton step, x_new = x - f(x)/f'(x)
		const float xNew = xr - yr / evaluateQuadratic(deriv, xr);
		if (xNew >= x0 && xNew <= x1)
		{
			// The Newton step was valid!
			const float stepSize = abs(xr - xNew);
			xr = xNew;
			if (stepSize <= epsError)
			{
				break; // Accept xr
			}
		}
		else
		{
			// Newton step failed; do bisection instead.
			xr = (x0 + x1) * 0.5f;
			if (x1 - x0 <= epsError2)
			{
				break;
			}
		}
		// Set up for the next iteration
		yr = evaluateCubic(k, xr);
	}

	diagnostics.x0 = x0;
	diagnostics.x1 = x1;
	diagnostics.iterations = safetyCounter;

	return xr;

#else
	// Use analytical cubic solver instead
	// For quadratic and cubic equations, we use the algorithms from
	// Numerical Recipes, 3rd edition, section 5.6,
	// "Quadratic and Cubic Equations".
	//
	// Note: Tuning the tolerances in this solver is very important --
	// the effects are visible!
	//
	// The first thing we do is divide by the highest coefficient -- so we
	// need logic to reduce to solving lower-degree polynomials if it's 0.
	// This uses a threshold to determine when coefficients are 0 at the
	// moment, but maybe comparing the highest coefficient's magnitude to
	// the maximum of the other coefficients' magnitudes would be better.
	const constexpr float coefficientEpsilon = 1e-5f;
	if (abs(k[3]) <= coefficientEpsilon)
	{
		if (abs(k[2]) <= coefficientEpsilon)
		{
			if (abs(k[1]) <= coefficientEpsilon)
			{
				// k[0] is the only non-zero coefficient here.
				// Now, this should never happen for
				// trilinearIntersection(), but let's handle it anyway.
				if (k[0] == 0.0f)
				{
					return x0;
				}
				else
				{
					return c_infinity;
				}
			}
			else
			{
				// Linear system, c[1] x + c[0] == 0
				const float x = -k[0] / k[1];
				if (x0 <= x && x <= x1)
				{
					return x;
				}
				else
				{
					return c_infinity;
				}
			}
		}
		else
		{
			// Quadratic equation

			// Equation (5.6.4)
			const float determinant = k[1] * k[1] - 4.f * k[2] * k[0];
			if (determinant < 0.f)
				return c_infinity;
			// We don't use sign() here because that would give the wrong
			// result for b==0.
			const float sgnB = (k[1] >= 0.f ? 1.f : -1.f);
			const float q = -.5f * (k[1] + sgnB * sqrt(determinant));
			// Equation (5.6.5)
			float bestX = c_infinity;
			float x = q / k[2];
			if (x0 <= x && x <= x1)
				bestX = x;
			// If q == 0, then c[1] == 0 and determinant == 0, so c[0] == 0
			// and we have a double root at 0.
			if (q != 0)
			{
				x = k[0] / q;
				if (x0 <= x && x <= x1 && x < bestX)
				{
					bestX = x;
				}
			}
			return bestX;
		}
	}
	else
	{
		// Cubic equation
		// (5.6.9)
		const float a = k[2] / k[3];
		const float b = k[1] / k[3];
		const float c = k[0] / k[3];
		// (5.6.10)
		const float a2 = a * a;
		const float q = (a2 - 3.f * b) / 9.f;
		const float r = (2.f * a2 * a - 9.f * a * b + 27.f * c) / 54.f;
		const float r2 = r * r;
		const float q3 = q * q * q;
		if (r2 < q3)
		{
			// 3 real roots, (5.6.11)
			// We must have q > 0, because r2 >= 0 and q3 > r2.
			const float sqrtQ = sqrt(q);
			const float sqrtQ3 = sqrtQ * sqrtQ * sqrtQ;
			const float theta = acos(clamp(r / sqrtQ3, -.999999f, .999999f));
			// (5.6.12), in slightly different form
			const float thetaDiv3 = theta / 3.f;
			const float mult = -2.f * sqrtQ;
			const float addend = -a / 3.f;
			float bestX = c_infinity;
			for (int i = 0; i < 3; i++)
			{
				const float x = mult * cos(thetaDiv3 + float(i) * c_twoPi / 3.f) + addend;
				if (x0 <= x && x <= x1 && x < bestX)
				{
					bestX = x;
				}
			}
			return bestX;
		}
		else
		{
			// 1 real root, equation (5.6.15)
			const float sgnR = (r >= 0.f) ? 1.f : -1.f;
			const float bigA = -sgnR * pow(abs(r) + sqrt(r2 - q3), 1.f / 3.f);
			// Equation (5.6.16)
			float bigB = 0.f;
			if (bigA != 0.f)
			{
				bigB = q / bigA;
			}
			const float x = (bigA + bigB) - a / 3.f;
			if (x0 <= x && x <= x1)
			{
				return x;
			}
			else
			{
				return c_infinity;
			}
		}
	}
#endif
}

} // namespace nv_cubic_solver
