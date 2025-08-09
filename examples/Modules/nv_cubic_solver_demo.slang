// A fast, robust cubic polynomial root finding library for Slang.
// This is the demo; the library itself is contained in nv_cubic_solver.slang.

import nv_cubic_solver;

RWTexture2D<float4> texFrame;
uniform float time;
uniform float2 screenSize;

// Oscillator with 0 derivative when it crosses 0, so it spends time around numerically challenging cases
float osc(float x)
{
	x = 0.5 * x;
	return 0.5 * sin(x) - 0.25 * sin(2 * x);
}

[shader("compute")]
[numthreads(16,16,1)]
void main(uint2 dispatchThreadID: SV_DispatchThreadID)
{
	// Create a cubic polynomial with time-varying coefficients
	const float coefficients[4] = {0.25 * osc(time), osc(0.618 * time), osc(0.382 * time), osc(0.236 * time)};

	// Screen coordinates to grid coordinates
	const float maxX = 3.5f;
	const float2 dxydPixel = 2.0 * maxX / float2(screenSize); // Incorporates grid scale
	// Flip for GL-style thread order
	const float2 glThread = float2(dispatchThreadID.x, screenSize.y - float(dispatchThreadID.y) - 1);
	const float2 xy = (glThread - float2(screenSize / 2)) * dxydPixel;

	float3 col = float3(0.0);

	// Gridlines
	float gridlines = 0.0;
	for(int level = 0; level <= 1; level++)
	{
		const float rcpIntervalUnits = pow(5.0, float(level));
		const float2 scaled = xy * rcpIntervalUnits;
		const float2 distanceToLineUnits = abs(round(scaled) - scaled) / rcpIntervalUnits;
		const float2 distanceToLinePixels = distanceToLineUnits / dxydPixel;
		const float halfLineWidthPixels = 1.0 / rcpIntervalUnits;
		const float edgeBlendPixels = 1.0;

		const float2 gridline = max(0.0,
				(min(halfLineWidthPixels, distanceToLinePixels + edgeBlendPixels * 0.5)
			- max(-halfLineWidthPixels, distanceToLinePixels - edgeBlendPixels * 0.5))
				/ edgeBlendPixels);

		gridlines = max(gridlines, max(gridline.x, gridline.y));
	}
	col = float3(0.1) * gridlines;

	// Cubic curve
	{
		const float cubicY = nv_cubic_solver.evaluateCubic(coefficients, xy.x);
		// iq's distance estimator from https://iquilezles.org/articles/distance/
		const float dCubicYdX = nv_cubic_solver.evaluateQuadratic({coefficients[1], 2.0 * coefficients[2], 3.0 * coefficients[3]}, xy.x);
		const float cubicDistance = abs(cubicY - xy.y) / sqrt(1.0 + dCubicYdX * dCubicYdX);

		const float cubicPlot = saturate((0.01 - cubicDistance)/dxydPixel.y);
		col = lerp(col, float3(0.18, 0.5, 0.0), cubicPlot);
	}

	// First onscreen root
	nv_cubic_solver.Diagnostics diagnostics;
	const float root = nv_cubic_solver.solveCubic(coefficients, -maxX, maxX, diagnostics);
	if(root != nv_cubic_solver.c_infinity)
	{
		const float2 markerDistance = abs(xy - float2(root, 0.0)) / dxydPixel;
		const float2 markerMask = saturate(float2(1.0, 50.0) - markerDistance);
		col = lerp(col, float3(0.015, 0.41, 0.61), min(markerMask.x, markerMask.y));
	}

	// Diagnostic text
	{
		const float rootY = nv_cubic_solver.evaluateCubic(coefficients, root);

		int2 cell = int2(200.0 * glThread / screenSize);
		cell.x -= 2;
		cell.y *= -1;
		const int textRow = (42 + cell.y) / 6 - 1;
		const int textCol = (4 + cell.x) / 4 - 1;
		const int glyphRow = (600 + cell.y) % 6;
		const int glyphCol = cell.x % 4;
		if(0 <= textRow && textRow < 6 && 0 <= textCol && textCol < 15 && glyphCol < 3 && glyphRow < 5)
		{
			uint c = 0;
			const uint cZero = 30;
			float valueToPrint = 0.0;
			switch(textRow)
			{
				case 0: {
					// "    X:"
					uint text[] = {0,0,0,0,24,27};
					if(textCol < 6) c = text[textCol];
					valueToPrint = root;
					break;
				}
				case 1: {
					// "    Y:"
					uint text[] = {0,0,0,0,25,27};
					if(textCol < 6) c = text[textCol];
					valueToPrint = nv_cubic_solver.evaluateCubic(coefficients, root);
					break;
				}
				case 2: {
					// "   X0:"
					uint text[] = {0,0,0,24,cZero,27};
					if(textCol < 6) c = text[textCol];
					valueToPrint = diagnostics.x0;
					break;
				}
				case 3: {
					// "   X1:"
					uint text[] = {0,0,0,24,cZero+1,27};
					if(textCol < 6) c = text[textCol];
					valueToPrint = diagnostics.x1;
					break;
				}
				case 4: {
					// " CASE:"
					uint text[] = {0,3,1,19,5,27};
					if(textCol < 6) c = text[textCol];
					valueToPrint = float(diagnostics.caseChosen / 10.0);
					break;
				}
				case 5: {
					// " ITER:"
					uint text[] = {0,9,20,5,18,27};
					if(textCol < 6) c = text[textCol];
					valueToPrint = float(diagnostics.iterations);
					break;
				}
			}

			// Format float
			if(textCol == 6)
			{
				if(valueToPrint < 0.0)
				{
					c = 29;
				}
			}
			valueToPrint = abs(valueToPrint);
			if(textCol == 7)
			{
				c = cZero + uint(floor(valueToPrint));
			}
			else if(textCol == 8)
			{
				c = 28;
			}
			else if(textCol >= 9)
			{
				uint last6Digits = uint(floor(valueToPrint * 1000000)); 
				for(uint i = 0; i < 14 - textCol; i++)
				{
					last6Digits /= 10;
				} 
				uint digit = last6Digits % 10u;
				c = cZero + digit;
			}

			//              ,     A,B,     C,D,     E,F,G,H,     I,J,K,     L,M,N,O,P,Q,     R,     S,     T,     U,     V,W,     X,     Y,Z,     :,     .,    -,     0,     1,     2,     3,     4,     5,     6,     7,     8,     9
			const uint font[] = {0,0x5BEF,0,0x724F,0,0x73CF,0,0,0,0x7497,0,0,0x7249,0,0,0,0,0,0x5AEB,0x388E,0x2497,0x7B6D,0x256D,0,0x5AAD,0x24AD,0,0x0410,0x2000,0x1C0,0x7B6F,0x249A,0x73E7,0x38E3,0x49ED,0x38CF,0x7BCF,0x4927,0x7BEF,0x79EF};
			if(c > cZero + 10) c = 0;
			const uint glyph = font[c];
			const uint on = (glyph >> (3 * glyphRow + glyphCol)) & 1;
			if(on != 0)
			{
				col = float3(1.0);
			}
		}
	}
	texFrame[dispatchThreadID] = float4(pow(col, 0.45454545), 1.0);
}
