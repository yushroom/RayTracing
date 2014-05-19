/*
 *	FileName:	ShowTriangleIndex.fx
 *
 *	Programmer:	Jiayin Cao
 */

//the world * view * projection matrix
float4x4	CompositeMatrix;

//the input structure for vertex shader
struct VS_INPUT
{
	float4	pos : POSITION;	// (x,y,z) position , (w) triangle Index + 1
};

//the output of vertex shader and input for pixel shader
struct VS_OUTPUT
{
	float4	pos : POSITION;			//the position for the output
	float	triIndex : TEXCOORD0;	//the triangle index
};

//the vertex shader
VS_OUTPUT ShowTriIndexVS( VS_INPUT input )
{
	//the result
	VS_OUTPUT output;

	//transform the vertex
	output.pos = mul( float4( input.pos.xyz , 1.0f ) , CompositeMatrix );

	//pass the tri index
	output.triIndex = input.pos.w;

	//return
	return output;
}

//the pixel shader just outputs the triangle index
float4	ShowTriIndexPS( VS_OUTPUT input ) : COLOR0
{
	return float4( input.triIndex , 0 , 0 , 0 );
}

//the technique
technique ShowTriIndexTec
{
	pass Default
	{
		VertexShader = compile vs_2_0 ShowTriIndexVS();
		PixelShader = compile ps_2_0 ShowTriIndexPS();
	}
}