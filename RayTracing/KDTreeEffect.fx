/*
 *	FileName:	KDTreeEffect.fx
 *
 *	Programmer:	Jiayin Cao
 */

//the composite matrix
float4x4	CompositeMatrix;

//////////////////////////////////////////////////////////////////////////////////////////////////////
// render with normal and texture

//input for the vertex shader
struct VS_INPUT
{
	//the position of the vertex
	float3	position : POSITION;
};

struct VS_OUTPUT
{
	//position in NDC
	float4	position : POSITION;
};

//vertex shader for vertexes with position , normal , texture coordinate
VS_OUTPUT  DefaultVS( VS_INPUT vertex )
{
	//the output
	VS_OUTPUT outVertex;

	//transform the vertex
	outVertex.position = mul( float4( vertex.position.xyz , 1.0f ) , CompositeMatrix );

	//return the vertex
	return outVertex;
}

//pixel shader
float4	DefaultPS( VS_OUTPUT pixel ) : COLOR
{
	//the default ambient color
	return float4( 1.0f , 1.0f , 1.0f , 1.0f );
}

//the technique
technique Default_Tec
{
	pass DefaultPass
	{
		VertexShader =  compile vs_2_0 DefaultVS();
		PixelShader = compile ps_2_0 DefaultPS();
	}
}