/*
 *	Programmer:	Jiayin Cao
 *
 *	FileName:	MeshEffect.fx
 */

//the composite matrix
float4x4	CompositeMatrix;
//the world matrix
float4x4	WorldMatrix;
//light position
float4x4	LightPosition;
//light diffuse color
float4		LightDiffuse;
//Ambient Color
float4		AmbientColor;
//Model diffuse color
float4		DiffuseColor;
//specular color
float4		SpecularColor;
//specular power
int			SpecularPower;
//eye position
float3		EyePosition;

// the texture
texture DiffuseTex;
sampler	DiffuseSampler = sampler_state
{
	Texture = DiffuseTex;
	MINFILTER = LINEAR;
	MAGFILTER = LINEAR;
	MIPFILTER = LINEAR;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////
// render with normal and texture

//input for the vertex shader
struct VS_INPUT_PNT
{
	//the position of the vertex
	float3	position : POSITION;
	//the normal of the vertex
	float3	normal : NORMAL;
	//the texture coordinate
	float2	texCoord : TEXCOORD0;
};

struct VS_OUTPUT_PNT
{
	//position in NDC
	float4	position : POSITION;
	//the normal of the vertex in world coordinate
	float3	normal : TEXCOORD0;
	//view direction
	float3	intersected : TEXCOORD1;
	//texture coordinate
	float2	texCoord : TEXCOORD2;
};

//vertex shader for vertexes with position , normal , texture coordinate
VS_OUTPUT_PNT  VertexShaderPNT( VS_INPUT_PNT vertex )
{
	//the output
	VS_OUTPUT_PNT outVertex;

	//transform the vertex
	outVertex.position = mul( float4( vertex.position.xyz , 1.0f ) , CompositeMatrix );

	//transform normal
	outVertex.normal = normalize( mul( vertex.normal , WorldMatrix ) );

	//output the view direction
	outVertex.intersected = mul( float4(vertex.position.xyz,1.0f) , WorldMatrix );

	//pass texture coordinate
	outVertex.texCoord = vertex.texCoord;

	return outVertex;
}

//pixel shader
float4	PixelShaderPNT( VS_OUTPUT_PNT pixel ) : COLOR
{
	//the texture color
	float4 texColor = tex2D( DiffuseSampler , pixel.texCoord );

	float3 eyeDir = normalize(pixel.intersected - EyePosition);

	//the result
	float4 color = AmbientColor;

	for( int i = 0 ; i < 2 ; i++ )
	{
		if( LightPosition[i].w == 0 )
			break;

		//the light direction
		float3 lightDir = normalize( LightPosition[i] - pixel.intersected );

		//the directional light
		float d0 = saturate( dot( pixel.normal , lightDir ) * LightPosition[i].w );

		color += d0 * ( texColor * DiffuseColor );

		//add specular if possible
		if( SpecularPower > 0 )
		{
			float3 reflectDir = reflect( lightDir , pixel.normal );
			float speDen = saturate( dot( reflectDir , eyeDir ) );
			float4 specular = pow( speDen , SpecularPower ) * float4( SpecularColor.xyz , 0.0f );

			color += specular;
		}
	}

	//return pixel.position;
	return color;
}

//the technique
technique PNT_Tec
{
	pass DefaultPass
	{
		VertexShader =  compile vs_2_0 VertexShaderPNT();
		PixelShader = compile ps_2_0 PixelShaderPNT();
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
// render with normal and diffuse color

//input for the vertex shader
struct VS_INPUT_PN
{
	//the position of the vertex
	float3	position : POSITION;
	//the normal of the vertex
	float3	normal : NORMAL;
};

struct VS_OUTPUT_PN
{
	//position in NDC
	float4	position : POSITION;
	//the normal of the vertex in world coordinate
	float3	normal : TEXCOORD0;
	//view direction
	float4	intersected : TEXCOORD1;
};

//vertex shader for vertexes with position , normal , texture coordinate
VS_OUTPUT_PN  VertexShaderPN( VS_INPUT_PN vertex )
{
	//the output
	VS_OUTPUT_PN outVertex;

	//transform the vertex
	outVertex.position = mul( float4( vertex.position.xyz , 1.0f ) , CompositeMatrix );

	//transform normal
	outVertex.normal = normalize( mul( vertex.normal , WorldMatrix ) );

	//output the view direction
	outVertex.intersected = mul( float4(vertex.position.xyz,1.0f) , WorldMatrix );

	return outVertex;
}

//pixel shader
float4	PixelShaderPN( VS_OUTPUT_PN pixel ) : COLOR
{
	//the eye direction
	float3 eyeDir = normalize(pixel.intersected - EyePosition);

	//the result
	float4 color = AmbientColor;

	for( int i = 0 ; i < 2 ; i++ )
	{
		if( LightPosition[i].w == 0 )
			break;

		//the light direction
		float3 lightDir = normalize( LightPosition[i] - pixel.intersected );

		//the directional light
		float d0 = saturate( dot( pixel.normal , lightDir ) * LightPosition[i].w );

		color += d0 * DiffuseColor ;

		//add specular if possible
		if( SpecularPower > 0 )
		{
			float3 reflectDir = reflect( lightDir , pixel.normal );
			float speDen = saturate( dot( reflectDir , eyeDir ) );
			float4 specular = pow( speDen , SpecularPower ) * float4( SpecularColor.xyz , 0.0f );

			color += specular;
		}
	}

	//return pixel.position;
	return color;
}

//the technique
technique PN_Tec
{
	pass DefaultPass
	{
		VertexShader =  compile vs_2_0 VertexShaderPN();
		PixelShader = compile ps_2_0 PixelShaderPN();
	}
}
