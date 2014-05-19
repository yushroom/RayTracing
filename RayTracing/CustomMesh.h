/*
 *	Programmer:	Jiayin Cao
 *
 *	FileName:	CustomMesh.h
 *
 *	Description:	Load mesh from obj file
 */

#pragma once

#include <string>
#include <vector>
#include <d3d9.h>
#include <d3dx9.h>
#include "datatype.h"

using namespace std;

//custom material for mesh
#define CustomVertexFormatFNT	(D3DFVF_XYZ | D3DFVF_NORMAL | D3DFVF_TEX1)
#define	CustomVertexFormatFN	(D3DFVF_XYZ | D3DFVF_NORMAL)

struct Custom_Material
{
	//the material name
	string	m_MaterialName;

	_float4			m_Ambient;		//ambient color
	_float4			m_Diffuse;		//diffuse color
	_float4			m_Specular;		//specular color
	int				m_nPower;		//the power of specular

	float			m_fReflect;		//the power of reflection

	float			m_fRefract;		//the power of refract
	float			m_fRefractRate;	//the rate of refraction

	//diffuse texture
	string	m_DifTextureName;
	//the index for the texture
	int		m_iTextureIndex;

	//the texture
	LPDIRECT3DTEXTURE9	m_Texture;

	//default constructor
	Custom_Material::Custom_Material()
	{
		m_Texture = NULL;
		m_nPower = 0;
		m_iTextureIndex = -1;
		m_fReflect = 0;
		m_fRefract = 0;
		m_fRefractRate = 0;
	}
};

//custom vertex
struct Custom_Vertex
{
	// the position
	float	x , y , z;
	// the normal
	float	n_x , n_y , n_z;
	// texture coordinate
	float	u , v;
};

//////////////////////////////////////////////////////////
class	CustomMesh
{
//public method
public:
	//constructor and destructor
	CustomMesh();
	~CustomMesh();

	//Parse obj from file
	void	LoadObjFromFile( const char* filename );

	//Render the mesh
	void	DrawMesh();

	//Release content
	void	ReleaseContent();

	//Update matrix
	void	Update( D3DXMATRIX* composite , D3DXVECTOR3* eye );

	//Set directional light
	void	SetLightPosition( float* dir );

	//Set world matrix
	void	SetWorldMatrix( D3DXMATRIX&	mat );

	//Get world matrix
	D3DXMATRIX&	GetWorldMatrix();

	//Get the vertex buffer
	vector<Custom_Vertex>&	GetVertexBuffer();

	//Get atrribute buffer
	vector<int>&			GetAttributeBuffer();

	//Get Material list
	vector<Custom_Material*>& GetCustomMaterial();

	//Get the number of vertex
	UINT	GetVertexNumber();

	//Get Material number
	UINT	GetMaterialNumber();

//private field
private:
	//the material list
	vector<Custom_Material*>	m_Materials;

	//the path of the media
	char	m_MediaPath[256];

	//the final vertex buffer
	vector<Custom_Vertex>	m_VertexBuffer;			//for each vertex
	vector<int>				m_AttributeBuffer;		//for each face
	vector<int>				m_SubsetVertexCount;	//for each subset

	//the world matrix for the mesh
	D3DXMATRIX				m_WorldMatrix;
	//composite matrix
	D3DXMATRIX				m_CompositeMatrix;
	//eye vector
	D3DXVECTOR4				m_EyePosition;

	//the vertex and index buffer
	LPDIRECT3DVERTEXBUFFER9	m_lpVertexes;

//private method
	
	//Initialize default
	void	InitializeDefault();

	//Update material for shader effect
	void	UpdateMaterial( int matIndex );

	//Parse material file
	void	ParseMaterial( const char* filename );

	//Release the material
	void	ReleaseMaterial();

	//Get material index
	int		GetMaterialIndex( const char* matName );

	//parse the index
	void	ParseIndex( Custom_Vertex* vertex , const char* buffer , vector<_float4>& pos , vector<_float4>& nor , vector<_float4>& tex );

	//create the vertex buffer
	void	CreateVertexBuffer();
};