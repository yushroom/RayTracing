/*
 *	Programmer:	Jiayin Cao
 *
 *	FileName:	CustomScene.h
 */

#pragma once

#include "CustomMesh.h"
#include "TinyXml\TinyXml.h"
#include "KDTree.h"
#include "datatype.h"

///////////////////////////////////////////////////////////////////////
class	CustomScene
{
//public method
public:
	//constructor and destructor
	CustomScene();
	~CustomScene();

	//release the resource
	void	ReleaseContent();

	//Load scene from file
	bool	LoadScene( const char* filename );

	//Draw scene
	void	DrawScene();

	//Update the scene
	void	Update( D3DXMATRIX* composite , D3DXVECTOR3* eye );

	//Set directional light
	void	SetDirectionalLight();

	//Get Light dir
	void	GetLightPosition( float* dir , int index );

	//Get Vertex number
	UINT	GetVertexNumber();

	//get vertex stride
	UINT	GetVertexStride();

	//Get kd-tree
	KDTree*	GetKDTree();

	//Get total vertex buffer
	_float4*	GetVertexBuffer();
	//get normal buffer
	_float4*	GetNormalBuffer();
	//get texture coordinate buffer
	float*	GetTextureCoodinateBuffer();
	//get material buffer
	Custom_Material** GetMaterialBuffer();
	//get attribute buffer
	int*	GetAttributeBuffer();

	//Get cuda vertex buffer
	float4*	GetCUDAVertexBuffer();
	float4*	GetCUDANormalBuffer();
	float2*	GetCUDATexCoordinateBuffer();
	int*	GetCUDAAttributeBuffer();
	float4*	GetCUDALightBuffer();
	float4*	GetCUDAMaterialBuffer();

	//get the number of the lights
	int		GetLightNumber();

	//show triId in render target
	void	DrawTriangleID( LPDIRECT3DSURFACE9	rt , D3DXMATRIX* proj );

//private field
private:
	//the list for the mesh
	CustomMesh*		m_pMeshList;

	//the number of entities
	int				m_iEntityNumber;

	//the custom vertex for ray tracing
	_float4*			m_pVertexBuffer;
	//the normal buffer
	_float4*			m_pNormalBuffer;
	//the texture coordinate buffer
	float*				m_pTexCoordinateBuffer;
	//the attribute buffer for the vertexes
	int*				m_pAttributeBuffer;
	//the material buffer
	Custom_Material**	m_pMaterialBuffer;

	//cuda memory
	float4*			m_cVertexBuffer;
	float4*			m_cNormalBuffer;
	float2*			m_cTexCoordinateBuffer;
	int*			m_cAttributeBuffer;
	float4*			m_cMaterialBuffer;
	float4*			m_cLightBuffer;

	//the number of vertex
	int				m_iVertexNumber;
	//the material number
	int				m_iMaterialNumber;
	//the face number
	int				m_iFaceNumber;

	//the direction for the light
	D3DXVECTOR4		m_LightPos[2];
	//total light number
	int				m_LightNum;

	//the kd-tree
	KDTree			m_KDTree;
	//enable kd-tree rendering
	bool			m_bKDTreeVisible;

//private method

	//Parse Entity
	void	ParseEntities( TiXmlElement* node );
	void	ParseEntity( TiXmlElement* node , CustomMesh* mesh );
	
	//Load kd-tree
	void	ParseKDTree( TiXmlElement* node );

	//Parse the lights
	void	ParseLights( TiXmlElement* node );

	//initialize default
	void	InitializeDefault();

	//Parse matrix
	D3DXMATRIX	ParseMatrix( const char* matrixBuffer );

	//Load the vertex buffer
	void	LoadVertexBuffer();
	//Load attribute buffer
	void	LoadAttributeBuffer();

	//copy memory to gpu
	void	CopyMemoryToGPU();
};