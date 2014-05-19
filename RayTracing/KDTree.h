/*
 *	FileName:	KDTree.h
 *
 *	Programmer:	Jiayin Cao
 */

#pragma once

#include <windows.h>
#include <d3d9.h>
#include <d3dx9.h>
#include <cuda.h>
#include <cuda_runtime.h>

//the vertex format to render the kd-tree
struct KDTree_Vertex
{
	//the position
	float	x , y , z;

	//the format
	static const DWORD KD_FVF = (D3DFVF_XYZ);
};

/////////////////////////////////////////////////////////////////////////
//	KDTree
class	KDTree
{
//public
public:
	//constructor and destructor
	KDTree();
	~KDTree();

	//load the kd-tree from file
	void	LoadKDTree( const char* filename );

	//save kd-tree for debugging
	void	SaveKDTree( const char* filename );

	//release the content
	void	ReleaseContent();

	//draw kd-tree
	void	DrawKDTree( int level = -1 );

	//Update the scene
	void	Update( D3DXMATRIX* composite );

	//the buffer of the kd-tree
	float*	GetBuffer();

	//get the index buffer
	int*	GetIndexBuffer();

	//get offset buffer
	int*	GetOffsetBuffer();

	//get cuda vertex buffer
	float4*	GetCUDABuffer();

	//get the cuda index buffer
	int*	GetCUDAIndexBuffer();

	//get cuda offset buffer
	int*	GetCUDAOffsetBuffer();

//private field
private:
	//the number of nodes in the kd-tree
	int		m_iNodeNumber;

	//the number of primitives
	int		m_iTriNumber;

	//the kd-tree buffer
	float*	m_KDTreeBuffer;

	//the triangle index buffer
	int*	m_TriIndexBuffer;

	//cuda kd-tree buffer
	float4*	m_cKDTreeBuffer;

	//cuda triangle index buffer
	int*	m_cTriIndexBuffer;

	//current bounding box offset
	int		m_iBBOffset;

	//the index buffer for offset ( for large scene )
	int*	m_pOffsetBuffer;

	//the offset buffer for triangle index ( for large scene )
	int*	m_cOffsetBuffer;

	//the vertex buffer for kd-tree
	LPDIRECT3DVERTEXBUFFER9	m_lpVertexes;

//private method

	//initiailze default
	void	InitializeDefault();

	//push current bounding box into the buffer
	void	PushBoundingBox( int index , int level , float* p );

	//push bounding box
	void	PushBoundingBox( KDTree_Vertex*	dest , float* bb );

	//copy memory to gpu
	void	CopyMemoryToGPU();

	//create vertex buffer
	void	CreateVertexBuffer();
};