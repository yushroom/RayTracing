/*
 *	FileName:	GPURayTracer.h
 *
 *	Programmer:	Jiayin Cao
 */

#pragma once

#include <d3d9.h>
#include <d3dx9.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "CustomScene.h"
#include "Timer.h"

///////////////////////////////////////////////////////////////////////////////////
//	GPURayTracer
class	GPURayTracer
{
//public method
public:
	//constructor and destructor
	GPURayTracer();
	~GPURayTracer();

	//set projection and view matrix
	void	SetMatrix( D3DXMATRIX* view );

	//bind render target
	void	BindRenderTarget( COLORREF*	buf , int w , int h );

	//set scene info
	void	SetScene( CustomScene* scene );

	//ray tracing
	void	RayTrace( bool raster = true );

	//Load content
	void	LoadContent();
	//release content
	void	ReleaseContent();

	//create cuda memory
	void	CreateCUDAMemory();

	//get elapsed time
	int		GetElapsedTime();

//private field
private:
	//the composite matrix
	D3DXMATRIX	m_ViewMatrix;

	//the projection matrix
	D3DXMATRIX	m_ProjectionMatrix;

	//the view matrix
	D3DXMATRIX	m_InvViewMatrix;

	//current resolution for the image
	int		m_iImageWidth;
	int		m_iImageHeight;

	int		m_iResolutionLevel;

	//the depth for ray tracing
	int		m_iDepth;

	//current image buffer
	COLORREF*		m_pImageBuffer;

	//custom scene
	CustomScene*	m_pScene;

	//the cuda memory
	float4*		m_cImageBuffer;		//the buffer for image
	float4*		m_cFinalImageBuffer;//the buffer without noise
	float4*		m_cRayOri;			//the original point of the ray
	float4*		m_cRayDir;			//the direction of the ray
	float4*		m_cIntersectedPoint;//the intersected point
	float4*		m_cTempRayOri;		//temporary ray ori
	float4*		m_cTempRayDir;		//temporary ray dir
	int*		m_cRayMarkedBuffer;	//the marked buffer for the rays
	float4*		m_cBackNormalBuffer;	//the normal buffer
	int*		m_cInteropImage;	//the image buffer for interop
	LPDIRECT3DTEXTURE9	m_d3dInteropRT[3];	//the render target for interop
	LPDIRECT3DSURFACE9	m_d3dInteropSurf[3];//the surface for render target

	LPDIRECT3DTEXTURE9	tex;

	float*		m_cInvViewMatrix;	//the inverse of view matrix
	float4*		m_cKDTreeBuffer;	//kd-tree buffer
	int*		m_cTriIndexBuffer;	//triangle index buffer
	float4*		m_cVertexBuffer;	//the vertex buffer
	float4*		m_cNormalBuffer;	//the normal buffer
	float2*		m_cTexCoordinateBuffer;	//the texture coordinate buffer
	int*		m_cAttributeBuffer;	//the attribute buffer
	float4*		m_cLightBuffer;		//the light buffer
	float4*		m_cMaterialBuffer;	//the material buffer
	float4*		m_cCustomTexture;	//the custom texture memory
	int*		m_cTextureOffset;	//cuda texture offset
	int*		m_cOffsetBuffer;	//the offset buffer

	//the timer
	Timer	m_Timer;
	int		m_ElapsedTime;

//private method

	//initilize default value
	void	InitializeDefault();

	//release cuda memory
	void	ReleaseCUDAMemory();

	//bind gpu buffer
	void	BindBuffer();

	//copy the buffer to result
	void	CopyImageBuffer();

	//Generate primary rays and initialize the buffer
	void	InitializeBuffer( int pixelNum );

	//do raster
	void	RasterTriIndex( int rayNumber );
};