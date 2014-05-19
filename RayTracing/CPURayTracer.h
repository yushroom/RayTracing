/*
 *	FileName:	CPURayTracer.h
 *
 *	Programmer:	Jiayin Cao
 */

#pragma once

#include <d3d9.h>
#include <d3dx9.h>
#include "CustomScene.h"
#include "datatype.h"
#include "Timer.h"

///////////////////////////////////////////////////////////////////
//	class	CPURayTracer
class	CPURayTracer
{
//public method
public:
	//constructor and destructor
	CPURayTracer();
	~CPURayTracer();

	//set projection and view matrix
	void	SetMatrix( D3DXMATRIX* view );

	//bind render target
	void	BindRenderTarget( COLORREF*	buf , int w , int h );

	//set scene info
	void	SetScene( CustomScene* scene );

	//ray tracing
	void	RayTrace();

	//reset current pixel number
	void	ResetCurrentPixelNum();

	//get current pixel number
	int		GetCurrentPixelNum();

	//get elapsed time
	int		GetElapsedTime();

	//enable force stop
	void	ForceStop( bool enable );

//private field
private:
	//the projection matrix
	D3DXMATRIX	m_ProjectionMatrix;

	//the view matrix
	D3DXMATRIX	m_InvViewMatrix;

	//current resolution for the image
	int		m_iImageWidth;
	int		m_iImageHeight;

	//current pixel number
	int		m_iCurrentPixelNum;

	//current image buffer
	COLORREF*		m_pImageBuffer;

	//custom scene
	CustomScene*	m_pScene;

	//the buffer for scene
	float*				m_pKDTreeBuffer;	//kd-tree buffer
	_float4*			m_pVertexBuffer;	//vertex buffer
	_float4*			m_pNormalBuffer;	//the normal buffer
	float*				m_pTexCoordinateBuffer;	//texture coordinate buffer
	int*				m_pIndexBuffer;		//index buffer
	int*				m_pAttributeBuffer;	//attribute buffer
	Custom_Material**	m_pMaterialBuffer;	//the material buffer
	_float4				m_EyePosition;		//the eye position
	_float4*			m_pCustomTexture;	//the custom texture
	int*				m_pTextureOffset;	//the texture offset
	int*				m_pOffsetBuffer;	//the offset buffer for triangles

	//the image buffer
	_float4*			m_pImageBackBuffer;

	//the timer
	Timer	m_Timer;
	//elapsed time
	int		m_ElapsedTime;

	//force the ray tracer to stop
	bool	m_bForceStop;

//private method

	//initialize default value
	void	InitializeDefault();

	//Generate a ray for current pixel
	void	GenerateRay( int x , int y , _float4* ori , _float4* dir );

	//set image resolution
	void	SetImageResolution( int w , int h );

	//set buffer
	void	SetImageBuffer( COLORREF* buf );

	//trace a ray
	_float4 Trace( _float4 ori , _float4 dir );

	//intersection test for a ray and a bounding box
	bool	GetIntersectedBoundingBox( _float4 ori , _float4 dir , _float4 min , _float4 max , _float4* intersected , float len = -1 );

	//get intersected point between a ray and a plane
	bool	GetIntersectedTriangle( _float4 ori , _float4 dir , _float4 v1 , _float4 v2 , _float4 v3 , _float4* intersected );

	//get intersected point between a ray and a triangle
	bool	GetIntersectedPlane( _float4 ori , _float4 dir , _float4 v1 , _float4 v2 , _float4 v3 , _float4* intersected );

	//traverse a ray through kd-tree to find a intersected point
	bool	TraverseRay( _float4 ori , _float4 dir , _float4* intersected  , int nodeIndex = 0 , float len = -1 );

	//check if a point is in the bounding box
	bool	PointInBoundingBox( _float4 p , _float4 min , _float4 max );

	//shade a pixel
	_float4	PixelShader( _float4* intersected , _float4 ori , float den = 1.0f );

	//interplote normal
	void	GetInterploteFactor( _float4 v1 , _float4 v2 , _float4 v3 , _float4 intersected , _float4* factor );
	//get interploted normal
	void	GetInterplotedNormal( _float4 factor , _float4 n1 , _float4 n2 , _float4 n3 , _float4* result );
	//get interploted texture coordinate
	void	GetInterplotedTextureCoord( float* factor , float* t1 , float* t2 , float* t3 , float* result );

	//get a pixel from texture
	void	GetPixelFromTexture( int texIndex , float u , float v , _float4* color );

	//check if the intersected point is in the shadow
	bool	PointInShadow( _float4 p , _float4 lightPos );

	//bind buffer for current scene
	void	BindBuffer();

	//clear the noise of the image
	void	ClearNoise( _float4* buffer );
};