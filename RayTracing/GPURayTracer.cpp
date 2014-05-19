/*
 *	FileName:	GPURayTracer.cpp
 *
 *	Programmer:	Jiayin Cao
 */

#include "GPURayTracer.h"
#include "Kernel_Interface.h"
#include "D3DResource.h"
#include <cuda_d3d9_interop.h>

//constructor and destructor
GPURayTracer::GPURayTracer()
{
	//set default value
	InitializeDefault();
}

GPURayTracer::~GPURayTracer()
{
	//release cuda memory
	ReleaseCUDAMemory();
}

//create cuda memory
void GPURayTracer::CreateCUDAMemory()
{
	//create large enough memory for gpu
	cudaMalloc( (void**)&m_cFinalImageBuffer , sizeof( float4 ) * 768 * 1024 );
	cudaMalloc( (void**)&m_cImageBuffer , sizeof( float4 ) * 768 * 1024 );
	cudaMalloc( (void**)&m_cInvViewMatrix , sizeof( float ) * 16 );
	cudaMalloc( (void**)&m_cRayOri , sizeof( float4 ) * 768 * 1024 );
	cudaMalloc( (void**)&m_cRayDir , sizeof( float4 ) * 768 * 1024 );
	cudaMalloc( (void**)&m_cIntersectedPoint , sizeof( float4 ) * 768 * 1024 );
	cudaMalloc( (void**)&m_cTempRayOri , sizeof( float4 ) * 768 * 1024 );
	cudaMalloc( (void**)&m_cTempRayDir , sizeof( float4 ) * 768 * 1024 );
	cudaMalloc( (void**)&m_cRayMarkedBuffer , sizeof( int ) * ( 768 * 1024 + 512 ) );
	cudaMalloc( (void**)&m_cBackNormalBuffer , sizeof( float4 ) * 768 * 1024 );
	cudaMalloc( (void**)&m_cInteropImage , sizeof( int ) * 768 * 1024 );

	LPDIRECT3DDEVICE9 device = D3DResource::GetSingleton()->GetDevice();
	device->CreateOffscreenPlainSurface( 640 , 480 , D3DFMT_R32F , D3DPOOL_SYSTEMMEM , &m_d3dInteropSurf[0] , NULL );
	device->CreateOffscreenPlainSurface( 800 , 600 , D3DFMT_R32F , D3DPOOL_SYSTEMMEM , &m_d3dInteropSurf[1] , NULL );
	device->CreateOffscreenPlainSurface( 1024 , 768 , D3DFMT_R32F , D3DPOOL_SYSTEMMEM , &m_d3dInteropSurf[2] , NULL );
}

//release cuda memory
void GPURayTracer::ReleaseCUDAMemory()
{
	//release memory
	SAFE_RELEASE_CUDA( m_cIntersectedPoint );
	SAFE_RELEASE_CUDA( m_cImageBuffer );
	SAFE_RELEASE_CUDA( m_cInvViewMatrix );
	SAFE_RELEASE_CUDA( m_cRayOri );
	SAFE_RELEASE_CUDA( m_cRayDir );
	SAFE_RELEASE_CUDA( m_cTempRayOri );
	SAFE_RELEASE_CUDA( m_cTempRayDir );
	SAFE_RELEASE_CUDA( m_cRayMarkedBuffer );
	SAFE_RELEASE_CUDA( m_cBackNormalBuffer );
	SAFE_RELEASE_CUDA( m_cInteropImage );
	SAFE_RELEASE_CUDA( m_cFinalImageBuffer );

	for( int i = 0 ; i < 3 ; i++ )
		SAFE_RELEASE( m_d3dInteropSurf[i] );
}

//initilize default value
void GPURayTracer::InitializeDefault()
{
	m_iImageWidth = 0;
	m_iImageHeight = 0;
	m_pImageBuffer = 0;
	m_pScene = 0;
	m_iDepth = 5;
	m_iResolutionLevel = 0;
	m_ElapsedTime = 0;

	m_cFinalImageBuffer = 0;
	m_cImageBuffer = 0;
	m_cRayOri = 0;
	m_cRayDir = 0;
	m_cVertexBuffer = 0;
	m_cVertexBuffer = 0;
	m_cNormalBuffer = 0;
	m_cTexCoordinateBuffer = 0;
	m_cInvViewMatrix = 0;
	m_cIntersectedPoint = 0;
	m_cLightBuffer = 0;
	m_cMaterialBuffer = 0;
	m_cAttributeBuffer = 0;
	m_cCustomTexture = 0;
	m_cTextureOffset = 0;
	m_cTempRayOri = 0;
	m_cTempRayDir = 0;
	m_cRayMarkedBuffer = 0;
	m_cBackNormalBuffer = 0;
	m_cInteropImage = 0;
}

//set projection and view matrix
void GPURayTracer::SetMatrix( D3DXMATRIX* view )
{
	//the view matrix
	m_ViewMatrix = *view;

	//the inverse of view matrix
	D3DXMatrixInverse( &m_InvViewMatrix , NULL , view );

	//copy the memroy to gpu
	cudaMemcpy( (void*)m_cInvViewMatrix , &m_InvViewMatrix , sizeof( float ) * 16 , cudaMemcpyHostToDevice );
}

//bind render target
void GPURayTracer::BindRenderTarget( COLORREF*	buf , int w , int h )
{
	//check if there is need to allocate the data for new resolution
	if( w == m_iImageWidth && h == m_iImageHeight )
		return;

	//copy the data
	m_iImageWidth = w;
	m_iImageHeight = h;

	float aspect = (float) w / (float) h;
	D3DXMatrixPerspectiveFovLH( &m_ProjectionMatrix , D3DX_PI/4 , aspect , 1.0f , 5000.0f );

	//set image buffer
	m_pImageBuffer = buf;

	//set resolution level
	switch( w )
	{
	case 640:
		m_iResolutionLevel = 0;
		break;
	case 800:
		m_iResolutionLevel = 1;
		break;
	case 1024:
		m_iResolutionLevel = 2;
		break;
	}
}

//set scene info
void GPURayTracer::SetScene( CustomScene* scene )
{
	m_pScene = scene;
}

//ray tracing
void GPURayTracer::RayTrace( bool raster )
{
	//reset the timer
	m_Timer.Reset();
	m_Timer.Start();

	//bind the buffer first
	BindBuffer();

	//current trace level
	int curTraceLevel = 0;

	//current level ray number
	int rayNumber = m_iImageWidth * m_iImageHeight;

	//generate primary rays
	InitializeBuffer(rayNumber);

	//raster tri index
	if( raster == true )
		RasterTriIndex(rayNumber);

	while( curTraceLevel < m_iDepth && rayNumber > 0 )
	{
		if( curTraceLevel != 0 || raster == false )
		{
			//get intersected point
			cudaGetIntersectedPoint( m_cRayOri , m_cRayDir , m_cKDTreeBuffer , m_cTriIndexBuffer , m_cOffsetBuffer , m_cVertexBuffer , rayNumber , m_cIntersectedPoint );
		}

		//do pixel shader
		cudaPixelShader(m_cIntersectedPoint , m_cVertexBuffer , m_cNormalBuffer , m_cTexCoordinateBuffer , m_cKDTreeBuffer , m_cTriIndexBuffer , m_cOffsetBuffer ,
						m_cLightBuffer , m_cAttributeBuffer , m_cMaterialBuffer , m_cTextureOffset , m_cCustomTexture , rayNumber , m_cRayDir , 
						m_cRayMarkedBuffer , m_cBackNormalBuffer , m_cImageBuffer );

		//generate next level rays
		cudaGenerateNextLevelRays(	m_cMaterialBuffer , m_cIntersectedPoint , m_cBackNormalBuffer , 
									m_cRayOri , m_cRayDir , rayNumber , m_cTempRayOri , m_cTempRayDir , m_cRayMarkedBuffer );

		//do scan on gpu
		cudaScan( m_cRayMarkedBuffer , rayNumber + 1 );

		//get the number of rays
		int newrayNum;
		cudaMemcpy( &newrayNum , m_cRayMarkedBuffer + rayNumber , sizeof(int) , cudaMemcpyDeviceToHost );
		if( newrayNum == 0 )
			break;

		//copy the new rays
		cudaCopyNewRays( m_cTempRayOri , m_cTempRayDir , m_cRayMarkedBuffer , rayNumber , m_cRayOri , m_cRayDir , m_cRayMarkedBuffer );

		//update the ray number
		rayNumber = newrayNum;

		//update current trace level
		curTraceLevel++;
	}

	cudaClearNoise( m_cImageBuffer , m_iImageWidth , m_iImageHeight , m_cFinalImageBuffer );

	//stop the timer
	m_Timer.Stop();
	m_ElapsedTime = (int)m_Timer.GetElapsedTime();

	//copy the buffer
	CopyImageBuffer();
}

//bind gpu buffer
void GPURayTracer::BindBuffer()
{
	m_cKDTreeBuffer = m_pScene->GetKDTree()->GetCUDABuffer();
	m_cTriIndexBuffer = m_pScene->GetKDTree()->GetCUDAIndexBuffer();
	m_cVertexBuffer = m_pScene->GetCUDAVertexBuffer();
	m_cNormalBuffer = m_pScene->GetCUDANormalBuffer();
	m_cTexCoordinateBuffer = m_pScene->GetCUDATexCoordinateBuffer();
	m_cAttributeBuffer = m_pScene->GetCUDAAttributeBuffer();
	m_cLightBuffer = m_pScene->GetCUDALightBuffer();
	m_cMaterialBuffer = m_pScene->GetCUDAMaterialBuffer();
	m_cCustomTexture = D3DResource::GetSingleton()->GetCUDATexture();
	m_cTextureOffset = D3DResource::GetSingleton()->GetCUDATextureOffset();
	m_cOffsetBuffer = m_pScene->GetKDTree()->GetCUDAOffsetBuffer();
}

//copy the buffer to result
void GPURayTracer::CopyImageBuffer()
{
	//the size of the image
	int size = m_iImageWidth * m_iImageHeight;

	//allocate the new memory
	_float4* img = new _float4[size];

	//copy the memory back
	cudaMemcpy( img , m_cFinalImageBuffer , size * sizeof( float4 ) , cudaMemcpyDeviceToHost );

	//copy the data to buffer
	for( int i = 0 ; i < m_iImageHeight ; i++ )
	{
		for( int j = 0 ; j < m_iImageWidth ; j++ )
		{
			int srcOffset = i * m_iImageWidth + j;
			int destOffset = ( m_iImageHeight - i - 1 ) * m_iImageWidth + j;

			m_pImageBuffer[srcOffset] = RGB_FLOAT4(img[destOffset]);
		}
	}

	//delete the memroy
	delete[] img;
}

//Generate primary rays
void GPURayTracer::InitializeBuffer( int pixelNum )
{
	//generate primary ray
	float4 viewInfo;
	viewInfo.x = (float)m_iImageWidth;
	viewInfo.y = (float)m_iImageHeight;
	viewInfo.z = m_ProjectionMatrix._11;
	viewInfo.w = m_ProjectionMatrix._22;
	cudaGeneratePrimaryRays( viewInfo , m_cInvViewMatrix , m_cRayOri , m_cRayDir );

	//initialize buffer
	cudaInitBuffer( m_cImageBuffer , m_cRayMarkedBuffer , pixelNum );
}

//load content
void	GPURayTracer::LoadContent()
{
	//get the device
	LPDIRECT3DDEVICE9 device = D3DResource::GetSingleton()->GetDevice();

	//create the texture
	device->CreateTexture(	640 , 480 , 1 , D3DUSAGE_RENDERTARGET , D3DFMT_R32F , D3DPOOL_DEFAULT , &m_d3dInteropRT[0] , NULL );
	device->CreateTexture(	800 , 600 , 1 , D3DUSAGE_RENDERTARGET , D3DFMT_R32F , D3DPOOL_DEFAULT , &m_d3dInteropRT[1] , NULL );
	device->CreateTexture(	1024 , 768 , 1 , D3DUSAGE_RENDERTARGET , D3DFMT_R32F , D3DPOOL_DEFAULT , &m_d3dInteropRT[2] , NULL );
}

//release content
void GPURayTracer::ReleaseContent()
{
	//release texture
	for( int i = 0; i < 3 ; i++ )
		SAFE_RELEASE(m_d3dInteropRT[i]);
}

//do raster
void GPURayTracer::RasterTriIndex( int rayNumber )
{
	//get the render target
	LPDIRECT3DSURFACE9	surface;

	//get surface level
	m_d3dInteropRT[m_iResolutionLevel]->GetSurfaceLevel( 0 , &surface );

	//draw tri id
	D3DXMATRIX composite = m_ViewMatrix * m_ProjectionMatrix ;
	m_pScene->DrawTriangleID( surface , &composite );

	//get the data out
	LPDIRECT3DDEVICE9 device = D3DResource::GetSingleton()->GetDevice();
	HRESULT hr = device->GetRenderTargetData( surface , m_d3dInteropSurf[m_iResolutionLevel] );

	//get the data out
	D3DSURFACE_DESC desc;
	m_d3dInteropSurf[m_iResolutionLevel]->GetDesc( &desc );
	int width = desc.Width;
	int height = desc.Height;

	int* data = new int[width*height];

	//lock the rect
	D3DLOCKED_RECT d3d_Rect;
	m_d3dInteropSurf[m_iResolutionLevel]->LockRect( &d3d_Rect , NULL , D3DLOCK_READONLY );

	//get the data
	float*	pData = (float*)d3d_Rect.pBits;

	//copy the data
	for( int i = 0 ; i < width * height ; i++ )
	{
		//get the index
		int x = i % width;
		int y = i / width;

		//get the pixel
		data[i] = (int)pData[ y * d3d_Rect.Pitch / sizeof( float ) + x ] - 1;
	}

	//pass the memory to gpu
	cudaMemcpy( m_cInteropImage , data , sizeof( float ) * width * height , cudaMemcpyHostToDevice );

	//unlock the data
	m_d3dInteropSurf[m_iResolutionLevel]->UnlockRect();

	//generate primary ray intersected result by raster
	cudaGenerateIntersectedPoint( m_cRayOri , m_cRayDir , m_cVertexBuffer , rayNumber , m_cInteropImage , m_cIntersectedPoint );

	//release the surface
	SAFE_RELEASE( surface );

	//delete the data
	delete[] data;
}

//get elapsed time
int	GPURayTracer::GetElapsedTime()
{
	return m_ElapsedTime;
}