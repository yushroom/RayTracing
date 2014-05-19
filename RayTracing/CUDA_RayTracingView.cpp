// CUDA_RayTracingView.cpp : implementation of the CCUDA_RayTracingView class
//

#include "stdafx.h"
#include "define.h"
#include "CUDA_RayTracing.h"
#include "CUDA_RayTracingDoc.h"
#include "CUDA_RayTracingView.h"
#include "D3DResource.h"
#include "define.h"
#include "MainFrm.h"
#include <process.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CCUDA_RayTracingView

IMPLEMENT_DYNCREATE(CCUDA_RayTracingView, CView)

BEGIN_MESSAGE_MAP(CCUDA_RayTracingView, CView)
	ON_WM_SIZE()
	ON_WM_DESTROY()
	ON_WM_TIMER()
	ON_WM_MOUSEMOVE()
	ON_WM_MOUSEWHEEL()
	ON_WM_CREATE()
END_MESSAGE_MAP()

// CCUDA_RayTracingView construction/destruction

CCUDA_RayTracingView::CCUDA_RayTracingView()
{
	//Initialize Default Value first
	InitializeDefault();
}

CCUDA_RayTracingView::~CCUDA_RayTracingView()
{
}

BOOL CCUDA_RayTracingView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	return CView::PreCreateWindow(cs);
}

// CCUDA_RayTracingView drawing

void CCUDA_RayTracingView::OnDraw(CDC* pDC)
{
	CCUDA_RayTracingDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;
}

// CCUDA_RayTracingView diagnostics

#ifdef _DEBUG
void CCUDA_RayTracingView::AssertValid() const
{
	CView::AssertValid();
}

void CCUDA_RayTracingView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CCUDA_RayTracingDoc* CCUDA_RayTracingView::GetDocument() const // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CCUDA_RayTracingDoc)));
	return (CCUDA_RayTracingDoc*)m_pDocument;
}
#endif //_DEBUG

////////////////////////////////////////////////////////////////////////////////////////
//	MFC MESSAGE

//Initialize
void CCUDA_RayTracingView::OnInitialUpdate()
{
	CView::OnInitialUpdate();

	//Create D3D
	bool hr = InitializeD3D();
	if( false == hr )
		return;

	//show the window
	m_LoadWindow.ShowWindow( SW_SHOWNORMAL );

	//set the device
	D3DResource::GetSingleton()->SetDevice( m_lpd3dDevice );

	//Load the scene
	m_LoadWindow.UpdateProgress( 0 );
	m_Scene[0].LoadScene( "scene/cornell.xml" );
	m_LoadWindow.UpdateProgress( 25 );
	m_Scene[1].LoadScene( "scene/toasters.xml" );
	m_LoadWindow.UpdateProgress( 50 );
	m_Scene[2].LoadScene( "scene/chess.xml" );
	m_LoadWindow.UpdateProgress( 75 );
	m_Scene[3].LoadScene( "scene/dragon_on_table.xml" );
	m_LoadWindow.UpdateProgress( 95 );

	//create cuda memory
	m_gpuRayTracer.CreateCUDAMemory();

	//load the content for gpu ray tracer
	m_gpuRayTracer.LoadContent();

	//load the resource
	D3DResource::GetSingleton()->LoadContent();

	//close the window
	m_LoadWindow.CloseWindow();
	m_LoadWindow.DestroyWindow();

	//Set the timer
	SetTimer( 0 , 16 , NULL );
}

//Update if the size of the client is changed
void CCUDA_RayTracingView::OnSize(UINT nType, int cx, int cy)
{
	CView::OnSize(nType, cx, cy);

	if( m_lpD3D == 0 )
		return;
	if( cx == 0 || cy == 0 )
		return;
	if( cx == m_parm.BackBufferWidth && cy == m_parm.BackBufferHeight )
		return;

	//update the projection matrix
	UpdateProjectionMatrix( cx , cy );

	//set camera client size
	m_Camera.SetClientRect( cx , cy );
}

//destroy the demo
void CCUDA_RayTracingView::OnDestroy()
{
	//stop ray tracing first
	m_cpuRayTracer.ForceStop( true );
	Sleep(10);

	CView::OnDestroy();

	//destroy the singleton
	D3DResource::Destroy();
	ReleaseD3D();

	//release the content
	for( int i = 0 ; i < 4 ; i++ )
		m_Scene[i].ReleaseContent();
}

//on timer
void CCUDA_RayTracingView::OnTimer(UINT_PTR nIDEvent)
{
	if( nIDEvent == 0 )
	{
		//update logic
		Update();

		//draw the scene
		Render();

		//send message if interactive ray tracing is enable
		if( m_bEnableInteractiveRT && m_bShowNextFrame )
		{
			//disable first
			m_bShowNextFrame = false;

			//send a timer message
			SendMessage( WM_TIMER , 2 );
		}
	}else if( nIDEvent == 1 )
	{
		//update the buffer
		m_RenderWindow.ShowImage();
	}else if( nIDEvent == 2 )
	{
		//draw a gpu frame
		RayTraceOnGPU();

		//show the image
		m_RenderWindow.Refresh();

		//enable the next frame
		m_bShowNextFrame = true;
	}

	CView::OnTimer(nIDEvent);
}

//mouse move
void CCUDA_RayTracingView::OnMouseMove(UINT nFlags, CPoint point)
{
	//update the cursor position
	m_CursorPosition.x = point.x;
	m_CursorPosition.y = point.y;

	//whether left button is down
	m_bLeftButton = ((nFlags & MK_LBUTTON )!=0);

	//whether to move the camera
	m_bMoveCamera = ((nFlags & MK_MBUTTON )!=0 || (nFlags & MK_RBUTTON )!=0);

	CView::OnMouseMove(nFlags, point);
}

//mouse wheel
BOOL CCUDA_RayTracingView::OnMouseWheel(UINT nFlags, short zDelta, CPoint pt)
{
	m_iWheelPos += (int)((float)zDelta / 50.0f);

	return CView::OnMouseWheel(nFlags, zDelta, pt);
}

//on create
int CCUDA_RayTracingView::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CView::OnCreate(lpCreateStruct) == -1)
		return -1;

	//create render window
	m_RenderWindow.Create( IDD_RENDERWINDOW );
	m_LoadWindow.Create( IDD_LOADING );

	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////
//	CUSTOM FUNCTIONS

//Initialize default value
void CCUDA_RayTracingView::InitializeDefault()
{
	m_iSelecetedSceneID = 0;
	m_lpD3D = NULL;
	m_lpd3dDevice = NULL;
	m_iWheelPos = 0;
	m_bLeftButton = 0;
	m_bMoveCamera = 0;
	m_bDeviceLost = false;
	m_Camera.InitializeDefault();
	m_Camera.SetCamera( 0.0 , 0.3f , 10.0f , D3DXVECTOR3( 0 , 0 , 0 ) );
	m_iKDTreeLevel = -1;
	m_iMaxResolution.x = 1280;
	m_iMaxResolution.y = 1024;
	m_bShowKDTree = false;
	m_bEnableInteractiveRT = false;
	m_bShowNextFrame = true;
}

//initialize direct3d
bool CCUDA_RayTracingView::InitializeD3D()
{
	//create d3d
	m_lpD3D = Direct3DCreate9( D3D_SDK_VERSION );

	//if create d3d failed, return
	if( NULL == m_lpD3D )
	{
		MessageBox( L"Create D3D Failed" , L"Error" );
		return false;
	}

	//try to get the client size
	RECT	client;
	GetClientRect( &client );

	//clear the memory
	memset( &m_parm , 0 , sizeof( m_parm ) );
	m_parm.Windowed = true;
	m_parm.BackBufferWidth = m_iMaxResolution.x;
	m_parm.BackBufferHeight = m_iMaxResolution.y;
	m_parm.BackBufferFormat = D3DFMT_A8R8G8B8;
	m_parm.BackBufferCount = 1;
	m_parm.Flags = D3DPRESENTFLAG_DISCARD_DEPTHSTENCIL;
	m_parm.EnableAutoDepthStencil = true;
	m_parm.AutoDepthStencilFormat = D3DFMT_D24S8;
	m_parm.hDeviceWindow = m_hWnd;
	m_parm.SwapEffect = D3DSWAPEFFECT_DISCARD;

	//Create the direct3d device
	m_lpD3D->CreateDevice(	D3DADAPTER_DEFAULT , D3DDEVTYPE_HAL , m_hWnd , 
							D3DCREATE_HARDWARE_VERTEXPROCESSING , &m_parm , &m_lpd3dDevice );

	if( NULL == m_lpd3dDevice )
	{
		MessageBox( L"Create device Failed" , L"Error" );
		return false;
	}

	m_lpd3dDevice->SetRenderState( D3DRS_CULLMODE , D3DCULL_NONE );

	return true;
}

//release direct3d
void CCUDA_RayTracingView::ReleaseD3D()
{
	//release the com interface
	SAFE_RELEASE( m_lpd3dDevice );
	SAFE_RELEASE( m_lpD3D );
}

//Render the scene
void CCUDA_RayTracingView::Render()
{
	//Clear the client
	m_lpd3dDevice->Clear( 0 , NULL , D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER , D3DXCOLOR( 0.5f , 0.5f , 0.5f , 1.0f ) , 1.0f , 0 );

	//if the device is not lost
	if( m_bDeviceLost == false && D3DResource::GetSingleton()->ResourceReady() )
	{
		//begin to draw
		m_lpd3dDevice->BeginScene();

		//draw the scene
		m_Scene[m_iSelecetedSceneID].DrawScene();

		//draw the kd-tree
		if( m_bShowKDTree )
			m_Scene[m_iSelecetedSceneID].GetKDTree()->DrawKDTree(m_iKDTreeLevel);

		//end drawing
		m_lpd3dDevice->EndScene();
	}

	//present the back buffer
	if( FAILED( m_lpd3dDevice->Present( 0 , 0 , 0 , 0 ) ) )
		m_bDeviceLost = true;

	//if device is lost , restore it
	if( m_bDeviceLost )
		RestoreDevice();
}

//Update the logic
void CCUDA_RayTracingView::Update()
{
	if( D3DResource::GetSingleton()->ResourceReady() == false )
		return;

	//Update the camera
	m_Camera.Update( &m_CursorPosition , m_bLeftButton , m_iWheelPos , m_bMoveCamera , m_ProjMatrix );

	//update matrix for the mesh
	D3DXMATRIX composite = m_Camera.GetViewMatrix() * m_ProjMatrix ;

	//update the scene
	m_Scene[m_iSelecetedSceneID].Update( &composite , &m_Camera.GetEyePosition() );
}

//Restore device
bool CCUDA_RayTracingView::RestoreDevice()
{
	HRESULT hr = m_lpd3dDevice->TestCooperativeLevel();

	if( hr != D3D_OK )
	{
		if( D3DERR_DEVICELOST == hr )
			return false;

		if( D3DERR_DEVICENOTRESET == hr )
		{
			//Load the content
			D3DResource::GetSingleton()->ReleaseContent( false );
			m_gpuRayTracer.ReleaseContent();

			//reset device
			if( FAILED( m_lpd3dDevice->Reset( &m_parm ) ) )
			{
				m_bDeviceLost = false;
				return false;
			}

			//Load content
			D3DResource::GetSingleton()->LoadContent();
			m_gpuRayTracer.LoadContent();

			//Get client rect
			RECT client;
			GetClientRect( &client );

			//Get the window size
			UpdateProjectionMatrix( client.right - client.top , client.bottom - client.top );

			//disable device lost
			m_bDeviceLost = false;
		}
	}

	return true;
}

//Update projection matrix
void CCUDA_RayTracingView::UpdateProjectionMatrix( int cx , int cy )
{
	//update the projection matrix
	float aspect = (float)cx / (float) cy;
	D3DXMatrixPerspectiveFovLH( &m_ProjMatrix , D3DX_PI/4 , aspect , 1.0f , 10000.0f );
}

//ray tracing on cpu
void CCUDA_RayTracingView::CPURayTracingThread( void* p )
{
	//get view
	CCUDA_RayTracingView* view = (CCUDA_RayTracingView*)p;

	//do ray tracing
	view->m_cpuRayTracer.RayTrace();

	//show the image
	view->m_RenderWindow.ShowImage();

	//kill the timer
	view->KillTimer(1);

	//update control panel
	view->UpdateControlPanel( view->m_Scene[view->m_iSelecetedSceneID].GetVertexNumber() , view->m_cpuRayTracer.GetElapsedTime() , CString( L"CPU" ) );

	//enable ray trace button
	view->m_pControlPanel->EnableAllComponent( TRUE );

	//set cpu ray tracing
	view->m_RenderWindow.SetCPURayTrace( false );
}

//do cpu ray tracing
void CCUDA_RayTracingView::RayTraceOnCPU()
{
	//reset current pixel number
	m_cpuRayTracer.ResetCurrentPixelNum();

	//change the render window resolution first
	m_RenderWindow.ChangeImageResolution();

	//clear the render window
	m_RenderWindow.ClearBuffer();

	//set cpu ray tracing
	m_RenderWindow.SetCPURayTrace( true );

	//set the resolution of the image
	COLORREF* buf = m_RenderWindow.GetBuffer();
	int	w = m_RenderWindow.GetImageWidth();
	int h = m_RenderWindow.GetImageHeight();
	m_cpuRayTracer.BindRenderTarget( buf , w , h );

	//update the matrix for the ray tracer
	m_cpuRayTracer.SetMatrix( &m_Camera.GetViewMatrix() );

	//set scene information
	m_cpuRayTracer.SetScene( &m_Scene[m_iSelecetedSceneID] );

	//stop force
	m_cpuRayTracer.ForceStop( false );

	//set a new timer to update the process
	SetTimer( 1 , 33 , 0 );

	//disable the button
	m_pControlPanel->EnableAllComponent( FALSE );

	//create a new thread
	_beginthread( CPURayTracingThread , 1 , (void*)this );
}

//do gpu ray tracing
void CCUDA_RayTracingView::RayTraceOnGPU()
{
	//change the render window resolution first
	m_RenderWindow.ChangeImageResolution();

	//set the resolution of the image
	COLORREF* buf = m_RenderWindow.GetBuffer();
	int	w = m_RenderWindow.GetImageWidth();
	int h = m_RenderWindow.GetImageHeight();
	m_gpuRayTracer.BindRenderTarget( buf , w , h );

	//update the matrix for the ray tracer
	m_gpuRayTracer.SetMatrix( &m_Camera.GetViewMatrix() );

	//set scene information
	m_gpuRayTracer.SetScene( &m_Scene[m_iSelecetedSceneID] );

	//do ray tracing
	m_gpuRayTracer.RayTrace(m_iSelecetedSceneID!=3);

	//update control panel
	UpdateControlPanel( m_Scene[m_iSelecetedSceneID].GetVertexNumber() , m_gpuRayTracer.GetElapsedTime() , CString( L"GPU" ) );
}

//update control panel info
void CCUDA_RayTracingView::UpdateControlPanel( int vertexNumber , int cost , CString& device )
{
	CString vb;
	vb.Format( L"%d" , vertexNumber/3 );
	CString co;
	co.Format( L"%d" , cost );
	m_pControlPanel->UpdateInfo( vb , co , device );
}