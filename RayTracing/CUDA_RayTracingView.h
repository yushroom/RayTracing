/*
 *	Programmer:	Jiayin Cao
 *
 *	FileName:	CUDA_RayTracingView.h
 */

#pragma once

//include the headers
#include "CUDA_RayTracingDoc.h"
#include <d3d9.h>
#include <d3dx9.h>
#include "Camera.h"
#include "CustomMesh.h"
#include "CustomScene.h"
#include "RenderWindow.h"
#include "CPURayTracer.h"
#include "GPURayTracer.h"
#include "ControlPanel.h"
#include "LoadingWindow.h"

/////////////////////////////////////////////////////////////////////////////
class CCUDA_RayTracingView : public CView
{
//private method
public:
	//Initialize default value
	void	InitializeDefault();
	//initialize direct3d
	bool	InitializeD3D();
	//release direct3d
	void	ReleaseD3D();
	//Render the scene
	void	Render();
	//Update the logic
	void	Update();
	//Restore device
	bool	RestoreDevice();
	//Update projection matrix
	void	UpdateProjectionMatrix( int cx , int cy );
	//do cpu ray tracing
	void	RayTraceOnCPU();
	//do gpu ray tracing
	void	RayTraceOnGPU();
	//update control panel info
	void	UpdateControlPanel(int vertexNumber , int cost , CString& device);

	//static thread function
	static void CPURayTracingThread( void* );

private:
	//the direct3d
	LPDIRECT3D9				m_lpD3D;
	//the device for d3d
	LPDIRECT3DDEVICE9		m_lpd3dDevice;
	//the structure
	D3DPRESENT_PARAMETERS	m_parm;

	//the projection matrix
	D3DXMATRIX		m_ProjMatrix;

	//the camera control
	Camera			m_Camera;

	//the cursor state
	POINT		m_CursorPosition;
	bool		m_bLeftButton;
	bool		m_bMoveCamera;
	int			m_iWheelPos;

	//whether the device is lost
	bool		m_bDeviceLost;

	//the scene
	CustomScene		m_Scene[4];
	//the selected scene number
	int				m_iSelecetedSceneID;

	//kd-tree level
	int			m_iKDTreeLevel;

	//whether kd-tree is visible
	bool		m_bShowKDTree;

	//the rendering window
	CRenderWindow	m_RenderWindow;

	//cpu ray tracer
	CPURayTracer	m_cpuRayTracer;
	//gpu ray tracer
	GPURayTracer	m_gpuRayTracer;

	//the maxium resolution for the client
	POINT	m_iMaxResolution;

	//the pointer to control panel
	CControlPanel* m_pControlPanel;

	//the loading window
	CLoadingWindow	m_LoadWindow;

	//enable interactive ray tracing
	bool	m_bEnableInteractiveRT;
	bool	m_bShowNextFrame;

//set control panel as friend
	friend class CControlPanel;
	friend class CRenderWindow;

protected:
	//Constructor and destructor
	CCUDA_RayTracingView();
	DECLARE_DYNCREATE(CCUDA_RayTracingView)

// Attributes
public:
	CCUDA_RayTracingDoc* GetDocument() const;

// Operations
public:

// Overrides
public:
	virtual void OnDraw(CDC* pDC);  // overridden to draw this view
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);

// Implementation
public:
	virtual ~CCUDA_RayTracingView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()
public:
	virtual void OnInitialUpdate();
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnDestroy();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
	afx_msg BOOL OnMouseWheel(UINT nFlags, short zDelta, CPoint pt);
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
};

#ifndef _DEBUG  // debug version in CUDA_RayTracingView.cpp
inline CCUDA_RayTracingDoc* CCUDA_RayTracingView::GetDocument() const
   { return reinterpret_cast<CCUDA_RayTracingDoc*>(m_pDocument); }
#endif

