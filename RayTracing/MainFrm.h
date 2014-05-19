// MainFrm.h : interface of the CMainFrame class
//


#pragma once
#include "MySplitterWnd.h"
#include "ControlPanel.h"
#include "CUDA_RayTracingView.h"

class CMainFrame : public CFrameWnd
{
	
protected: // create from serialization only
	CMainFrame();
	DECLARE_DYNCREATE(CMainFrame)

// Attributes
public:

// Operations
public:

// Overrides
public:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);

// Implementation
public:
	virtual ~CMainFrame();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

	//get view
	CCUDA_RayTracingView*	GetView();

protected:  // control bar embedded members
	CStatusBar		m_wndStatusBar;
//	CToolBar		m_wndToolBar;
	CMySplitterWnd	m_WndSplitter;

// Generated message map functions
protected:
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	DECLARE_MESSAGE_MAP()
	virtual BOOL OnCreateClient(LPCREATESTRUCT lpcs, CCreateContext* pContext);
};


