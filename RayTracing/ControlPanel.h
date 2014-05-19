#pragma once

#include <String.h>
#include "afxwin.h"
// CControlPanel form view


class CCUDA_RayTracingView;

////////////////////////////////////////////////////////////////////
class CControlPanel : public CFormView
{
//public
public:
	//update control panel info
	void	UpdateInfo( const CString& vertexNum , const CString& elapsedTime , const CString& device );

	//enable all component
	void	EnableAllComponent( BOOL enable );

//private field
private:
	//the view panel
	CCUDA_RayTracingView* m_pView;

	//whether kd-tree is visible
	BOOL m_KDTreeVisible;

	//the resolution index
	int	m_ResIndex;

protected:
	//constructor and destructor
	CControlPanel();
	virtual ~CControlPanel();

public:
	enum { IDD = IDD_CONTROLPANEL };
#ifdef _DEBUG
	virtual void AssertValid() const;
#ifndef _WIN32_WCE
	virtual void Dump(CDumpContext& dc) const;
#endif
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	DECLARE_MESSAGE_MAP()

public:
	//initliaze
	virtual void OnInitialUpdate();

	DECLARE_DYNCREATE(CControlPanel)

//message handler
public:
	afx_msg void OnCPURayTracing();
	afx_msg void OnEnableKDTree();
	afx_msg void OnResolution1();
	afx_msg void OnResolution2();
	afx_msg void OnResolution3();
	afx_msg void OnGPURayTracing();
	CComboBox m_SelectedScene;
	afx_msg void OnCbnSelchangeCombo1();
	afx_msg void OnActiveInteractiveRT();
};


