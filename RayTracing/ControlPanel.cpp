// ControlPanel.cpp : implementation file
//

#include "stdafx.h"
#include "CUDA_RayTracing.h"
#include "ControlPanel.h"
#include "CUDA_RayTracingView.h"
#include "MainFrm.h"
#include "RenderWindow.h"

// CControlPanel

IMPLEMENT_DYNCREATE(CControlPanel, CFormView)

CControlPanel::CControlPanel()
	: CFormView(CControlPanel::IDD)
	, m_KDTreeVisible(FALSE)
{
	m_ResIndex = 0;
}

CControlPanel::~CControlPanel()
{
}

void CControlPanel::DoDataExchange(CDataExchange* pDX)
{
	CFormView::DoDataExchange(pDX);
	DDX_Check(pDX, IDC_CHECK1, m_KDTreeVisible);
	DDX_Control(pDX, IDC_COMBO1, m_SelectedScene);
}

BEGIN_MESSAGE_MAP(CControlPanel, CFormView)
	ON_BN_CLICKED(IDC_BUTTON4, &CControlPanel::OnCPURayTracing)
	ON_BN_CLICKED(IDC_CHECK1, &CControlPanel::OnEnableKDTree)
	ON_BN_CLICKED(IDC_RESOLUTION1, &CControlPanel::OnResolution1)
	ON_BN_CLICKED(IDC_RESOLUTION2, &CControlPanel::OnResolution2)
	ON_BN_CLICKED(IDC_RESOLUTION3, &CControlPanel::OnResolution3)
	ON_BN_CLICKED(IDC_BUTTON5, &CControlPanel::OnGPURayTracing)
	ON_CBN_SELCHANGE(IDC_COMBO1, &CControlPanel::OnCbnSelchangeCombo1)
	ON_BN_CLICKED(IDC_BUTTON1, &CControlPanel::OnActiveInteractiveRT)
END_MESSAGE_MAP()


// CControlPanel diagnostics

#ifdef _DEBUG
void CControlPanel::AssertValid() const
{
	CFormView::AssertValid();
}

#ifndef _WIN32_WCE
void CControlPanel::Dump(CDumpContext& dc) const
{
	CFormView::Dump(dc);
}
#endif
#endif //_DEBUG

void CControlPanel::OnInitialUpdate()
{
	CFormView::OnInitialUpdate();

	//get main frame
	CMainFrame *pFrmWnd = (CMainFrame *)(AfxGetApp()->m_pMainWnd);
	
	//get view
	m_pView = (CCUDA_RayTracingView*)pFrmWnd->GetView();

	//bind the control panel
	m_pView->m_pControlPanel = this;

	//set the 640*480 as default resolution
	((CButton *)GetDlgItem(IDC_RESOLUTION1))->SetCheck(TRUE);

	//push the scene
	m_SelectedScene.AddString( L"Cornell Box" );
	m_SelectedScene.AddString( L"Toasters" );
	m_SelectedScene.AddString( L"Chess" );
	m_SelectedScene.AddString( L"Dragon on Table" );
	m_SelectedScene.SetCurSel(0);
}

//do ray tracing on cpu here
void CControlPanel::OnCPURayTracing()
{
	//do ray tracing
	m_pView->RayTraceOnCPU();

	//show the window
	m_pView->m_RenderWindow.ShowImageWindow();
}

//do ray tracing on gpu here
void CControlPanel::OnGPURayTracing()
{
	//do ray tracing
	m_pView->RayTraceOnGPU();

	//show the window
	m_pView->m_RenderWindow.ShowImageWindow();
}

void CControlPanel::OnEnableKDTree()
{
	//update the data first
	UpdateData();

	//enable/disable kd-tree rendering
	m_pView->m_bShowKDTree = ( m_KDTreeVisible == TRUE );
}

//change the resolution of the image
void CControlPanel::OnResolution1()
{
	m_pView->m_RenderWindow.SetImageResolution( 640 , 480 );

	m_ResIndex = 0;
}

void CControlPanel::OnResolution2()
{
	m_pView->m_RenderWindow.SetImageResolution( 800 , 600 );

	m_ResIndex = 1;
}

void CControlPanel::OnResolution3()
{
	m_pView->m_RenderWindow.SetImageResolution( 1024 , 768 );

	m_ResIndex = 2;
}

//update control panel info
void CControlPanel::UpdateInfo( const CString& triNum , const CString& elapsedTime , const CString& device )
{
	//update infomation
	CButton* vbButton = ((CButton *)GetDlgItem(IDC_VERTEX_NUM));
	wstring s0(triNum);
	vbButton->SetWindowTextW( s0.c_str() );
	vbButton->Invalidate();

	vbButton = ((CButton *)GetDlgItem(IDC_COST));
	wstring s1(elapsedTime);
	vbButton->SetWindowTextW( s1.c_str() );
	vbButton->Invalidate();

	vbButton = ((CButton *)GetDlgItem(IDC_DEVICE));
	wstring s2(device);
	vbButton->SetWindowTextW( s2.c_str() );
	vbButton->Invalidate();
}

//update the scene
void CControlPanel::OnCbnSelchangeCombo1()
{
	//update the data
	this->UpdateData( 0 );

	//update the scene
	m_pView->m_iSelecetedSceneID = m_SelectedScene.GetCurSel();

	//update infomation
	CButton* vbButton = ((CButton *)GetDlgItem(IDC_VERTEX_NUM));
	CString str;
	str.Format( L"%d" , m_pView->m_Scene[m_pView->m_iSelecetedSceneID].GetVertexNumber()/3 );
	wstring s0(str);
	vbButton->SetWindowTextW( s0.c_str() );
	vbButton->Invalidate();

	vbButton = ((CButton *)GetDlgItem(IDC_COST));
	vbButton->SetWindowTextW( L"" );
	vbButton->Invalidate();

	vbButton = ((CButton *)GetDlgItem(IDC_DEVICE));
	vbButton->SetWindowTextW( L"" );
	vbButton->Invalidate();
}

//active interactive ray tracing
void CControlPanel::OnActiveInteractiveRT()
{
	if( m_ResIndex != 0 )
	{
		if( IDYES != MessageBox( L"It's highly not recommanded for ray tracing on large resolution, because it will be very slow, even stuck. Are you sure you want to continue?" , L"Warning" , MB_YESNO ) )
			return;
	}

	//disable the buttons first
	EnableAllComponent( FALSE );

	//disable tool bar in the render window
	m_pView->m_RenderWindow.EnableToolBar( FALSE );

	//show the image first
	m_pView->m_RenderWindow.ShowImageWindow();

	//enable interactive ray tracing
	m_pView->m_bEnableInteractiveRT = true;
}

//enable all component
void CControlPanel::EnableAllComponent( BOOL enable )
{
	((CButton *)GetDlgItem(IDC_BUTTON1))->EnableWindow( enable );
	((CButton *)GetDlgItem(IDC_BUTTON4))->EnableWindow( enable );
	((CButton *)GetDlgItem(IDC_BUTTON5))->EnableWindow( enable );
	((CButton *)GetDlgItem(IDC_CHECK1))->EnableWindow( enable );
	((CButton *)GetDlgItem(IDC_RESOLUTION1))->EnableWindow( enable );
	((CButton *)GetDlgItem(IDC_RESOLUTION2))->EnableWindow( enable );
	((CButton *)GetDlgItem(IDC_RESOLUTION3))->EnableWindow( enable );
	((CButton *)GetDlgItem(IDC_COMBO1))->EnableWindow( enable );
}