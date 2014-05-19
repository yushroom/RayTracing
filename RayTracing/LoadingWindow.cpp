/*
 *	FileName:	LoadingWindow.h
 *
 *	Programmer:	Jiayin Cao
 */

//include the headers
#include "stdafx.h"
#include "CUDA_RayTracing.h"
#include "LoadingWindow.h"


// CLoadingWindow
IMPLEMENT_DYNAMIC(CLoadingWindow, CDialog)

//constructor and destructor
CLoadingWindow::CLoadingWindow()
{
	memset( m_Str , 0 , sizeof( WCHAR ) * 1024 );
}

CLoadingWindow::~CLoadingWindow()
{
}


//the message map
BEGIN_MESSAGE_MAP(CLoadingWindow, CDialog)
END_MESSAGE_MAP()

//update progress dialog
void CLoadingWindow::UpdateProgress( int prog )
{
	//set the value
	wsprintf( m_Str , L"Loading.... (%d%%)" , prog );

	//update progress
	UpdateProgress();
}

//update progress
void CLoadingWindow::UpdateProgress()
{
	//update the text
	CButton* textCtl = (CButton*)GetDlgItem( IDC_LOADING );
	textCtl->SetWindowTextW( m_Str );

	//update the dialog
	Invalidate();
}