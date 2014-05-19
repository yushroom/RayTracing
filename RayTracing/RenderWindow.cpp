// RenderWindow.cpp : implementation file
//

#include "stdafx.h"
#include "CUDA_RayTracing.h"
#include "RenderWindow.h"
#include "MainFrm.h"

// CRenderWindow dialog

IMPLEMENT_DYNAMIC(CRenderWindow, CDialog)

CRenderWindow::CRenderWindow(CWnd* pParent /*=NULL*/)
	: CDialog(CRenderWindow::IDD, pParent) , m_ToolBarHeight( 32 )
{

}

CRenderWindow::~CRenderWindow()
{
	//release the memory
	ReleaseMemory();
}

void CRenderWindow::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CRenderWindow, CDialog)
	ON_WM_PAINT()
	ON_WM_SIZE()
	ON_COMMAND( ID_SAVEIMAGE , CRenderWindow::OnSaveImage )
	ON_COMMAND( ID_RED_CHANNEL , CRenderWindow::OnRedChannel )
	ON_COMMAND( ID_BLUE_CHANNEL , CRenderWindow::OnBlueChannel )
	ON_COMMAND( ID_GREEN_CHANNEL , CRenderWindow::OnGreenChannel )
	ON_WM_CLOSE()
END_MESSAGE_MAP()


// CRenderWindow message handlers

BOOL CRenderWindow::OnInitDialog()
{
	CDialog::OnInitDialog();

	//load toolbar
	m_ToolBar.Create( this );
	m_ToolBar.LoadToolBar( IDR_IMAGETOOLBAR );

	//set default value
	InitializeDefault();

	return TRUE;
}

//draw the window
void CRenderWindow::OnPaint()
{
	//show the window
	ShowImageWindow( false );
}

//draw the image window
void CRenderWindow::ShowImageWindow( bool show )
{
	CPaintDC dc(this);

	//draw the tool bar
	m_ToolBar.ShowWindow( SW_SHOWNORMAL );

	//set tool bar position
	m_ToolBar.SetWindowPos( NULL , 0 , 0 , m_iWidth , m_ToolBarHeight , SWP_SHOWWINDOW );

	//show the image
	ShowImage();

	if( show )
		ShowWindow( SW_SHOWNORMAL );
}

//change the size of the window
void CRenderWindow::OnSize(UINT nType, int cx, int cy)
{
	CDialog::OnSize(nType, cx, cy);

	//update the width of the window
	m_iWidth = cx;

	//get the image dialog
	CStatic* imageborder = (CStatic*)GetDlgItem( IDC_IMAGE );
	if( imageborder )
		imageborder->SetWindowPos( NULL , 0 , m_ToolBarHeight , cx , cy - m_ToolBarHeight , SWP_SHOWWINDOW );

	//update the window
	InvalidateRect(NULL);
}

//initialize default value
void CRenderWindow::InitializeDefault()
{
	m_pImageBuffer = 0;
	m_pBackBuffer = 0;
	m_bCPURayTracing = 0;
	
	//set default image resolution
	m_iDestImageWidth = 640;
	m_iDestImageHeight = 480;
	m_iCurImageWidth = 0;
	m_iCurImageHeight = 0;

	m_bRedChannelEnable = true;
	m_bBlueChannelEnable = true;
	m_bGreenChannelEnable = true;
}

//show the image
void CRenderWindow::ShowImage()
{
	//update image resolution
	ChangeImageResolution();

	//present the buffer
	PresentBuffer();

	//image item
	CWnd* pWnd = GetDlgItem( IDC_IMAGE );

	//get device content
	CDC* pDC = pWnd->GetDC();

	//set the bitmat header
	BITMAPINFOHEADER header;
	memset( &header , 0 , sizeof( header ) );

	header.biWidth = m_iCurImageWidth;
	header.biHeight = m_iCurImageHeight;
	header.biSize = sizeof( header );
	header.biPlanes = 1;
	header.biBitCount = 32;
	header.biCompression = BI_RGB;

	//bitmap info
	BITMAPINFO bmi;
	bmi.bmiHeader = header;

	//draw the image
	SetDIBitsToDevice(	pDC->GetSafeHdc() , 2 , 0 , header.biWidth , header.biHeight , 0 , 0 , 0 , 
						header.biHeight , m_pImageBuffer , &bmi , DIB_RGB_COLORS );

	//release device content
	ReleaseDC( pDC );
}

//set image resolution
void CRenderWindow::SetImageResolution( int w , int h )
{
	m_iDestImageWidth = w;
	m_iDestImageHeight = h;
}

//release the memory
void CRenderWindow::ReleaseMemory()
{
	SAFE_DELETEARRAY( m_pImageBuffer );
	SAFE_DELETEARRAY( m_pBackBuffer );
}

//change image resolution
void CRenderWindow::ChangeImageResolution()
{
	//set the window position
	SetWindowPos( NULL , 0 , 0 , m_iDestImageWidth + 10 , m_iDestImageHeight + 50 , SWP_NOMOVE );

	if( m_iDestImageWidth == m_iCurImageWidth &&
		m_iDestImageHeight == m_iCurImageHeight )
		return;

	//delete the previous memory
	SAFE_DELETEARRAY( m_pImageBuffer );
	SAFE_DELETEARRAY( m_pBackBuffer );

	//allocate new memory
	m_pImageBuffer = new COLORREF[ m_iDestImageWidth * m_iDestImageHeight ];
	m_pBackBuffer = new COLORREF[ m_iDestImageWidth * m_iDestImageHeight ];

	//update the image size
	m_iCurImageWidth = m_iDestImageWidth;
	m_iCurImageHeight = m_iDestImageHeight;
}

//save the image
void CRenderWindow::SaveImage(LPCWSTR saveFileAddName , COLORREF* imageData , int width , int height)
{
	//the size
	int bytes = width * height * 4;

	//set the bitmap header
	BITMAPINFOHEADER header;
	memset( &header , 0 , sizeof( header ) );
	header.biWidth = width; 
	header.biHeight = height; 
	header.biSize = sizeof(BITMAPINFOHEADER); 
	header.biPlanes = 1; 
	header.biBitCount = 32;
	header.biCompression = BI_RGB;

	BITMAPFILEHEADER bmfh;
	ZeroMemory(&bmfh , sizeof(BITMAPFILEHEADER));
	*((char *)&bmfh.bfType) = 'B';
	*(((char *)&bmfh.bfType) + 1) = 'M';
	*(((char *)&bmfh.bfType) + 2) = 'P';
	bmfh.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
	bmfh.bfSize = bmfh.bfOffBits + bytes;

	CFile file;
	if(file.Open(saveFileAddName , CFile::modeWrite | CFile::modeCreate))
	{
		file.Write(&bmfh , sizeof(BITMAPFILEHEADER));
		file.Write(&header , sizeof(BITMAPINFOHEADER));
		file.Write(imageData , bytes);
		file.Close();
	}
}

void CRenderWindow::OnSaveImage()
{
	//popup a file dialog
	CFileDialog dialog( false , 0 , 0 , 2 , L".bmp File(*.bmp)|*.bmp||");

	//popup the dialog
	if( dialog.DoModal() == IDOK )
	{
		//get the file name
		CString fileName(dialog.GetPathName());

		//check if there is extension in the name
		int length = fileName.GetLength();
		if( length < 4 || -1 == fileName.Find( L".bmp" , length - 4 ) )
			fileName.AppendFormat( L".bmp" );

		//get the file name
		SaveImage( fileName , m_pImageBuffer , m_iCurImageWidth , m_iCurImageHeight );
	}
}

//change the channel state
void CRenderWindow::OnRedChannel()
{
	//change red channel state
	m_bRedChannelEnable = !m_bRedChannelEnable;

	//update the window
	InvalidateRect( NULL , false );
}

void CRenderWindow::OnGreenChannel()
{
	//change red channel state
	m_bGreenChannelEnable = !m_bGreenChannelEnable;

	//update the window
	InvalidateRect( NULL , false );
}

void CRenderWindow::OnBlueChannel()
{
	//change red channel state
	m_bBlueChannelEnable = !m_bBlueChannelEnable;

	//update the window
	InvalidateRect( NULL , false );
}

//present buffer
void CRenderWindow::PresentBuffer()
{
	//the mask
	DWORD mask = 0xffffffff;

	if( m_bRedChannelEnable == false )
		mask &= 0xff00ffff;
	if( m_bGreenChannelEnable == false )
		mask &= 0xffff00ff;
	if( m_bBlueChannelEnable == false )
		mask &= 0xffffff00;

	for( int i = 0 ; i < m_iCurImageWidth * m_iCurImageHeight ; i++ )
		m_pImageBuffer[i] = m_pBackBuffer[i] & mask;
}

//get the buffer of the image
COLORREF* CRenderWindow::GetBuffer()
{
	return m_pBackBuffer;
}

//get the resolution of the image
int	CRenderWindow::GetImageWidth()
{
	return m_iCurImageWidth;
}
int	CRenderWindow::GetImageHeight()
{
	return m_iCurImageHeight;
}

//clear the buffer
void CRenderWindow::ClearBuffer()
{
	for( int i = 0 ; i < m_iCurImageHeight * m_iCurImageWidth ; i++ )
		m_pBackBuffer[i] = 0;
}

//set cpu ray tracing
void CRenderWindow::SetCPURayTrace( bool enable )
{
	m_bCPURayTracing = true;
}

void CRenderWindow::OnCancel()
{
	//stop ray tracing
	StopRayTracing();

	CDialog::OnCancel();
}

void CRenderWindow::OnOK()
{
	//stop ray tracing
	StopRayTracing();

	CDialog::OnOK();
}

//refresh the window
void CRenderWindow::Refresh()
{
	//refresh the window
	Invalidate();
}

//on close
void CRenderWindow::OnClose()
{
	//stop ray tracing
	StopRayTracing();

	CDialog::OnClose();
}

//stop RT
void CRenderWindow::StopRayTracing()
{
	//get main frame
	CMainFrame *pFrmWnd = (CMainFrame *)(AfxGetApp()->m_pMainWnd);
	
	//get view
	CCUDA_RayTracingView* view = (CCUDA_RayTracingView*)pFrmWnd->GetView();

	//force the cpu to stop 
	if( m_bCPURayTracing )
		view->m_cpuRayTracer.ForceStop( true );

	//disable interactive ray tracing
	view->m_bEnableInteractiveRT = false;

	//enable the buttons
	view->m_pControlPanel->EnableAllComponent( TRUE );

	//enable the tool bar
	EnableToolBar( TRUE );
}

//enable tool bar
void CRenderWindow::EnableToolBar( BOOL enable )
{
	//enable the tool bar
	m_ToolBar.EnableWindow( enable );
}